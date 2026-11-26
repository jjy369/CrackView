# feature/make_sfm_features.py
"""
Stage 2: Extract per-view features from manifest CSV for the PPO state.

Feature per image-pose pair — 397 dims total:
  DINO    (384)  vit_small_patch16_224 visual embedding
  Quality (  4)  sharpness, exposure mean/std, saturation ratio
  Geo     (  9)  camera center(3) + forward_dir(3) + up_dir(3)

Geo feature (9 dims) — no intrinsics required:
  [0:3]  C = -R^T t    camera center in world space (absolute)
  [3:6]  R[2,:]         forward direction, optical axis in world space
  [6:9]  R[1,:]         up direction in world space

  Since all cameras share identical intrinsics, frustum corner directions
  in camera space are constant vectors. World-space corners (C + R^T @ d_i)
  are a fixed linear map of (C, R) — no extra information vs. (C, R) directly.
  forward + up (6 dims) capture the full 3-DOF rotation without the 6
  redundant orthogonality constraints present in R_flat (9 dims).

  Values are stored as absolute world coordinates — no normalization.
  The PPO policy always sees all N candidates together, so absolute
  coords are directly comparable across views within the same scene.

Quality feature (4 dims):
  [0] log(Laplacian variance + 1) / 10   sharpness  (↑ = sharper)
  [1] gray mean / 255                     exposure level
  [2] gray std  / 128                     dynamic range
  [3] saturated pixel ratio               fraction of pixels <10 or >245

NPZ fields:
  scene_ids    : (N,) str
  pair_keys    : (N,) str    "{scene}||{image_stem}"
  image_paths  : (N,) str
  camera_paths : (N,) str
  features     : (N, 397)   DINO(384) + Quality(4) + Geo(9)
  extrinsics   : (N, 12)    raw [R|t] 3×4 flattened, for downstream use

Place at:  E:\\view_subset\\rl\\feature\\make_sfm_features.py
Run from PyCharm with working directory = E:\\view_subset\\rl
    python feature/make_sfm_features.py
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════════════
# ✅ CONFIG — edit paths here
# ════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ---- inputs ----
    "MANIFEST_CSV":      "data/train.csv",
    "DINO_BACKBONE_PTH": "crack_finetune/dino_vit_crack_backbone.pth",  # from train_dino_vit.py

    # ---- outputs ----
    "OUT_FEATURES_NPZ":  "feature/features_sfm2.npz",

    # ---- DINO (must match train_dino_vit.py) ----
    "VIT_NAME":    "vit_small_patch16_224",
    "IMG_SIZE":    224,
    "BATCH_SIZE":  16,
    "NUM_WORKERS": 0,   # keep 0 on Windows; raise only on Linux
}

DINO_DIM    = 384
QUALITY_DIM = 4
GEO_DIM     = 9    # camera_center(3) + forward_dir(3) + up_dir(3)
FEATURE_DIM = DINO_DIM + QUALITY_DIM + GEO_DIM   # 397
# ════════════════════════════════════════════════════════════════════════


# ────────────────────────────────────────────────────────────────────────
# Camera file parsing  (extrinsic only, intrinsic/depth blocks ignored)
# ────────────────────────────────────────────────────────────────────────
def _parse_camera(path: str) -> np.ndarray:
    """
    Read camera .txt and return the (4, 4) extrinsic matrix.
    Intrinsic and depth blocks are intentionally not read.

    File format:
        extrinsic
        r00 r01 r02 t0
        r10 r11 r12 t1
        r20 r21 r22 t2
        0   0   0   1

        intrinsic   ← ignored
        ...
        depth_min   ← ignored
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if "extrinsic" not in lines:
        raise ValueError(f"'extrinsic' block not found: {path}")

    ei  = lines.index("extrinsic")
    ext = np.array(
        [[float(x) for x in ln.split()] for ln in lines[ei + 1: ei + 5]],
        dtype=np.float32,
    )   # (4, 4)
    return ext


# ────────────────────────────────────────────────────────────────────────
# Geo feature — 9 dims, no intrinsics
# ────────────────────────────────────────────────────────────────────────
def _geo_feature(ext: np.ndarray) -> np.ndarray:
    """
    9-dim geometric feature from 4×4 extrinsic matrix.

    ext layout:
        [ R (3×3) | t (3×1) ]
        [ 0  0  0 |    1    ]

    Outputs:
      [0:3]  C = -R^T @ t    camera center in world space
      [3:6]  R[2, :]          forward direction (optical axis, world space)
      [6:9]  R[1, :]          up direction (world space)

    forward + up fully determine the camera orientation (3 DOF of SO(3))
    without the 6 redundant constraints in R_flat.
    right = R[0,:] is omitted — recoverable as cross(forward, up) if needed.
    """
    R = ext[:3, :3]          # (3, 3) rotation submatrix
    t = ext[:3, 3]            # (3,)   translation vector
    C       = (-R.T @ t).astype(np.float32)   # camera center in world space
    forward = R[2, :].astype(np.float32)      # optical axis direction
    up      = R[1, :].astype(np.float32)      # up direction
    return np.concatenate([C, forward, up])   # (9,)


# ────────────────────────────────────────────────────────────────────────
# Quality token — 4 dims
# ────────────────────────────────────────────────────────────────────────
def _quality_token(bgr: np.ndarray) -> np.ndarray:
    """
    4-dim image quality feature from raw BGR image.

    Gives the PPO policy a direct signal to deprioritise blurry or
    badly-exposed views that would degrade SfM feature matching.

    [0] log(Laplacian variance + 1) / 10   sharpness  (↑ = sharper)
    [1] gray mean  / 255                   exposure level
    [2] gray std   / 128                   dynamic range
    [3] saturated pixel ratio              fraction of pixels <10 or >245
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    lap_var   = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness = float(np.log1p(lap_var) / 10.0)
    exp_mean  = float(gray.mean()  / 255.0)
    exp_std   = float(gray.std()   / 128.0)
    sat_ratio = float(((gray < 10) | (gray > 245)).mean())

    return np.array([sharpness, exp_mean, exp_std, sat_ratio], dtype=np.float32)


# ────────────────────────────────────────────────────────────────────────
# DINO DataLoader
# ────────────────────────────────────────────────────────────────────────
class _ImageDataset(Dataset):
    def __init__(self, paths: List[str], img_size: int):
        self.paths = paths
        self.tfm   = transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std =(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        return self.tfm(Image.open(p).convert("RGB")), p


@torch.no_grad()
def _extract_dino_quality(
    paths:       List[str],
    vit:         torch.nn.Module,
    device:      str,
    batch_size:  int,
    num_workers: int,
    img_size:    int,
) -> np.ndarray:
    """Returns (N, DINO_DIM + QUALITY_DIM) = (N, 388)."""
    ds  = _ImageDataset(paths, img_size)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=False,
                     num_workers=num_workers,
                     pin_memory=(device == "cuda"))
    out = []

    for x_batch, path_batch in tqdm(dl, desc="  DINO+quality", leave=False):
        h = vit(x_batch.to(device, non_blocking=True)).cpu().numpy().astype(np.float32)

        quality = []
        for p in path_batch:
            bgr = cv2.imread(str(p))
            if bgr is None:
                raise RuntimeError(f"cv2.imread failed: {p}")
            quality.append(_quality_token(bgr))

        out.append(np.concatenate([h, np.stack(quality)], axis=1))

    return np.concatenate(out, axis=0)   # (N, 388)


# ────────────────────────────────────────────────────────────────────────
# Manifest → { scene: [(img_path, cam_path), ...] }
# ────────────────────────────────────────────────────────────────────────
def _collect_scene_pairs(csv_path: str) -> Dict[str, List[Tuple[str, str]]]:
    df = pd.read_csv(csv_path)
    scene_pairs: Dict[str, Dict[str, Tuple[str, str]]] = {}

    for _, row in df.iterrows():
        scene = str(row["scene_name"])
        imgs  = str(row["image_paths"]).split(";")
        cams  = str(row["camera_paths"]).split(";")
        pool  = scene_pairs.setdefault(scene, {})
        for ip, cp in zip(imgs, cams):
            ip, cp = ip.strip(), cp.strip()
            key    = f"{scene}||{Path(ip).stem}"
            pool.setdefault(key, (ip, cp))

    return {s: list(d.values()) for s, d in scene_pairs.items()}


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device      : {device}")
    print(f"Feature dim : {FEATURE_DIM}  "
          f"(DINO={DINO_DIM} + Quality={QUALITY_DIM} + Geo={GEO_DIM})")

    # ── 1. Manifest ──────────────────────────────────────────────────
    print(f"\nReading manifest : {CONFIG['MANIFEST_CSV']}")
    scene_pairs = _collect_scene_pairs(CONFIG["MANIFEST_CSV"])
    total_pairs = sum(len(v) for v in scene_pairs.values())
    pool_sizes  = sorted(len(v) for v in scene_pairs.values())
    print(f"  Scenes      : {len(scene_pairs)}")
    print(f"  Total pairs : {total_pairs}")
    print(f"  Pool size   : min={pool_sizes[0]}  "
          f"median={pool_sizes[len(pool_sizes) // 2]}  max={pool_sizes[-1]}")

    # ── 2. DINO backbone ─────────────────────────────────────────────
    print(f"\nLoading DINO backbone : {CONFIG['DINO_BACKBONE_PTH']}")
    vit = timm.create_model(
        CONFIG["VIT_NAME"],
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,   # must match build_vit_backbone in train_dino_vit.py
    ).to(device)
    state = torch.load(CONFIG["DINO_BACKBONE_PTH"], map_location="cpu")
    vit.load_state_dict(state, strict=True)
    vit.eval()
    print(f"  Embed dim : {vit.num_features}  (expected {DINO_DIM})")

    # ── 3. Per-scene feature extraction ──────────────────────────────
    all_scene_ids, all_pair_keys = [], []
    all_img_paths, all_cam_paths = [], []
    all_features,  all_extrinsics = [], []

    for scene_id, pairs in tqdm(scene_pairs.items(), desc="Scenes"):
        img_paths = [p[0] for p in pairs]
        cam_paths = [p[1] for p in pairs]
        pair_keys = [f"{scene_id}||{Path(p[0]).stem}" for p in pairs]
        n         = len(pairs)

        # 3a. DINO + quality  →  (n, 388)
        dino_quality = _extract_dino_quality(
            img_paths, vit, device,
            CONFIG["BATCH_SIZE"], CONFIG["NUM_WORKERS"], CONFIG["IMG_SIZE"],
        )

        # 3b. Per-view geo features + raw extrinsics
        geo_list = []
        exts_raw = []
        for cp in cam_paths:
            ext = _parse_camera(cp)
            geo_list.append(_geo_feature(ext))
            exts_raw.append(ext[:3].reshape(-1).astype(np.float32))  # (12,)

        geo_feats = np.stack(geo_list, axis=0)   # (n, 9)
        exts_raw  = np.stack(exts_raw, axis=0)   # (n, 12)

        # 3c. Full feature = [DINO+quality | geo]  →  (n, 397)
        full_feats = np.concatenate([dino_quality, geo_feats], axis=1)
        assert full_feats.shape[1] == FEATURE_DIM, \
            f"Feature dim mismatch: {full_feats.shape[1]} != {FEATURE_DIM}"

        all_scene_ids.extend([scene_id] * n)
        all_pair_keys.extend(pair_keys)
        all_img_paths.extend(img_paths)
        all_cam_paths.extend(cam_paths)
        all_features.append(full_feats)
        all_extrinsics.append(exts_raw)

    features   = np.concatenate(all_features,   axis=0)
    extrinsics = np.concatenate(all_extrinsics, axis=0)

    # ── 4. Save NPZ ───────────────────────────────────────────────────
    out_path = CONFIG["OUT_FEATURES_NPZ"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    np.savez_compressed(
        out_path,
        scene_ids    = np.array(all_scene_ids),
        pair_keys    = np.array(all_pair_keys),
        image_paths  = np.array(all_img_paths),
        camera_paths = np.array(all_cam_paths),
        features     = features,
        extrinsics   = extrinsics,
    )

    print(f"\nSaved : {out_path}")
    print(f"  features   shape : {features.shape}")
    print(f"  extrinsics shape : {extrinsics.shape}")
    print(f"\nFeature layout per row:")
    print(f"  [0:384]     DINO visual embedding")
    print(f"  [384:388]   Quality  (sharpness, exp_mean, exp_std, sat_ratio)")
    print(f"  [388:391]   Camera center  C = -R^T t  (world coords)")
    print(f"  [391:394]   Forward direction  R[2,:]  (optical axis)")
    print(f"  [394:397]   Up direction       R[1,:]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    CONFIG.update(MANIFEST_CSV=args.manifest, DINO_BACKBONE_PTH=args.backbone, OUT_FEATURES_NPZ=args.out)
    main()
