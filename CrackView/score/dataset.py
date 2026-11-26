# dataset.py
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


FINAL_GLOBAL_WEIGHT     = 0.40
FINAL_CRACK_CORE_WEIGHT = 0.60
POSE_DIM = 13


# ─────────────────────────────────────────────────────────────────────────────
# 相机解析（不变）
# ─────────────────────────────────────────────────────────────────────────────
def parse_camera_txt(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if "extrinsic" not in lines:
        raise ValueError(f"No 'extrinsic' in: {path}")
    idx = lines.index("extrinsic")
    ext_lines = lines[idx + 1: idx + 5]
    if len(ext_lines) != 4:
        raise ValueError(f"Invalid extrinsic in: {path}")
    return np.array([[float(x) for x in l.split()] for l in ext_lines], dtype=np.float32)


def build_pose_features(extrinsics, camera_center_mode="optical_center", position_scale=1.0):
    """
    13 维：R 展平、归一化相对中心、半径
    """
    Rs, positions = [], []
    for ext in extrinsics:
        R = ext[:3, :3]
        t = ext[:3, 3]
        C = t if camera_center_mode == "translation" else -R.T @ t
        Rs.append(R.reshape(-1).astype(np.float32))
        positions.append(C.astype(np.float32))

    Rs        = np.stack(Rs,        axis=0)
    positions = np.stack(positions, axis=0)
    scale     = float(position_scale) if float(position_scale) > 0 else 1.0
    raw_pos   = positions / scale
    center    = raw_pos.mean(axis=0, keepdims=True)
    rel_pos   = raw_pos - center
    radius    = max(np.linalg.norm(rel_pos, axis=1).max(), 1e-6)

    feat = np.concatenate([
        Rs,
        rel_pos / radius,
        np.full((len(extrinsics), 1), radius, dtype=np.float32),
    ], axis=1).astype(np.float32)

    assert feat.shape[1] == POSE_DIM, f"pose dim error: {feat.shape[1]}"
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# 图像变换
# ─────────────────────────────────────────────────────────────────────────────
def _build_transform(image_size, augment):
    ops = [transforms.Resize((image_size, image_size))]
    if augment:
        ops.append(transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02
            )
        ], p=0.5))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


# ─────────────────────────────────────────────────────────────────────────────
# SceneSubsetDataset
#
# 每个样本 = 一个完整场景的所有标注子集（N 个，各有 K_i 张图）。
#
# 新增：视角丢弃增强（view dropout augmentation）
#   训练时（augment=True）对每个子集随机丢掉 1~view_drop_max 张视角，
#   让模型见过不同大小、不同组合的子集，而不只是固定的 10 种标注选法。
#   这是模型能泛化到 RL 任意生成组合的关键。
#
#   丢掉视角后 GT 标签保持不变（近似假设：少几张不会剧烈改变重建质量）。
#   保证至少保留 view_drop_min 张视角（默认 5）。
# ─────────────────────────────────────────────────────────────────────────────
class SceneSubsetDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_size=224,
        augment=False,
        use_final_raw=True,
        camera_center_mode="optical_center",
        position_scale=1.0,
        view_drop_prob=0.5,    # 训练时触发视角丢弃的概率
        view_drop_max=3,       # 最多丢掉几张（视角数多时可以设大）
        view_drop_min=5,       # 丢弃后至少保留几张
    ):
        df = pd.read_csv(csv_path)
        if "scene_name" not in df.columns:
            raise ValueError("CSV 缺少 scene_name 列，请重新运行 build_manifest.py。")

        self.scenes = [
            (name, grp.reset_index(drop=True))
            for name, grp in df.groupby("scene_name")
        ]
        self.augment            = augment
        self.use_final_raw      = use_final_raw
        self.camera_center_mode = camera_center_mode
        self.position_scale     = position_scale
        self.transform          = _build_transform(image_size, augment)
        self.view_drop_prob     = view_drop_prob
        self.view_drop_max      = view_drop_max
        self.view_drop_min      = view_drop_min

    def __len__(self):
        return len(self.scenes)

    def _get_final_label(self, row, q_global, q_core):
        return float(row.get("q_final", FINAL_GLOBAL_WEIGHT * q_global + FINAL_CRACK_CORE_WEIGHT * q_core))

    def _load_row(self, row):
        """
        加载单个子集行。

        训练时（augment=True）以 view_drop_prob 的概率触发视角丢弃：
          随机丢掉 1~view_drop_max 张，保证至少剩 view_drop_min 张。
          GT 标签不变（小扰动近似假设）。

        这让模型训练时见过各种不完整的视角组合，
        推理时 RL 生成的任意子集就不再是 out-of-distribution 了。
        """
        image_paths  = str(row["image_paths"]).split(";")
        camera_paths = str(row["camera_paths"]).split(";")

        # ── 视角丢弃增强 ─────────────────────────────────────────────────
        if (self.augment
                and len(image_paths) > self.view_drop_min
                and random.random() < self.view_drop_prob):
            # 最多可以丢几张（不能让视角数低于 view_drop_min）
            max_drop = min(self.view_drop_max, len(image_paths) - self.view_drop_min)
            n_drop   = random.randint(1, max(1, max_drop))
            # 随机选要丢弃的视角索引
            drop_idx = set(random.sample(range(len(image_paths)), n_drop))
            image_paths  = [p for i, p in enumerate(image_paths)  if i not in drop_idx]
            camera_paths = [p for i, p in enumerate(camera_paths) if i not in drop_idx]

        images, extrinsics = [], []
        for ip, cp in zip(image_paths, camera_paths):
            images.append(self.transform(Image.open(ip).convert("RGB")))
            extrinsics.append(parse_camera_txt(cp))

        images_t = torch.stack(images)
        poses_t  = torch.tensor(
            build_pose_features(extrinsics, self.camera_center_mode, self.position_scale),
            dtype=torch.float32,
        )

        q_global = float(row["global_q_geo_raw"])
        q_core   = float(row["crack_core_q_geo_raw"])
        labels_t = torch.tensor(
            [q_global, q_core, self._get_final_label(row, q_global, q_core)],
            dtype=torch.float32,
        )
        return images_t, poses_t, labels_t

    def __getitem__(self, idx):
        scene_name, scene_df = self.scenes[idx]
        subsets_images, subsets_poses, subsets_labels = [], [], []
        subset_names, num_views = [], []

        for _, row in scene_df.iterrows():
            imgs, poses, labels = self._load_row(row)
            subsets_images.append(imgs)
            subsets_poses.append(poses)
            subsets_labels.append(labels)
            subset_names.append(str(row["sample_id"]))
            num_views.append(imgs.shape[0])

        return {
            "subsets_images": subsets_images,               # list[Tensor[K_i,3,H,W]]
            "subsets_poses":  subsets_poses,                # list[Tensor[K_i,13]]
            "subsets_labels": torch.stack(subsets_labels),  # [N, 3]
            "subset_names":   subset_names,
            "scene_name":     scene_name,
            "num_views":      num_views,
        }


# ─────────────────────────────────────────────────────────────────────────────
# scene_subset_collate_fn（不变，供 DataLoader 使用）
#
# 两层 padding：
#   1. 视角内：不同子集 K_i 不同 → pad 到 batch 内 max_K
#   2. 场景间：不同场景 N 不同  → pad 到 batch 内 max_N
#
# 视角丢弃增强后，同一 batch 内 max_K 可能比原来小，这是正常的。
# ─────────────────────────────────────────────────────────────────────────────
def scene_subset_collate_fn(batch):
    B     = len(batch)
    max_n = max(len(item["subsets_images"]) for item in batch)
    max_k = max(img.shape[0] for item in batch for img in item["subsets_images"])
    _, C, H, W = batch[0]["subsets_images"][0].shape
    pose_dim   = batch[0]["subsets_poses"][0].shape[1]

    padded_images = torch.zeros(B, max_n, max_k, C, H, W)
    padded_poses  = torch.zeros(B, max_n, max_k, pose_dim)
    view_masks    = torch.ones(B, max_n, 1 + max_k, dtype=torch.bool)
    subset_masks  = torch.ones(B, max_n,             dtype=torch.bool)
    labels        = torch.zeros(B, max_n, 3)
    all_subset_names, all_scene_names = [], []

    for b, item in enumerate(batch):
        N_b = len(item["subsets_images"])
        all_subset_names.append(item["subset_names"])
        all_scene_names.append(item["scene_name"])
        for n in range(N_b):
            img  = item["subsets_images"][n]
            pose = item["subsets_poses"][n]
            K_i  = img.shape[0]
            padded_images[b, n, :K_i] = img
            padded_poses[b,  n, :K_i] = pose
            view_masks[b, n, 0]           = False
            view_masks[b, n, 1: 1 + K_i] = False
            subset_masks[b, n]            = False
        labels[b, :N_b] = item["subsets_labels"]

    return {
        "images":        padded_images,     # [B, max_N, max_K, C, H, W]
        "poses":         padded_poses,      # [B, max_N, max_K, 13]
        "view_masks":    view_masks,        # [B, max_N, 1+max_K]
        "subset_masks":  subset_masks,      # [B, max_N]
        "labels":        labels,            # [B, max_N, 3]
        "subset_names":  all_subset_names,
        "scene_names":   all_scene_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# single_subset_collate_fn（新增）
#
# 供 RL 调用 forward_single 时使用。
# 每个样本是单个子集（[K,3,H,W] + [K,13]），batch 后输出
#   images [B, max_K, 3, H, W]、poses [B, max_K, 13]、view_masks [B, 1+max_K]
# ─────────────────────────────────────────────────────────────────────────────
def single_subset_collate_fn(batch):
    """
    batch: list of dict，每个 dict 含 images [K,3,H,W]、poses [K,13]、
           可选 labels [3]、sample_id str

    用于 DataLoader 在推理或 RL reward 计算时批量调用 forward_single。
    """
    max_k    = max(item["images"].shape[0] for item in batch)
    _, C, H, W = batch[0]["images"].shape
    pose_dim = batch[0]["poses"].shape[1]
    B        = len(batch)

    padded_images = torch.zeros(B, max_k, C, H, W)
    padded_poses  = torch.zeros(B, max_k, pose_dim)
    view_masks    = torch.ones(B, 1 + max_k, dtype=torch.bool)

    for b, item in enumerate(batch):
        K = item["images"].shape[0]
        padded_images[b, :K] = item["images"]
        padded_poses[b,  :K] = item["poses"]
        view_masks[b, 0]          = False
        view_masks[b, 1: 1 + K]  = False

    out = {
        "images":     padded_images,   # [B, max_K, C, H, W]
        "poses":      padded_poses,    # [B, max_K, 13]
        "view_masks": view_masks,      # [B, 1+max_K]
    }

    if "labels" in batch[0]:
        out["labels"] = torch.stack([item["labels"] for item in batch])
    if "sample_id" in batch[0]:
        out["sample_id"] = [item["sample_id"] for item in batch]

    return out