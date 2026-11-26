# build_manifest.py
import csv
import random
from pathlib import Path

import pandas as pd


# ── 路径配置 ──────────────────────────────────────────────────────────────────
PRED_ROOT       = Path(r"E:\view_subset\subdata_new\25")
POINT_SCORE_CSV = Path(r"E:\view_subset\subdata_new\point_score.csv")

OUTPUT_DIR  = Path("data")
MANIFEST_CSV = OUTPUT_DIR / "manifest.csv"
TRAIN_CSV    = OUTPUT_DIR / "train.csv"
VAL_CSV      = OUTPUT_DIR / "val.csv"

IMAGE_DIR_NAME  = "images"
CAMERA_DIR_NAME = "cams"
IMAGE_SUFFIXES  = [".png", ".jpg", ".jpeg", ".bmp"]
CAMERA_SUFFIX   = ".txt"

VAL_RATIO   = 0.2
RANDOM_SEED = 42
NUM_Q_BINS  = 10   # 按场景 q_final 中位数分桶做分层


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def find_images(image_dir: Path):
    imgs = []
    for s in IMAGE_SUFFIXES:
        imgs.extend(image_dir.glob(f"*{s}"))
    return sorted(imgs)


def find_cameras(camera_dir: Path):
    return sorted(camera_dir.glob(f"*{CAMERA_SUFFIX}"))


# ── 数据集划分：以 scene_name 为单位 ─────────────────────────────────────────
# 原来按行（每个子集）分层，导致同一场景的 10 个子集可能一半在 train、一半在 val，
# 造成数据泄露。现在以 scene_name 为单位整体划分，保证同一场景所有子集不跨集。
def scene_based_split(rows):
    df = pd.DataFrame(rows)

    # 每个场景取 q_final 中位数作为分层依据
    scene_stats = (
        df.groupby("scene_name")["q_final"]
        .median()
        .reset_index()
        .rename(columns={"q_final": "median_q"})
        .sample(frac=1.0, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    try:
        q = min(NUM_Q_BINS, len(scene_stats))
        scene_stats["q_bin"] = pd.qcut(
            scene_stats["median_q"], q=q, labels=False, duplicates="drop"
        ).fillna(0).astype(int)
    except Exception:
        print("[WARN] 无法分桶，退回到单桶。")
        scene_stats["q_bin"] = 0

    groups = list(scene_stats.groupby("q_bin"))
    target = round(len(scene_stats) * VAL_RATIO)
    quotas = [len(g) * VAL_RATIO for _, g in groups]
    counts = [int(x) for x in quotas]
    for i in sorted(range(len(groups)), key=lambda j: quotas[j]-counts[j], reverse=True)[:target-sum(counts)]:
        counts[i] += 1
    val_scenes = []
    for (bin_id, group), count in zip(groups, counts):
        val_scenes.extend(group.sample(n=count, random_state=RANDOM_SEED+int(bin_id))["scene_name"])

    val_set = set(val_scenes)
    train_df   = df[~df["scene_name"].isin(val_set)].sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    val_df     = df[ df["scene_name"].isin(val_set)].sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    manifest_df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"  Val scenes  : {len(val_set)}")
    print(f"  Train scenes: {len(scene_stats) - len(val_set)}")
    return manifest_df.to_dict("records"), train_df.to_dict("records"), val_df.to_dict("records")


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(POINT_SCORE_CSV)
    rows = []

    for _, row in df.iterrows():
        if str(row.get("status", "")) != "ok":
            continue

        pred_folder_key = str(row["pred_folder_key"])
        sample_dir      = PRED_ROOT / pred_folder_key
        image_dir       = sample_dir / IMAGE_DIR_NAME
        camera_dir      = sample_dir / CAMERA_DIR_NAME

        if not image_dir.exists():
            print(f"[WARN] image dir not found: {image_dir}"); continue
        if not camera_dir.exists():
            print(f"[WARN] camera dir not found: {camera_dir}"); continue

        images  = find_images(image_dir)
        cameras = find_cameras(camera_dir)

        if len(images) == 0:
            print(f"[WARN] no images: {image_dir}"); continue
        if len(cameras) == 0:
            print(f"[WARN] no cameras: {camera_dir}"); continue
        if len(images) != len(cameras):
            print(f"[WARN] mismatch {pred_folder_key}: {len(images)} imgs vs {len(cameras)} cams"); continue

        by_stem = {p.stem.removesuffix("_cam"): p for p in cameras}
        if any(p.stem not in by_stem for p in images):
            raise ValueError(f"Image and camera identifiers differ in {sample_dir}")
        cameras = [by_stem[p.stem] for p in images]
        # pred_folder_key 格式为 "subset_type/scene_name"，取最后一段作为 scene_name
        scene_name = pred_folder_key.split("/")[-1]

        rows.append({
            "sample_id":           pred_folder_key,
            "scene_name":          scene_name,
            "image_paths":         ";".join(str(p) for p in images),
            "camera_paths":        ";".join(str(p) for p in cameras),
            "num_views":           len(images),
            "global_q_geo_raw":    float(row["global_q_geo_raw"]),
            "crack_core_q_geo_raw": float(row["crack_core_q_geo_raw"]),
            "q_final":             float(row["q_final"]),
        })

    if not rows:
        raise RuntimeError("No valid samples found.")

    # 视角数不固定：打印分布，由 collate_fn 在 batch 内 padding 处理
    view_counts = sorted(set(r["num_views"] for r in rows))
    print(f"View counts found: {view_counts}")
    if len(view_counts) > 1:
        print("[INFO] 视角数不固定，将由 scene_subset_collate_fn 在 batch 内 padding。")

    manifest_rows, train_rows, val_rows = scene_based_split(rows)

    fieldnames = [
        "sample_id", "scene_name", "image_paths", "camera_paths",
        "num_views", "global_q_geo_raw", "crack_core_q_geo_raw", "q_final",
    ]

    def write_csv(path, data):
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    write_csv(MANIFEST_CSV, manifest_rows)
    write_csv(TRAIN_CSV,    train_rows)
    write_csv(VAL_CSV,      val_rows)

    train_q = pd.DataFrame(train_rows)["q_final"]
    val_q   = pd.DataFrame(val_rows)["q_final"]

    print(f"\nTotal : {len(manifest_rows)}")
    print(f"Train : {len(train_rows)}  mean={train_q.mean():.4f} std={train_q.std():.4f}")
    print(f"Val   : {len(val_rows)}    mean={val_q.mean():.4f} std={val_q.std():.4f}")
    print(f"\nSaved: {TRAIN_CSV}, {VAL_CSV}, {MANIFEST_CSV}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scores", type=Path, required=True)
    opts = p.parse_args()
    PRED_ROOT, POINT_SCORE_CSV = opts.root.resolve(), opts.scores
    main()
