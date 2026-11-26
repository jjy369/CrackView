# train.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from score.dataset import SceneSubsetDataset, scene_subset_collate_fn
from score.model import SubsetScoringNet, with_composite


# ── 超参数 ────────────────────────────────────────────────────────────────────
TRAIN_CSV = "data/train.csv"
VAL_CSV   = "data/val.csv"

IMAGE_SIZE  = 224
BATCH_SIZE  = 1
NUM_EPOCHS  = 120

# Windows 上 NUM_WORKERS > 0 需要 DataLoader 在 if __name__=="__main__" 内，
# 否则子进程 reimport 主脚本会崩溃。设为 0 最稳妥。
NUM_WORKERS = 4

BACKBONE_NAME       = "resnet18"
TRAIN_BACKBONE_FROM = "layer4"
HIDDEN_DIM          = 128
VIEW_LAYERS         = 1
SCENE_LAYERS        = 2
NUM_HEADS           = 4
DROPOUT             = 0.3

HEAD_LR      = 1e-4
BACKBONE_LR  = 1e-5
WEIGHT_DECAY = 5e-4

LOSS_W_GLOBAL    = 0.4
LOSS_W_CORE      = 0.6
RANK_LOSS_WEIGHT = 0.1
TOP1_LOSS_WEIGHT = 0.2

EARLY_STOP_PATIENCE = 20
SCHEDULER_PATIENCE  = 10

ACCEPT_MARGIN = 0.05


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",            type=int, default=2024)
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints")
    parser.add_argument("--fresh",           action="store_true")
    return parser.parse_args()


def get_checkpoint_dir(root, seed):
    return Path(root) / f"seed{seed}"


def get_config(seed, pose_dim):
    return {
        "seed": int(seed), "model_type": "SubsetScoringNet",
        "pose_dim": int(pose_dim), "backbone_name": BACKBONE_NAME,
        "train_backbone_from": TRAIN_BACKBONE_FROM, "hidden_dim": HIDDEN_DIM,
        "view_layers": VIEW_LAYERS, "scene_layers": SCENE_LAYERS,
        "num_heads": NUM_HEADS, "dropout": DROPOUT,
        "head_lr": HEAD_LR, "backbone_lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY,
        "loss_w_global": LOSS_W_GLOBAL, "loss_w_core": LOSS_W_CORE,
        "rank_loss_weight": RANK_LOSS_WEIGHT,
        "top1_loss_weight": TOP1_LOSS_WEIGHT, "accept_margin": ACCEPT_MARGIN,
        "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def detect_pose_dim(dataset):
    return int(dataset[0]["subsets_poses"][0].shape[1])


def metric_tag(val_loss, spearman, top1_margin):
    """生成嵌入文件名的指标字符串，去掉小数点避免路径问题。
    例：loss=0.0429, sp=0.7000, top1=0.9000
      → loss00429_sp07000_top109000
    """
    return (
        f"loss{f'{val_loss:.4f}'.replace('.', '')}"
        f"_sp{f'{spearman:.4f}'.replace('.', '')}"
        f"_top1{f'{top1_margin:.4f}'.replace('.', '')}"
    )


def remove_old_best(ckpt_dir: Path, prefix: str):
    """删除同前缀的旧 best 文件，只保留最新一个。"""
    for f in ckpt_dir.glob(f"{prefix}_*.pth"):
        try:
            f.unlink()
        except Exception:
            pass


# ── 损失函数 ──────────────────────────────────────────────────────────────────
def scene_rank_loss(pred_final, label_final, subset_masks):
    """只惩罚真实分差 > ACCEPT_MARGIN 的对中顺序错误的情况。"""
    B, N = pred_final.shape
    losses = []
    for b in range(B):
        valid = ~subset_masks[b]
        p = pred_final[b][valid]
        t = label_final[b][valid]
        if p.shape[0] < 2:
            continue
        p_diff = p[:, None] - p[None, :]
        t_diff = t[:, None] - t[None, :]
        mask   = torch.triu(t_diff.abs() > ACCEPT_MARGIN, diagonal=1)
        if mask.sum() == 0:
            continue
        losses.append(torch.relu(-torch.sign(t_diff) * p_diff)[mask].mean())
    if not losses:
        return pred_final.sum() * 0.0
    return torch.stack(losses).mean()


def top1_focused_loss(pred_final, label_final, subset_masks):
    """要求 good 组整体预测分高于 bad 组，hinge margin=0.1。"""
    B, N = pred_final.shape
    losses = []
    for b in range(B):
        valid  = ~subset_masks[b]
        p      = pred_final[b][valid]
        t      = label_final[b][valid]
        t_best = t.max()
        good   = (t_best - t) <= ACCEPT_MARGIN
        bad    = (t_best - t) >  ACCEPT_MARGIN
        if good.sum() == 0 or bad.sum() == 0:
            continue
        p_good = p[good][:, None]
        p_bad  = p[bad][None, :]
        losses.append(torch.relu(p_bad - p_good + 0.1).mean())
    if not losses:
        return pred_final.sum() * 0.0
    return torch.stack(losses).mean()


def calc_loss(scores, labels, subset_masks, loss_fn):
    valid       = ~subset_masks
    loss_global = loss_fn(scores[..., 0][valid], labels[..., 0][valid])
    loss_core   = loss_fn(scores[..., 1][valid], labels[..., 1][valid])
    rank_loss   = scene_rank_loss( scores[..., 2], labels[..., 2], subset_masks)
    top1_loss   = top1_focused_loss(scores[..., 2], labels[..., 2], subset_masks)
    loss = (
        LOSS_W_GLOBAL    * loss_global
        + LOSS_W_CORE    * loss_core
        + RANK_LOSS_WEIGHT * rank_loss
        + TOP1_LOSS_WEIGHT * top1_loss
    )
    return loss, {
        "loss_global": loss_global.detach().item(),
        "loss_core":   loss_core.detach().item(),
        "rank_loss":   rank_loss.detach().item(),
        "top1_loss":   top1_loss.detach().item(),
    }


# ── 评估指标 ──────────────────────────────────────────────────────────────────
def compute_metrics(all_scores, all_labels, all_masks):
    metrics = {}
    valid      = ~all_masks
    pred_valid = all_scores[valid]
    gt_valid   = all_labels[valid]

    for i, name in enumerate(["global_q_geo_raw", "crack_core_q_geo_raw", "q_final"]):
        err = pred_valid[:, i] - gt_valid[:, i]
        sp  = 0.0
        if len(pred_valid) > 1:
            r  = spearmanr(pred_valid[:, i], gt_valid[:, i]).correlation
            sp = 0.0 if np.isnan(r) else float(r)
        metrics[f"{name}_mae"]      = float(np.mean(np.abs(err)))
        metrics[f"{name}_rmse"]     = float(np.sqrt(np.mean(err ** 2)))
        metrics[f"{name}_spearman"] = sp

    hit_exact, hit_margin, total = 0, 0, 0
    for m in range(all_scores.shape[0]):
        v = ~all_masks[m]
        if v.sum() < 2:
            continue
        pred_s  = all_scores[m, v, 2]
        gt_s    = all_labels[m, v, 2]
        pred_i  = int(np.argmax(pred_s))
        gt_best = float(gt_s.max())
        hit_exact  += int(pred_i == int(np.argmax(gt_s)))
        hit_margin += int(gt_best - gt_s[pred_i] <= ACCEPT_MARGIN)
        total      += 1

    metrics["top1_exact_rate"]         = float(hit_exact  / total) if total > 0 else 0.0
    metrics["top1_within_margin_rate"] = float(hit_margin / total) if total > 0 else 0.0
    return metrics


# ── 优化器 ────────────────────────────────────────────────────────────────────
def build_optimizer(model):
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (backbone_params if name.startswith("image_encoder") else head_params).append(param)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": BACKBONE_LR,
                       "weight_decay": WEIGHT_DECAY, "name": "backbone"})
    if head_params:
        groups.append({"params": head_params, "lr": HEAD_LR,
                       "weight_decay": WEIGHT_DECAY, "name": "head"})
    return torch.optim.AdamW(groups)


def print_trainable_params(model):
    total, trainable, groups = 0, 0, defaultdict(int)
    for name, param in model.named_parameters():
        n = param.numel(); total += n
        if param.requires_grad:
            trainable += n
            g = ("image_encoder." + name.split(".")[1]
                 if name.startswith("image_encoder") and len(name.split(".")) >= 2
                 else "head")
            groups[g] += n
    print(f"\nTotal params    : {total/1e6:.3f} M")
    print(f"Trainable params: {trainable/1e6:.3f} M ({100.*trainable/total:.1f}%)")
    for k, v in groups.items():
        print(f"  {k:<26}: {v/1e6:.3f} M")
    print()


# ── Checkpoint ────────────────────────────────────────────────────────────────
def config_matches_checkpoint(ckpt_cfg, seed, pose_dim):
    if not ckpt_cfg:
        return False, ["no config in checkpoint"]
    cur  = get_config(seed, pose_dim)
    keys = ["pose_dim", "backbone_name", "train_backbone_from",
            "hidden_dim", "view_layers", "scene_layers", "num_heads", "dropout"]
    mm   = [f"{k}: ckpt={ckpt_cfg.get(k)} cur={cur.get(k)}"
            for k in keys if ckpt_cfg.get(k) != cur.get(k)]
    return len(mm) == 0, mm


def save_checkpoint(path, model, optimizer, scheduler, epoch,
                    best_val_loss, best_top1, best_spearman,
                    no_improve, metrics, seed, pose_dim):
    torch.save({
        "model":             model.state_dict(),
        "optimizer":         optimizer.state_dict(),
        "scheduler":         scheduler.state_dict(),
        "epoch":             epoch,
        "best_val_loss":     best_val_loss,
        "best_top1":         best_top1,         # top1_within_margin_rate 历史最优
        "best_spearman":     best_spearman,     # q_final_spearman 历史最优
        "no_improve_epochs": no_improve,
        "metrics":           metrics,
        "config":            get_config(seed, pose_dim),
    }, path)


def try_resume(model, optimizer, scheduler, device, resume_ckpt, seed, pose_dim, fresh):
    """
    断点续训，兼容两种旧 checkpoint 格式：
      - 新格式：checkpoint 里有独立的 best_top1 和 best_spearman 键
      - 旧格式：只有 best_spearman 键，但它实际存的是 top1_margin 的值
                （上一版代码的历史遗留问题）

    兼容策略：
      1. 读 best_top1，若不存在则置 -1.0
      2. 读 best_spearman，若其值 > 0.85（超出正常 spearman 范围），
         说明是旧格式把 top1 值错存在这里了：
           → best_top1 取 checkpoint 里 metrics 记录的真实 top1_within_margin_rate
           → best_spearman 取 checkpoint 里 metrics 记录的真实 q_final_spearman
    """
    start = 1
    best_loss, best_top1, best_sp, no_improve = float("inf"), -1.0, -1.0, 0

    if fresh or not resume_ckpt.exists():
        print("Start from scratch." if fresh else "No checkpoint found, start from scratch.")
        return start, best_loss, best_top1, best_sp, no_improve

    print(f"Resume: {resume_ckpt}")
    try:
        ckpt = torch.load(resume_ckpt, map_location=device)
        ok, mm = config_matches_checkpoint(ckpt.get("config", {}), seed, pose_dim)
        if not ok:
            print("[WARN] Config mismatch, start from scratch.")
            for m in mm: print(f"  {m}")
            return start, best_loss, best_top1, best_sp, no_improve

        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

        start     = int(ckpt.get("epoch", 0)) + 1
        best_loss = float(ckpt.get("best_val_loss", float("inf")))
        no_improve = int(ckpt.get("no_improve_epochs", 0))

        # ── 兼容新旧 checkpoint 格式 ─────────────────────────────────────────
        saved_top1 = float(ckpt.get("best_top1",     -1.0))
        saved_sp   = float(ckpt.get("best_spearman", -1.0))

        # metrics 里记录了最后一轮的真实指标值，作为修正基准
        last_metrics = ckpt.get("metrics", {})
        true_sp      = float(last_metrics.get("q_final_spearman",         -1.0))
        true_top1    = float(last_metrics.get("top1_within_margin_rate",  -1.0))

        # 旧格式判定：best_spearman > 0.85 不可能是真实 spearman（通常 < 0.75）
        # 说明旧代码把 top1_margin 值错存在了 best_spearman 键里
        if saved_sp > 0.85 and saved_top1 < 0:
            print(f"[INFO] 检测到旧版 checkpoint 格式（best_spearman={saved_sp:.4f} "
                  f"实为 top1_margin 的值），自动修正：")
            best_top1 = saved_sp      # 旧的 best_spearman 其实是 top1
            best_sp   = max(true_sp, 0.0)
            print(f"       best_top1    ← {best_top1:.4f}（原 best_spearman 字段）")
            print(f"       best_spearman← {best_sp:.4f}（来自 metrics）")
        else:
            best_top1 = saved_top1 if saved_top1 >= 0 else max(true_top1, 0.0)
            best_sp   = saved_sp   if saved_sp   >= 0 else max(true_sp,   0.0)

        print(
            f"Resumed epoch={start} | "
            f"best_top1={best_top1:.3f} | "
            f"best_spearman={best_sp:.4f} | "
            f"best_loss={best_loss:.6f} | "
            f"no_improve={no_improve}"
        )

    except Exception as e:
        print(f"[WARN] Failed to resume: {e}. Start from scratch.")

    return start, best_loss, best_top1, best_sp, no_improve


# ── 训练 / 验证 ───────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    loss_fn = nn.SmoothL1Loss()
    total_loss, seen = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", ncols=120, leave=False)
    for batch in pbar:
        images       = batch["images"].to(device, non_blocking=True)
        poses        = batch["poses"].to(device, non_blocking=True)
        view_masks   = batch["view_masks"].to(device, non_blocking=True)
        subset_masks = batch["subset_masks"].to(device, non_blocking=True)
        labels       = batch["labels"].to(device, non_blocking=True)

        scores = with_composite(model(images, poses, view_masks=view_masks, subset_masks=subset_masks))
        loss, items = calc_loss(scores, labels, subset_masks, loss_fn)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        seen       += images.size(0)
        pbar.set_postfix({
            "loss": f"{loss.item():.5f}",
            "top1": f"{items['top1_loss']:.5f}",
            "rank": f"{items['rank_loss']:.5f}",
        })
    return total_loss / max(seen, 1)


@torch.no_grad()
def validate(model, loader, device, epoch):
    model.eval()
    loss_fn = nn.SmoothL1Loss()
    total_loss, seen = 0.0, 0
    all_scores, all_labels, all_masks = [], [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]", ncols=120, leave=False)
    for batch in pbar:
        images       = batch["images"].to(device, non_blocking=True)
        poses        = batch["poses"].to(device, non_blocking=True)
        view_masks   = batch["view_masks"].to(device, non_blocking=True)
        subset_masks = batch["subset_masks"].to(device, non_blocking=True)
        labels       = batch["labels"].to(device, non_blocking=True)

        scores = with_composite(model(images, poses, view_masks=view_masks, subset_masks=subset_masks))
        loss, _ = calc_loss(scores, labels, subset_masks, loss_fn)

        total_loss += loss.item() * images.size(0)
        seen       += images.size(0)
        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_masks.append(subset_masks.cpu().numpy())
        pbar.set_postfix({"val_loss": f"{loss.item():.5f}"})

    max_n = max(m.shape[1] for m in all_masks)
    def padded(groups, fill):
        result = []
        for group in groups:
            pads = [(0,0)] * group.ndim
            pads[1] = (0, max_n-group.shape[1])
            result.append(np.pad(group,pads,constant_values=fill))
        return np.concatenate(result)
    metrics = compute_metrics(padded(all_scores,0), padded(all_labels,0), padded(all_masks,True))
    return total_loss / max(seen, 1), metrics


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    seed = args.seed
    ckpt_dir    = get_checkpoint_dir(args.checkpoint_root, seed)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_ckpt = ckpt_dir / "last.pth"

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Seed: {seed}")

    train_set = SceneSubsetDataset(TRAIN_CSV, image_size=IMAGE_SIZE, augment=True)
    val_set   = SceneSubsetDataset(VAL_CSV,   image_size=IMAGE_SIZE, augment=False)
    print(f"Train scenes: {len(train_set)} | Val scenes: {len(val_set)}")

    pose_dim = detect_pose_dim(train_set)
    print(f"Detected pose_dim: {pose_dim}")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
        collate_fn=scene_subset_collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
        collate_fn=scene_subset_collate_fn,
    )

    model = SubsetScoringNet(
        pose_dim=pose_dim, hidden_dim=HIDDEN_DIM, out_dim=2,
        backbone_name=BACKBONE_NAME, pretrained=True,
        train_backbone_from=TRAIN_BACKBONE_FROM,
        view_layers=VIEW_LAYERS, scene_layers=SCENE_LAYERS,
        num_heads=NUM_HEADS, dropout=DROPOUT,
    ).to(device)

    print_trainable_params(model)

    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=SCHEDULER_PATIENCE,
    )

    start, best_loss, best_top1, best_sp, no_improve = try_resume(
        model, optimizer, scheduler, device,
        resume_ckpt, seed, pose_dim, args.fresh,
    )

    for epoch in range(start, NUM_EPOCHS + 1):
        train_loss        = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, metrics = validate(model, val_loader, device, epoch)

        top1_margin = metrics["top1_within_margin_rate"]
        top1_exact  = metrics["top1_exact_rate"]
        q_spearman  = metrics["q_final_spearman"]

        scheduler.step(top1_margin)

        lr_text = ", ".join(f"{g['lr']:.2e}" for g in optimizer.param_groups)
        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"train={train_loss:.6f} | val={val_loss:.6f} | "
            f"top1_margin={top1_margin:.3f} | "
            f"top1_exact={top1_exact:.3f} | "
            f"spearman={q_spearman:.4f} | "
            f"lr=[{lr_text}]"
        )

        improved_loss = val_loss    < best_loss
        improved_top1 = top1_margin > best_top1
        improved_sp   = q_spearman  > best_sp

        if improved_loss: best_loss = val_loss
        if improved_top1: best_top1 = top1_margin
        if improved_sp:   best_sp   = q_spearman

        no_improve = 0 if improved_top1 else no_improve + 1

        ckpt_kw = dict(
            model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
            best_val_loss=best_loss, best_top1=best_top1, best_spearman=best_sp,
            no_improve=no_improve, metrics=metrics, seed=seed, pose_dim=pose_dim,
        )

        # last.pth：始终保存，用于断点续训
        save_checkpoint(ckpt_dir / "last.pth", **ckpt_kw)

        # 三类 best：文件名嵌入三项指标，每类只保留最新一个
        tag = metric_tag(val_loss, q_spearman, top1_margin)

        if improved_loss:
            remove_old_best(ckpt_dir, "best_loss")
            p = ckpt_dir / f"best_loss_ep{epoch:03d}_{tag}.pth"
            save_checkpoint(p, **ckpt_kw)
            print(f"  → {p.name}")

        if improved_top1:
            remove_old_best(ckpt_dir, "best_top1")
            p = ckpt_dir / f"best_top1_ep{epoch:03d}_{tag}.pth"
            save_checkpoint(p, **ckpt_kw)
            print(f"  → {p.name}")

        if improved_sp:
            remove_old_best(ckpt_dir, "best_spearman")
            p = ckpt_dir / f"best_spearman_ep{epoch:03d}_{tag}.pth"
            save_checkpoint(p, **ckpt_kw)
            print(f"  → {p.name}")

        if no_improve >= EARLY_STOP_PATIENCE:
            print(
                f"Early stop at epoch {epoch}. "
                f"best_top1={best_top1:.3f} | "
                f"best_spearman={best_sp:.4f} | "
                f"best_loss={best_loss:.6f}"
            )
            break

    print(
        f"\nDone. "
        f"best_top1_within_margin={best_top1:.3f} | "
        f"best_spearman={best_sp:.4f} | "
        f"best_val_loss={best_loss:.6f}"
    )


if __name__ == "__main__":
    main()
