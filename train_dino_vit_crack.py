# feature/train_dino_vit_crack_bbox.py
"""
方案 C（增强版）：DINO + 裂缝 bbox 密集监督联合训练

相对 train_dino_vit_crack.py（原方案 C，只用二值标签）的升级
--------------------------------------------------------------
原版：CLS token → 单 logit → BCE（只学「整张图有没有裂缝」，1 bit/图）。
本版：直接吃 YOLO bbox，三路监督一起教 backbone：

  ① Dense（patch 级，核心）：每个 patch token → 1 logit，
       预测该 patch 是否落在任何裂缝框内。强迫 backbone 在
       空间上 localize 裂缝。密集监督，表征增益最大。
  ② Global presence/area/count：CLS token → 回归裂缝面积占比、
       归一化数量，并做有/无裂缝二值。区分「裂缝密集 vs 稀疏」。
  ③ Global orientation：横/竖朝向（仅对有裂缝图回归）。

    L_total = L_dino + crack_alpha × (w_dense·L_dense + w_global·L_global)

L_dino 对所有图照常计算（保留视觉多样性表征，MMR 需要它）。
crack_alpha 三段式调度（warmup→ramp→稳定），避免早期干扰 DINO。

★ 推理约束（与原方案 C 完全一致）★
  所有 crack head 训练完即丢弃，只保存 backbone。
  推理时：图像 → backbone → 384 维 embedding，pose → 9 维，合并 397 维 → PPO。
  全程不需要任何标注。make_sfm_features.py 不变，RL 代码不变。

标注格式（YOLO，与目标检测训练用的同一套）
--------------------------------------------
每张「有裂缝」的图片对应一个同名 .txt，每行：

    <class_id> <x_center> <y_center> <width> <height>     （坐标归一化到 [0,1]）

例如：
    0 0.810937 0.442083 0.378125 0.880833

「没裂缝」的图片没有 .txt 文件，视为确定的负样本（patch mask 全 0）。

支持的目录结构（自动识别）
    结构 A（YOLO 标准）：dataset/scene_X/images/img_001.jpg
                         dataset/scene_X/labels/img_001.txt
    结构 B（同目录）：   dataset/scene_X/img_001.jpg + img_001.txt
    结构 C（手动）：     EXTRA_LABEL_DIRS 里按文件名 stem 搜索

输出（默认 crack_finetune/ 目录）
---------------------------------
  crack_finetune/dino_vit_crack_backbone.pth    ← 用这个替换 make_sfm 的 backbone
  crack_finetune/dino_vit_crack_full.pth        ← 完整 student state_dict
  crack_finetune/dino_vit_crack_checkpoint.pth  ← 可断点续训

Place at:  E:\\view_subset\\rl\\feature\\train_dino_vit_crack_bbox.py
Run from:  E:\\view_subset\\rl
    python feature/train_dino_vit_crack_bbox.py
"""

import os
import glob
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════════════
# ✅ CONFIG
# ════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ---- 数据路径 ----
    "UNLABELED_DIR":   r"E:\view_subset\rl_v10_C\dataset\labelimg_work\images",   # 全部图片根目录（含有/无裂缝）

    # YOLO 标注查找：脚本会按「结构 A/B」自动找与图片同名的 .txt。
    # 若你的标注放在统一目录，在这里补充（按文件名 stem 搜索）。
    "EXTRA_LABEL_DIRS": [],

    # 只把指定 class_id 当裂缝；None = 所有类都算裂缝
    "CLASS_IDS": None,

    # 没有 .txt 的图片是否当作「确定无裂缝」的负样本。
    #   True  → 所有图都参与 crack loss（你的情况：没裂缝就没 txt，推荐 True）
    #   False → 没 txt 的图只走 DINO loss（适合标注不完整、存在「未标注」灰区）
    "MISSING_TXT_AS_NOCRACK": True,

    # ---- 输出路径 ----
    "OUT_BACKBONE":     "crack_finetune/dino_vit_crack_backbone.pth",
    "OUT_STUDENT_FULL": "crack_finetune/dino_vit_crack_full.pth",
    "CHECKPOINT_PATH":  "crack_finetune/dino_vit_crack_checkpoint.pth",

    # ---- 初始化：强烈建议填入已跑完 100 epoch 的无监督 backbone ----
    "PRETRAIN_BACKBONE_PTH": r"E:\view_subset\rl_v10_C\feature\100epoch\dino_vit_student.pth",
    "USE_PRETRAIN_BACKBONE": True,
    "PRETRAIN_STRICT":       True,

    # ---- 断点续训 ----
    "AUTO_RESUME":      True,
    "RESUME_STRICT":    True,
    "SAVE_EVERY_EPOCHS": 1,

    # ── 裂缝辅助监督权重 ──────────────────────────────────────────────
    # 总权重 crack_alpha（三段式调度），建议 0.05~0.25
    "CRACK_ALPHA":          0.12,
    "CRACK_WARMUP_EPOCHS":  5,     # 前 5 epoch crack_alpha = 0（纯 DINO）
    "CRACK_RAMP_EPOCHS":    5,

    # crack 内部子 loss 权重（dense 为主，global 为辅）
    "CRACK_DENSE_WEIGHT":   1.00,  # patch 级分割 loss
    "CRACK_GLOBAL_WEIGHT":  0.80,  # CLS 级 presence/area/count/orient loss

    # patch BCE 正样本权重（裂缝 patch 远少于背景 patch，需上调）
    # 经验值 5~12；裂缝越细越稀疏取越大
    "CRACK_POS_WEIGHT":     2.5,

    # crack view 的水平翻转概率（会同步翻转 bbox）
    "CRACK_HFLIP_P":        0.5,

    # ---- 模型 ----
    "VIT_NAME":     "vit_small_patch16_224",
    "IMG_SIZE":     224,
    "PATCH_SIZE":   16,

    # ---- multi-crop（DINO 自监督部分，不变） ----
    "GLOBAL_CROPS_SCALE":  (0.4, 1.0),
    "LOCAL_CROPS_SCALE":   (0.05, 0.4),
    "LOCAL_CROPS_NUMBER":  2,
    "LOCAL_CROP_SIZE":     96,

    # ---- 训练 ----
    "BATCH_SIZE_PER_GPU": 16,
    "EPOCHS":             80,
    "NUM_WORKERS":        0,
    "SEED":               0,

    # ---- DINO head / loss ----
    "OUT_DIM":           65536,
    "NORM_LAST_LAYER":   True,
    "USE_BN_IN_HEAD":    False,

    # ---- teacher temperature ----
    "WARMUP_TEACHER_TEMP":        0.04,
    "TEACHER_TEMP":               0.04,
    "WARMUP_TEACHER_TEMP_EPOCHS": 0,

    # ---- 优化 ----
    "USE_FP16":          True,
    "WEIGHT_DECAY":      0.04,
    "WEIGHT_DECAY_END":  0.4,
    "CLIP_GRAD":         3.0,
    "LR":                0.0003,
    "WARMUP_EPOCHS":     5,
    "MIN_LR":            1e-6,
    "FREEZE_LAST_LAYER": 1,
    "DROP_PATH_RATE":    0.1,

    # ---- teacher EMA ----
    "MOMENTUM_TEACHER": 0.996,

    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}
# ════════════════════════════════════════════════════════════════════════


# ────────────────────────────────────────────────────────────────────────
# YOLO 标注查找与解析（从 add_crack_features.py 移植）
# ────────────────────────────────────────────────────────────────────────
_IMG_DIR_NAMES = {"images", "image", "imgs", "img", "JPEGImages", "rgb"}


def _find_label_path(img_path: str,
                     extra_dirs: Optional[List[str]] = None) -> Optional[str]:
    """
    按优先级查找与 img_path 对应的 YOLO .txt：
      策略 1：把路径中的 images/img 目录替换为 labels，扩展名换 .txt
      策略 2：与图片同目录、同名 .txt
      策略 3：在 extra_dirs 里按 stem 搜索
    找不到返回 None（视为无裂缝）。
    """
    p    = Path(img_path)
    stem = p.stem

    # 策略 1
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part.lower() in _IMG_DIR_NAMES:
            cand_parts = parts[:i] + ["labels"] + parts[i + 1:]
            cand = Path(*cand_parts).with_suffix(".txt")
            if cand.exists():
                return str(cand)
            break

    # 策略 2
    cand = p.parent / (stem + ".txt")
    if cand.exists():
        return str(cand)

    # 策略 3
    if extra_dirs:
        for d in extra_dirs:
            cand = Path(d) / (stem + ".txt")
            if cand.exists():
                return str(cand)

    return None


def _parse_yolo(label_path: Optional[str],
                class_ids: Optional[List[int]] = None
                ) -> List[Tuple[float, float, float, float]]:
    """读 YOLO txt → [(cx, cy, w, h), ...]（归一化）。不存在返回空列表。"""
    if not label_path or not os.path.isfile(label_path):
        return []
    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue
            if class_ids is not None and cls not in class_ids:
                continue
            try:
                cx, cy, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            if w > 0 and h > 0 and 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0:
                boxes.append((cx, cy, w, h))
    return boxes


# ────────────────────────────────────────────────────────────────────────
# bbox → 训练目标（patch mask + 全局属性）
# ────────────────────────────────────────────────────────────────────────
def boxes_to_patch_mask(boxes: List[Tuple[float, float, float, float]],
                        grid: int) -> np.ndarray:
    """
    把归一化 bbox 转成 grid×grid 的 patch 级二值 mask（展平为 (grid*grid,)）。
    任何与某个 bbox 相交的 patch 记为 1。框监督是粗糙的（框内含背景），
    目的不是精确分割，而是引导 backbone「这一带有裂缝相关内容」。
    """
    mask = np.zeros((grid, grid), dtype=np.float32)
    for (cx, cy, w, h) in boxes:
        x1 = max(0.0, cx - w / 2.0) * grid
        x2 = min(1.0, cx + w / 2.0) * grid
        y1 = max(0.0, cy - h / 2.0) * grid
        y2 = min(1.0, cy + h / 2.0) * grid
        ix1 = int(np.clip(np.floor(x1), 0, grid))
        ix2 = int(np.clip(np.ceil(x2),  0, grid))
        iy1 = int(np.clip(np.floor(y1), 0, grid))
        iy2 = int(np.clip(np.ceil(y2),  0, grid))
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = 1.0
    return mask.reshape(-1)   # (grid*grid,)


def boxes_to_global(boxes: List[Tuple[float, float, float, float]]) -> np.ndarray:
    """
    全局 4 维目标 [presence, area_ratio, count_norm, orientation]：
      presence    1=有裂缝 0=无裂缝
      area_ratio  ∑(w×h)，clip 到 [0,1]
      count_norm  min(n/8, 1)
      orientation 面积加权朝向：1=横向(w>h) 0=竖向（无裂缝时占位 0.5，不计 loss）
    """
    if not boxes:
        return np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)

    areas = np.array([w * h for (cx, cy, w, h) in boxes], dtype=np.float64)
    ws    = np.array([w     for (cx, cy, w, h) in boxes], dtype=np.float64)
    hs    = np.array([h     for (cx, cy, w, h) in boxes], dtype=np.float64)

    total_area = float(areas.sum())
    w_norm     = areas / (total_area + 1e-12)
    orient     = float(np.dot(w_norm, (ws > hs).astype(np.float64)))

    return np.array([
        1.0,
        float(np.clip(total_area, 0.0, 1.0)),
        float(min(len(boxes) / 8.0, 1.0)),
        orient,
    ], dtype=np.float32)


# ────────────────────────────────────────────────────────────────────────
# 裂缝辅助头（dense + global，训练后全部丢弃）
# ────────────────────────────────────────────────────────────────────────
class CrackDenseHead(nn.Module):
    """patch token (B, Np, D) → 每 patch 1 logit (B, Np)。"""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 1)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.fc(patch_tokens).squeeze(-1)   # (B, Np)


class CrackGlobalHead(nn.Module):
    """CLS pooled feature (B, D) → 4 维 [presence_logit, area, count, orient]。"""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 4),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, cls_feat: torch.Tensor) -> torch.Tensor:
        return self.net(cls_feat)   # (B, 4)


# ────────────────────────────────────────────────────────────────────────
# 工具函数（沿用原版）
# ────────────────────────────────────────────────────────────────────────
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def list_images(root: str) -> List[str]:
    exts  = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(root, "**", e), recursive=True)
    paths.sort()
    if not paths:
        raise RuntimeError(f"No images found under: {root}")
    return paths


def fix_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_scheduler(base: float, final: float, epochs: int,
                     iters_per_epoch: int, warmup_epochs: int = 0) -> List[float]:
    total  = epochs * iters_per_epoch
    warmup = warmup_epochs * iters_per_epoch
    sched  = []
    for i in range(total):
        if warmup > 0 and i < warmup:
            val = base * (i + 1) / warmup
        else:
            t   = (i - warmup) / max(1, total - warmup)
            val = final + 0.5 * (base - final) * (1 + math.cos(math.pi * t))
        sched.append(val)
    return sched


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, m: float):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_((1.0 - m) * ps.data)


def get_params_groups(model: nn.Module):
    reg, noreg = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            noreg.append(p)
        else:
            reg.append(p)
    return [{"params": reg}, {"params": noreg, "weight_decay": 0.0}]


def get_crack_alpha(epoch: int) -> float:
    """warmup 前为 0，之后 ramp 个 epoch 线性升至 CRACK_ALPHA，再保持。"""
    warmup = int(CONFIG["CRACK_WARMUP_EPOCHS"])
    ramp   = int(CONFIG["CRACK_RAMP_EPOCHS"])
    alpha  = float(CONFIG["CRACK_ALPHA"])
    if epoch < warmup:
        return 0.0
    elif epoch < warmup + ramp:
        return alpha * (epoch - warmup) / max(ramp, 1)
    else:
        return alpha


# ────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ────────────────────────────────────────────────────────────────────────
def _torch_load_checkpoint(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy":  np.random.get_state(),
        "torch":  torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception as exc:
            print(f"Warning: could not restore CUDA RNG state: {exc}")


def save_checkpoint(path: str,
                    epoch_next: int,
                    iteration: int,
                    student: nn.Module,
                    teacher: nn.Module,
                    crack_dense_head: nn.Module,
                    crack_global_head: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scaler,
                    dino_loss: nn.Module):
    ensure_dir(path)
    ckpt = {
        "epoch_next":        epoch_next,
        "iteration":         iteration,
        "student":           student.state_dict(),
        "teacher":           teacher.state_dict(),
        "crack_dense_head":  crack_dense_head.state_dict(),
        "crack_global_head": crack_global_head.state_dict(),
        "optimizer":         optimizer.state_dict(),
        "scaler":            scaler.state_dict() if scaler is not None else None,
        "dino_loss":         dino_loss.state_dict(),
        "rng_state":         get_rng_state(),
        "config":            dict(CONFIG),
    }
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)
    print(f"Checkpoint saved: {path}  "
          f"(next epoch={epoch_next + 1}, iteration={iteration})")


def load_checkpoint_if_available(path: str,
                                  student: nn.Module,
                                  teacher: nn.Module,
                                  crack_dense_head: nn.Module,
                                  crack_global_head: nn.Module,
                                  optimizer: torch.optim.Optimizer,
                                  scaler,
                                  dino_loss: nn.Module,
                                  device: str,
                                  strict: bool = True) -> Tuple[int, int]:
    if not os.path.isfile(path):
        print(f"No checkpoint found at: {path}")
        return 0, 0

    print(f"Loading checkpoint: {path}")
    ckpt = _torch_load_checkpoint(path, map_location="cpu")

    student.load_state_dict(ckpt["student"], strict=strict)
    teacher.load_state_dict(ckpt["teacher"], strict=strict)
    if "crack_dense_head" in ckpt:
        crack_dense_head.load_state_dict(ckpt["crack_dense_head"], strict=strict)
    if "crack_global_head" in ckpt:
        crack_global_head.load_state_dict(ckpt["crack_global_head"], strict=strict)
    optimizer.load_state_dict(ckpt["optimizer"])

    if scaler is not None and ckpt.get("scaler") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as exc:
            print(f"Warning: could not restore AMP scaler state: {exc}")

    if "dino_loss" in ckpt:
        dino_loss.load_state_dict(ckpt["dino_loss"], strict=strict)

    set_rng_state(ckpt.get("rng_state", {}))

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    start_epoch = int(ckpt.get("epoch_next", ckpt.get("epoch", 0)))
    iteration   = int(ckpt.get("iteration", start_epoch))
    print(f"Resume from epoch {start_epoch + 1}/{CONFIG['EPOCHS']} "
          f"(global iteration={iteration})")
    return start_epoch, iteration


# ────────────────────────────────────────────────────────────────────────
# Pretrained backbone 初始化
# ────────────────────────────────────────────────────────────────────────
def _strip_prefix_if_present(state_dict: dict, prefix: str) -> dict:
    if not state_dict:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _extract_backbone_state(raw_state) -> dict:
    if not isinstance(raw_state, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(raw_state)}")
    for key in ("state_dict", "student", "model", "teacher"):
        if key in raw_state and isinstance(raw_state[key], dict):
            raw_state = raw_state[key]
            break
    if any(k.startswith("backbone.") for k in raw_state.keys()):
        raw_state = {
            k[len("backbone."):]: v
            for k, v in raw_state.items()
            if k.startswith("backbone.")
        }
    raw_state = _strip_prefix_if_present(raw_state, "module.")
    raw_state = _strip_prefix_if_present(raw_state, "backbone.")
    raw_state = {k: v for k, v in raw_state.items()
                 if not k.startswith("head.")}
    return raw_state


def load_pretrained_backbone_if_available(backbone_s: nn.Module,
                                          backbone_t: nn.Module,
                                          path: str,
                                          strict: bool = True) -> bool:
    if not path or not os.path.isfile(path):
        print(f"Pretrained backbone not found, skip: {path}")
        return False
    print(f"Loading pretrained DINO backbone: {path}")
    raw_state = _torch_load_checkpoint(path, map_location="cpu")
    state     = _extract_backbone_state(raw_state)
    backbone_s.load_state_dict(state, strict=strict)
    backbone_t.load_state_dict(state, strict=strict)
    print("Pretrained backbone loaded into student and teacher backbones.")
    return True


# ────────────────────────────────────────────────────────────────────────
# 数据增强：DINO multi-crop（不变）
# ────────────────────────────────────────────────────────────────────────
class DataAugmentationDINO:
    def __init__(self,
                 global_crops_scale: Tuple[float, float],
                 local_crops_scale:  Tuple[float, float],
                 local_crops_number: int,
                 img_size:           int,
                 local_crop_size:    int):

        norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std =(0.229, 0.224, 0.225)),
        ])
        flip_color = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        self.global_1 = transforms.Compose([
            transforms.RandomResizedCrop(
                img_size, scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC),
            flip_color,
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            norm,
        ])
        self.global_2 = transforms.Compose([
            transforms.RandomResizedCrop(
                img_size, scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC),
            flip_color,
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.RandomSolarize(threshold=128, p=0.2),
            norm,
        ])
        self.local_crops_number = local_crops_number
        self.local = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crop_size, scale=local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC),
            flip_color,
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            norm,
        ])

    def __call__(self, image: Image.Image) -> List[torch.Tensor]:
        crops = [self.global_1(image), self.global_2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local(image))
        return crops


# ────────────────────────────────────────────────────────────────────────
# 数据增强：crack view（空间对齐，bbox 监督专用）
# ────────────────────────────────────────────────────────────────────────
class CrackViewTransform:
    """
    为 bbox 监督单独构造一个【空间对齐】的 view：
      直接 resize 到 img_size×img_size（归一化 bbox 坐标保持有效）
      → 可选水平翻转（同步翻转 bbox 的 cx → 1-cx）
      → 颜色级增强（不改变空间布局）
      → ToTensor + Normalize
    这样 bbox→patch 的映射是确定的，不像 DINO 的 RandomResizedCrop 会错位。
    """
    def __init__(self, img_size: int, grid: int, hflip_p: float):
        self.img_size = int(img_size)
        self.grid     = int(grid)
        self.hflip_p  = float(hflip_p)
        self.color = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        ])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std =(0.229, 0.224, 0.225)),
        ])

    def __call__(self, pil_img: Image.Image,
                 boxes: List[Tuple[float, float, float, float]]
                 ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        img = pil_img.resize((self.img_size, self.img_size), Image.BICUBIC)

        if random.random() < self.hflip_p:
            img   = img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [(1.0 - cx, cy, w, h) for (cx, cy, w, h) in boxes]

        img    = self.color(img)
        tensor = self.norm(img)

        patch_mask = boxes_to_patch_mask(boxes, self.grid)   # (grid*grid,)
        global_t   = boxes_to_global(boxes)                  # (4,)
        return tensor, patch_mask, global_t


# ────────────────────────────────────────────────────────────────────────
# Dataset：同时产出 DINO crops 与 crack view + 监督目标
# ────────────────────────────────────────────────────────────────────────
class MultiCropDataset(Dataset):
    """
    __getitem__ 返回:
      crops         : List[Tensor]  DINO multi-crop（用于 dino loss）
      crack_img     : Tensor (3,H,W) 空间对齐 view（用于 crack loss）
      patch_target  : Tensor (grid*grid,) patch 级二值 mask
      global_target : Tensor (4,)  [presence, area, count, orient]
      valid         : int   1=参与 crack loss，0=只走 dino loss
    """
    def __init__(self,
                 paths:           List[str],
                 dino_transform:  DataAugmentationDINO,
                 crack_transform: CrackViewTransform,
                 label_path_map:  Dict[str, Optional[str]],
                 class_ids:       Optional[List[int]],
                 missing_as_nocrack: bool):
        self.paths              = paths
        self.dino_transform     = dino_transform
        self.crack_transform    = crack_transform
        self.label_path_map     = label_path_map
        self.class_ids          = class_ids
        self.missing_as_nocrack = bool(missing_as_nocrack)
        self.grid               = crack_transform.grid

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p   = self.paths[idx]
        img = Image.open(p).convert("RGB")

        crops = self.dino_transform(img)

        lp = self.label_path_map.get(p, None)
        if lp is not None:
            boxes = _parse_yolo(lp, self.class_ids)
            valid = 1
        else:
            boxes = []
            # 无 txt：是否当作确定负样本
            valid = 1 if self.missing_as_nocrack else 0

        crack_img, patch_mask, global_t = self.crack_transform(img, boxes)

        return (
            crops,
            crack_img,
            torch.from_numpy(patch_mask),
            torch.from_numpy(global_t),
            int(valid),
        )


# ────────────────────────────────────────────────────────────────────────
# DINO 模型（与原版一致）
# ────────────────────────────────────────────────────────────────────────
class DINOHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 use_bn: bool, norm_last_layer: bool):
        super().__init__()
        hidden = in_dim
        layers = [nn.Linear(in_dim, hidden)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden))
        layers += [nn.GELU(), nn.Linear(hidden, hidden)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(nn.GELU())
        self.mlp  = nn.Sequential(*layers)
        self.last = nn.utils.weight_norm(nn.Linear(hidden, out_dim, bias=False))
        self.last.weight_g.data.fill_(1.0)
        if norm_last_layer:
            self.last.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last(x)


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        return [self.head(self.backbone(inp)) for inp in x]


def build_vit_backbone(vit_name: str, drop_path_rate: float) -> nn.Module:
    return timm.create_model(
        vit_name,
        pretrained=False,
        num_classes=0,
        drop_path_rate=drop_path_rate,
        dynamic_img_size=True,
    )


def forward_backbone_tokens(backbone: nn.Module, x: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回 (pooled_cls, patch_tokens)：
      pooled_cls   (B, D)     与 make_sfm_features.py 的 backbone(x) 一致
      patch_tokens (B, Np, D) 去掉前缀 token（CLS/register）后的 patch token
    """
    feats    = backbone.forward_features(x)          # (B, T, D)
    n_prefix = getattr(backbone, "num_prefix_tokens", 1)
    patch_tokens = feats[:, n_prefix:, :]            # (B, Np, D)
    try:
        pooled = backbone.forward_head(feats, pre_logits=True)   # (B, D)
    except TypeError:
        pooled = feats[:, 0]                          # 兜底：取 CLS
    return pooled, patch_tokens


class DINOLoss(nn.Module):
    def __init__(self, out_dim: int, ncrops: int,
                 warmup_teacher_temp: float, teacher_temp: float,
                 warmup_teacher_temp_epochs: int, nepochs: int,
                 student_temp: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        self.student_temp    = student_temp
        self.center_momentum = center_momentum
        self.ncrops          = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp,
                        warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
        )).astype(np.float32)

    def forward(self, student_out: List[torch.Tensor],
                teacher_out: List[torch.Tensor], epoch: int) -> torch.Tensor:
        s_logp = [F.log_softmax(s / self.student_temp, dim=-1)
                  for s in student_out]
        temp   = float(self.teacher_temp_schedule[epoch])
        t_probs = [F.softmax((t - self.center) / temp, dim=-1)
                   for t in teacher_out]
        total, n = 0.0, 0
        for iq, tq in enumerate(t_probs):
            for v, sv in enumerate(s_logp):
                if v == iq:
                    continue
                total += torch.sum(-tq * sv, dim=-1).mean()
                n     += 1
        total /= max(n, 1)
        self.update_center(torch.cat(teacher_out, dim=0))
        return total

    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor):
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center  = (self.center * self.center_momentum
                        + batch_center * (1 - self.center_momentum))


# ────────────────────────────────────────────────────────────────────────
# crack loss（dense + global）
# ────────────────────────────────────────────────────────────────────────
def compute_crack_loss(pooled: torch.Tensor,
                       patch_tokens: torch.Tensor,
                       crack_dense_head: nn.Module,
                       crack_global_head: nn.Module,
                       patch_target: torch.Tensor,
                       global_target: torch.Tensor,
                       valid_mask: torch.Tensor,
                       pos_weight: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回 (loss_crack, loss_dense, loss_global)。
    仅在 valid_mask=True 的样本上计算。无 valid 样本时返回 0。
    """
    device = pooled.device
    zero   = torch.zeros((), device=device)

    if valid_mask.sum() == 0:
        return zero, zero, zero

    p_tok = patch_tokens[valid_mask]      # (n, Np, D)
    p_cls = pooled[valid_mask]            # (n, D)
    p_tgt = patch_target[valid_mask]      # (n, Np)
    g_tgt = global_target[valid_mask]     # (n, 4)

    # ① dense：patch 级 BCE（pos_weight 缓解裂缝 patch 稀疏）
    dense_logits = crack_dense_head(p_tok)            # (n, Np)
    loss_dense = F.binary_cross_entropy_with_logits(
        dense_logits, p_tgt, pos_weight=pos_weight
    )

    # ② / ③ global：presence(BCE) + area/count(回归) + orient(仅有裂缝)
    g_pred         = crack_global_head(p_cls)         # (n, 4)
    presence_logit = g_pred[:, 0]
    area_pred      = torch.sigmoid(g_pred[:, 1])
    count_pred     = torch.sigmoid(g_pred[:, 2])
    orient_pred    = torch.sigmoid(g_pred[:, 3])

    presence_tgt = g_tgt[:, 0]
    area_tgt     = g_tgt[:, 1]
    count_tgt    = g_tgt[:, 2]
    orient_tgt   = g_tgt[:, 3]

    loss_presence = F.binary_cross_entropy_with_logits(presence_logit, presence_tgt)
    loss_area     = F.smooth_l1_loss(area_pred,  area_tgt)
    loss_count    = F.smooth_l1_loss(count_pred, count_tgt)

    crack_mask = presence_tgt > 0.5
    if crack_mask.any():
        loss_orient = F.smooth_l1_loss(orient_pred[crack_mask], orient_tgt[crack_mask])
    else:
        loss_orient = zero

    loss_global = loss_presence + loss_area + loss_count + 0.5 * loss_orient

    w_dense  = float(CONFIG["CRACK_DENSE_WEIGHT"])
    w_global = float(CONFIG["CRACK_GLOBAL_WEIGHT"])
    loss_crack = w_dense * loss_dense + w_global * loss_global
    return loss_crack, loss_dense, loss_global


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════
def main():
    fix_seeds(CONFIG["SEED"])
    device = CONFIG["DEVICE"]
    grid   = CONFIG["IMG_SIZE"] // CONFIG["PATCH_SIZE"]

    print("Device:", device)
    print(f"\n{'─'*64}")
    print(f"  DINO + 裂缝 bbox 密集监督联合训练")
    print(f"  patch 网格        = {grid}×{grid} = {grid*grid}")
    print(f"  crack_alpha 目标  = {CONFIG['CRACK_ALPHA']}  "
          f"(warmup {CONFIG['CRACK_WARMUP_EPOCHS']} / ramp {CONFIG['CRACK_RAMP_EPOCHS']})")
    print(f"  dense / global 权 = {CONFIG['CRACK_DENSE_WEIGHT']} / {CONFIG['CRACK_GLOBAL_WEIGHT']}")
    print(f"  patch pos_weight  = {CONFIG['CRACK_POS_WEIGHT']}")
    print(f"  无 txt 当负样本    = {CONFIG['MISSING_TXT_AS_NOCRACK']}")
    print(f"{'─'*64}\n")

    # ── 扫描图片，预构建 标注路径映射 ────────────────────────────────
    paths = list_images(CONFIG["UNLABELED_DIR"])
    label_path_map: Dict[str, Optional[str]] = {}
    n_with_txt = 0
    for p in paths:
        lp = _find_label_path(p, CONFIG["EXTRA_LABEL_DIRS"])
        label_path_map[p] = lp
        if lp is not None:
            n_with_txt += 1

    n_without_txt = len(paths) - n_with_txt
    print(f"Images found      : {len(paths)}")
    print(f"  有 YOLO txt（有裂缝）   : {n_with_txt}")
    print(f"  无 txt（无裂缝/未标注） : {n_without_txt}")
    if CONFIG["MISSING_TXT_AS_NOCRACK"]:
        print(f"  → 无 txt 图作为确定负样本，全部参与 crack loss")
    else:
        print(f"  → 无 txt 图只走 DINO loss（不参与 crack loss）")
    if n_with_txt == 0:
        print("  ⚠️  没找到任何 YOLO txt，crack loss 将退化（请检查标注路径）。")

    for out_path in [CONFIG["OUT_BACKBONE"],
                     CONFIG["OUT_STUDENT_FULL"],
                     CONFIG["CHECKPOINT_PATH"]]:
        ensure_dir(out_path)

    # ── Dataset & DataLoader ─────────────────────────────────────────
    dino_transform = DataAugmentationDINO(
        global_crops_scale  = CONFIG["GLOBAL_CROPS_SCALE"],
        local_crops_scale   = CONFIG["LOCAL_CROPS_SCALE"],
        local_crops_number  = CONFIG["LOCAL_CROPS_NUMBER"],
        img_size            = CONFIG["IMG_SIZE"],
        local_crop_size     = CONFIG["LOCAL_CROP_SIZE"],
    )
    crack_transform = CrackViewTransform(
        img_size = CONFIG["IMG_SIZE"],
        grid     = grid,
        hflip_p  = CONFIG["CRACK_HFLIP_P"],
    )
    ds = MultiCropDataset(
        paths              = paths,
        dino_transform     = dino_transform,
        crack_transform    = crack_transform,
        label_path_map     = label_path_map,
        class_ids          = CONFIG["CLASS_IDS"],
        missing_as_nocrack = CONFIG["MISSING_TXT_AS_NOCRACK"],
    )
    dl = DataLoader(
        ds,
        batch_size  = CONFIG["BATCH_SIZE_PER_GPU"],
        shuffle     = True,
        drop_last   = True,
        num_workers = CONFIG["NUM_WORKERS"],
        pin_memory  = (device == "cuda"),
    )

    # ── Student / Teacher ────────────────────────────────────────────
    backbone_s = build_vit_backbone(CONFIG["VIT_NAME"], CONFIG["DROP_PATH_RATE"]).to(device)
    backbone_t = build_vit_backbone(CONFIG["VIT_NAME"], 0.0).to(device)

    if CONFIG["USE_PRETRAIN_BACKBONE"]:
        load_pretrained_backbone_if_available(
            backbone_s, backbone_t,
            CONFIG["PRETRAIN_BACKBONE_PTH"],
            strict=CONFIG["PRETRAIN_STRICT"],
        )
    else:
        print("USE_PRETRAIN_BACKBONE=False，从随机初始化开始。")

    embed_dim = backbone_s.num_features
    print(f"Embed dim: {embed_dim}")

    head_s = DINOHead(embed_dim, CONFIG["OUT_DIM"],
                      CONFIG["USE_BN_IN_HEAD"], CONFIG["NORM_LAST_LAYER"]).to(device)
    head_t = DINOHead(embed_dim, CONFIG["OUT_DIM"],
                      CONFIG["USE_BN_IN_HEAD"], False).to(device)

    student = MultiCropWrapper(backbone_s, head_s).to(device)
    teacher = MultiCropWrapper(backbone_t, head_t).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # ── 裂缝辅助头（训练后丢弃） ─────────────────────────────────────
    crack_dense_head  = CrackDenseHead(embed_dim).to(device)
    crack_global_head = CrackGlobalHead(embed_dim).to(device)
    n_head_params = (sum(p.numel() for p in crack_dense_head.parameters())
                     + sum(p.numel() for p in crack_global_head.parameters()))
    print(f"Crack heads 参数量: {n_head_params}  (dense + global，最终丢弃)")

    pos_weight = torch.tensor(float(CONFIG["CRACK_POS_WEIGHT"]), device=device)

    # ── Loss & Optimizer ─────────────────────────────────────────────
    ncrops    = 2 + CONFIG["LOCAL_CROPS_NUMBER"]
    dino_loss = DINOLoss(
        out_dim                    = CONFIG["OUT_DIM"],
        ncrops                     = ncrops,
        warmup_teacher_temp        = CONFIG["WARMUP_TEACHER_TEMP"],
        teacher_temp               = CONFIG["TEACHER_TEMP"],
        warmup_teacher_temp_epochs = CONFIG["WARMUP_TEACHER_TEMP_EPOCHS"],
        nepochs                    = CONFIG["EPOCHS"],
    ).to(device)

    params_groups = get_params_groups(student)
    params_groups.append({"params": list(crack_dense_head.parameters())
                                     + list(crack_global_head.parameters())})
    optimizer = torch.optim.AdamW(
        params_groups, lr=0.0, weight_decay=CONFIG["WEIGHT_DECAY"]
    )

    iters_per_epoch = len(dl)
    global_batch    = CONFIG["BATCH_SIZE_PER_GPU"]
    base_lr         = CONFIG["LR"] * (global_batch / 256.0)

    lr_sched  = cosine_scheduler(base_lr, CONFIG["MIN_LR"],
                                 CONFIG["EPOCHS"], iters_per_epoch,
                                 warmup_epochs=CONFIG["WARMUP_EPOCHS"])
    wd_sched  = cosine_scheduler(CONFIG["WEIGHT_DECAY"], CONFIG["WEIGHT_DECAY_END"],
                                 CONFIG["EPOCHS"], iters_per_epoch)
    mom_sched = cosine_scheduler(CONFIG["MOMENTUM_TEACHER"], 1.0,
                                 CONFIG["EPOCHS"], iters_per_epoch)

    # ── AMP ─────────────────────────────────────────────────────────
    _amp_device = "cuda" if device == "cuda" else "cpu"
    _use_fp16   = CONFIG["USE_FP16"] and (device == "cuda")
    try:
        scaler = torch.amp.GradScaler(_amp_device, enabled=_use_fp16)
        def autocast_ctx():
            return torch.amp.autocast(_amp_device, enabled=_use_fp16)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=_use_fp16)
        def autocast_ctx():
            return torch.cuda.amp.autocast(enabled=_use_fp16)

    # ── 续训 ─────────────────────────────────────────────────────────
    if CONFIG["AUTO_RESUME"]:
        start_epoch, iteration = load_checkpoint_if_available(
            CONFIG["CHECKPOINT_PATH"],
            student, teacher, crack_dense_head, crack_global_head,
            optimizer, scaler, dino_loss, device,
            strict=CONFIG["RESUME_STRICT"],
        )
    else:
        start_epoch, iteration = 0, 0

    if start_epoch >= CONFIG["EPOCHS"]:
        print(f"已达到目标 EPOCHS={CONFIG['EPOCHS']}，无需继续训练。")

    # ════════════════════════════════════════════════════════════════
    # 训练循环
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'─'*64}")
    print(f"  开始训练：epoch {start_epoch+1} → {CONFIG['EPOCHS']}")
    print(f"  batches/epoch: {iters_per_epoch}")
    print(f"{'─'*64}\n")

    for epoch in range(start_epoch, CONFIG["EPOCHS"]):
        student.train()
        crack_dense_head.train()
        crack_global_head.train()
        teacher.eval()

        cur_crack_alpha = get_crack_alpha(epoch)

        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']}  "
                              f"α_crack={cur_crack_alpha:.3f}")

        ep_dino = ep_dense = ep_global = 0.0
        ep_valid = n_batches = 0

        for batch in pbar:
            crops_batch, crack_img, patch_tgt, global_tgt, valid = batch

            crops        = [c.to(device, non_blocking=True) for c in crops_batch]
            global_crops = crops[:2]
            crack_img    = crack_img.to(device, non_blocking=True)     # (B,3,H,W)
            patch_tgt    = patch_tgt.to(device, non_blocking=True)     # (B,Np)
            global_tgt   = global_tgt.to(device, non_blocking=True)    # (B,4)
            valid_mask   = (valid.to(device) > 0)                      # (B,) bool
            n_valid      = int(valid_mask.sum().item())

            # ── Schedule ─────────────────────────────────────────────
            lr = lr_sched[iteration]
            wd = wd_sched[iteration]
            m  = mom_sched[iteration]
            for pg in optimizer.param_groups:
                pg["lr"] = lr
                if pg.get("weight_decay", 0.0) != 0.0:
                    pg["weight_decay"] = wd

            # ── Teacher forward（无梯度）──────────────────────────────
            with torch.no_grad():
                teacher_out = teacher(global_crops)

            # ── Student forward + losses ─────────────────────────────
            with autocast_ctx():
                # DINO loss（所有图）
                student_out = student(crops)
                loss_dino   = dino_loss(student_out, teacher_out, epoch)

                # crack loss（在空间对齐的 crack view 上，仅 valid 样本）
                loss_crack = torch.zeros((), device=device)
                loss_dense = torch.zeros((), device=device)
                loss_glob  = torch.zeros((), device=device)
                if cur_crack_alpha > 0 and n_valid > 0:
                    pooled, patch_tokens = forward_backbone_tokens(
                        student.backbone, crack_img
                    )
                    loss_crack, loss_dense, loss_glob = compute_crack_loss(
                        pooled, patch_tokens,
                        crack_dense_head, crack_global_head,
                        patch_tgt, global_tgt, valid_mask, pos_weight,
                    )

                total_loss = loss_dino + cur_crack_alpha * loss_crack

            # ── 反向传播 ────────────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()

            if epoch < CONFIG["FREEZE_LAST_LAYER"]:
                for n, p in student.named_parameters():
                    if "head.last" in n:
                        p.grad = None

            if CONFIG["CLIP_GRAD"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(student.parameters())
                    + list(crack_dense_head.parameters())
                    + list(crack_global_head.parameters()),
                    CONFIG["CLIP_GRAD"],
                )

            scaler.step(optimizer)
            scaler.update()

            ema_update(student, teacher, m)
            iteration += 1

            # ── 统计 ────────────────────────────────────────────────
            ep_dino   += float(loss_dino.item())
            ep_dense  += float(loss_dense.item())
            ep_global += float(loss_glob.item())
            ep_valid  += n_valid
            n_batches += 1

            pbar.set_postfix(
                dino=f"{loss_dino.item():.4f}",
                dense=f"{loss_dense.item():.4f}" if n_valid > 0 else "n/a",
                glob=f"{loss_glob.item():.4f}" if n_valid > 0 else "n/a",
                α=f"{cur_crack_alpha:.3f}",
                lr=f"{lr:.6f}",
            )

        avg_dino   = ep_dino   / max(n_batches, 1)
        avg_dense  = ep_dense  / max(n_batches, 1)
        avg_global = ep_global / max(n_batches, 1)
        print(f"  Epoch {epoch+1:3d}  "
              f"dino={avg_dino:.4f}  "
              f"dense={avg_dense:.4f}  "
              f"global={avg_global:.4f}  "
              f"valid_samples={ep_valid}  "
              f"α_crack={cur_crack_alpha:.3f}")

        if ((epoch + 1) % CONFIG["SAVE_EVERY_EPOCHS"] == 0
                or (epoch + 1) == CONFIG["EPOCHS"]):
            save_checkpoint(
                CONFIG["CHECKPOINT_PATH"],
                epoch_next        = epoch + 1,
                iteration         = iteration,
                student           = student,
                teacher           = teacher,
                crack_dense_head  = crack_dense_head,
                crack_global_head = crack_global_head,
                optimizer         = optimizer,
                scaler            = scaler,
                dino_loss         = dino_loss,
            )

    # ════════════════════════════════════════════════════════════════
    # 保存最终权重（两个 crack head 丢弃，只存 backbone）
    # ════════════════════════════════════════════════════════════════
    torch.save(student.backbone.state_dict(), CONFIG["OUT_BACKBONE"])
    torch.save(student.state_dict(),          CONFIG["OUT_STUDENT_FULL"])

    print(f"\n{'─'*64}")
    print(f"训练完成！")
    print(f"  Backbone（特征提取用）：{CONFIG['OUT_BACKBONE']}")
    print(f"  完整 student：          {CONFIG['OUT_STUDENT_FULL']}")
    print(f"  （crack dense/global head 已丢弃，推理不需要标注）")
    print(f"{'─'*64}")
    print(f"\n下一步：")
    print(f"  1. 修改 make_sfm_features.py：")
    print(f"       DINO_BACKBONE_PTH = \"{CONFIG['OUT_BACKBONE']}\"")
    print(f"     重新运行，生成新的 features_sfm.npz（仍是 397 维）。")
    print(f"  2. 用新 NPZ 训练 PPO，RL 代码无需任何改动（维度仍 397）。")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backbone", help="Pretrained pavement DINO backbone for crack supervision")
    parser.add_argument("--labels", action="append", default=[])
    args = parser.parse_args()
    dest = Path(args.output)
    CONFIG.update(UNLABELED_DIR=args.images, EXTRA_LABEL_DIRS=args.labels,
                  USE_PRETRAIN_BACKBONE=bool(args.backbone), PRETRAIN_BACKBONE_PTH=args.backbone,
                  OUT_BACKBONE=str(dest/"backbone.pth"), OUT_STUDENT_FULL=str(dest/"student.pth"),
                  CHECKPOINT_PATH=str(dest/"checkpoint.pth"))
    if not args.backbone:
        CONFIG["CRACK_ALPHA"] = 0.
    main()