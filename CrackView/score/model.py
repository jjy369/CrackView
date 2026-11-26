from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models


class SubsetScoringNet(nn.Module):
    def __init__(
        self,
        pose_dim=13,
        hidden_dim=128,
        out_dim=2,
        backbone_name="resnet18",
        pretrained=True,
        train_backbone_from="layer4",
        view_layers=1,
        scene_layers=2,   # 子集级 Transformer 层数（仅 forward_batch 使用）
        num_heads=4,
        dropout=0.3,
        geo_dim=64,       # 几何特征的编码维度
    ):
        super().__init__()

        # ── 保存超参（用于 checkpoint 兼容性校验）─────────────────────────
        self.pose_dim            = pose_dim
        self.hidden_dim          = hidden_dim
        self.backbone_name       = backbone_name
        self.train_backbone_from = train_backbone_from
        self.view_layers         = view_layers
        self.scene_layers        = scene_layers
        self.num_heads           = num_heads
        self.dropout             = dropout
        self.geo_dim             = geo_dim

        # ── 图像编码器（ResNet backbone）──────────────────────────────────
        backbone, image_feat_dim = _build_backbone(backbone_name, pretrained)
        self.image_encoder       = backbone
        _set_backbone_trainable(self.image_encoder, train_backbone_from)

        # ── 视角级编码 ─────────────────────────────────────────────────────
        self.pose_encoder    = _build_pose_encoder(pose_dim, dropout)
        self.token_proj      = _build_token_proj(image_feat_dim, hidden_dim, dropout)
        self.cls_token       = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.view_transformer = _build_transformer(hidden_dim, num_heads, view_layers, dropout)

        # ── 几何覆盖编码器（新增）─────────────────────────────────────────
        # 输入：从 pose 计算的 8 维覆盖统计量（见 compute_geometry_features）
        # 作用：让模型感知"这批视角在空间上覆盖得好不好"
        #       几何特征对 RL 生成的任意新组合泛化能力远强于纯图像特征
        self.geometry_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, geo_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── 视觉 + 几何融合投影 ────────────────────────────────────────────
        # 把 CLS 视觉特征和几何特征拼接后投影回 hidden_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim + geo_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── 子集级 Transformer（仅 forward_batch 使用）────────────────────
        # N 个候选子集互相 attend，使模型能感知相对质量
        self.scene_transformer = _build_transformer(
            hidden_dim, num_heads, scene_layers, dropout
        )

        # ── 打分头（两种模式共用）─────────────────────────────────────────
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    # ── 核心前向：视角编码（两种模式共用）────────────────────────────────
    def _encode_subsets(self, images, poses, view_masks, img_chunk_size=32):
        """
        对 M 个子集做视角级编码，输出每个子集的融合特征。

        images     : [M, K, 3, H, W]
        poses      : [M, K, pose_dim]
        view_masks : [M, 1+K]  True=padding 视角

        返回: subset_feat [M, hidden_dim]  ← 视觉特征 + 几何特征融合后
        """
        M, K, C, H, W = images.shape

        # ── 图像特征（分块，避免 OOM）────────────────────────────────────
        imgs_2d  = images.reshape(M * K, C, H, W)
        img_feat = torch.cat(
            [self.image_encoder(chunk).flatten(1)
             for chunk in imgs_2d.split(img_chunk_size)],
            dim=0,
        ).reshape(M, K, -1)                         # [M, K, img_feat_dim]

        # ── Pose 特征 ─────────────────────────────────────────────────────
        pose_feat = self.pose_encoder(poses)         # [M, K, 128]

        # ── Token 投影 + CLS token + 视角级 Transformer ───────────────────
        token = self.token_proj(torch.cat([img_feat, pose_feat], dim=-1))
        cls   = self.cls_token.expand(M, 1, -1)
        token = torch.cat([cls, token], dim=1)       # [M, 1+K, hidden]
        token = self.view_transformer(token, src_key_padding_mask=view_masks)
        vis_feat = token[:, 0]                       # [M, hidden]  CLS 视觉特征

        # ── 几何覆盖特征 ──────────────────────────────────────────────────
        geo_raw  = compute_geometry_features(poses, view_masks)  # [M, 8]，从 pose 计算
        geo_feat = self.geometry_encoder(geo_raw)    # [M, geo_dim]

        # ── 融合 ──────────────────────────────────────────────────────────
        fused = self.fusion_proj(
            torch.cat([vis_feat, geo_feat], dim=-1)  # [M, hidden+geo_dim]
        )                                            # [M, hidden]

        return fused

    def forward_batch(self, images, poses, view_masks=None, subset_masks=None,
                      img_chunk_size=32):
        """
        场景 A：N 个候选子集联合评分。
        RL 每步生成 N 个候选时调用，scene_transformer 提供跨候选对比。

        images       : [B, N, K, 3, H, W]
        poses        : [B, N, K, pose_dim]
        view_masks   : [B, N, 1+K]  True=padding 视角
        subset_masks : [B, N]       True=padding 子集

        返回两个预测分数 [B, N, 2]。
        """
        B, N, K, C, H, W = images.shape

        # 把 B*N 个子集展平，批量编码
        images_flat     = images.reshape(B * N, K, C, H, W)
        poses_flat      = poses.reshape(B * N, K, poses.shape[-1])
        view_masks_flat = (view_masks.reshape(B * N, 1 + K)
                           if view_masks is not None else None)

        # 视角编码 + 几何融合 → [B*N, hidden]
        subset_feat = self._encode_subsets(
            images_flat, poses_flat, view_masks_flat, img_chunk_size
        ).reshape(B, N, -1)                          # [B, N, hidden]

        # 子集级 Transformer：N 个候选互相 attend，感知相对质量
        subset_feat = self.scene_transformer(
            subset_feat, src_key_padding_mask=subset_masks
        )                                            # [B, N, hidden]

        return self.score_head(subset_feat)           # [B, N, 3]

    def forward(self, images, poses, view_masks=None, subset_masks=None,
                img_chunk_size=32):
        """向后兼容接口，自动分发到 forward_batch。"""
        return self.forward_batch(images, poses, view_masks, subset_masks,
                                  img_chunk_size)


def compute_geometry_features(poses, masks=None):
    rows = []
    for i, pose in enumerate(poses):
        valid = pose if masks is None else pose[~masks[i, 1:]]
        if not len(valid):
            rows.append(pose.new_zeros(8))
            continue
        xyz = valid[:, 9:12]
        var = xyz.var(0, unbiased=False)
        dist = torch.pdist(xyz)
        zero = xyz.new_zeros(())
        dirs = torch.nn.functional.normalize(valid[:, 6:9], dim=1)
        axis = torch.nn.functional.normalize(dirs.mean(0), dim=0)
        angle = torch.acos((dirs @ axis).clamp(-1, 1)).mean()
        stats = [dist.mean() if dist.numel() else zero, dist.min() if dist.numel() else zero,
                 angle, xyz.new_tensor(len(valid) / 50), var.prod().pow(1/3)]
        rows.append(torch.cat([var, torch.stack(stats)]))
    return torch.stack(rows)


def with_composite(parts):
    final = 0.4 * parts[..., 0] + 0.6 * parts[..., 1]
    return torch.cat([parts, final.unsqueeze(-1)], dim=-1)


def _build_backbone(backbone_name, pretrained):
    if backbone_name == "resnet18":
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif backbone_name == "resnet34":
        backbone = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif backbone_name == "resnet50":
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    feat_dim = backbone.fc.in_features
    encoder  = nn.Sequential(OrderedDict([
        ("conv1",   backbone.conv1),
        ("bn1",     backbone.bn1),
        ("relu",    backbone.relu),
        ("maxpool", backbone.maxpool),
        ("layer1",  backbone.layer1),
        ("layer2",  backbone.layer2),
        ("layer3",  backbone.layer3),
        ("layer4",  backbone.layer4),
        ("avgpool", backbone.avgpool),
    ]))
    return encoder, feat_dim


def _set_backbone_trainable(encoder, train_backbone_from):
    for p in encoder.parameters():
        p.requires_grad = False
    if train_backbone_from in [None, "none"]:
        return
    trainable_map = {
        "layer4": ["layer4"],
        "layer3": ["layer3", "layer4"],
        "layer2": ["layer2", "layer3", "layer4"],
        "all":    ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    }
    if train_backbone_from not in trainable_map:
        raise ValueError(f"Unsupported train_backbone_from: {train_backbone_from}")
    for name, module in encoder.named_children():
        if name in trainable_map[train_backbone_from]:
            for p in module.parameters():
                p.requires_grad = True


def _build_pose_encoder(pose_dim, dropout):
    return nn.Sequential(
        nn.Linear(pose_dim, 64), nn.ReLU(inplace=True),
        nn.Linear(64, 128),      nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    )


def _build_token_proj(image_feat_dim, hidden_dim, dropout):
    return nn.Sequential(
        nn.Linear(image_feat_dim + 128, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    )


def _build_transformer(hidden_dim, num_heads, num_layers, dropout):
    layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim, nhead=num_heads,
        dim_feedforward=hidden_dim * 4, dropout=dropout,
        activation="gelu", batch_first=True, norm_first=False,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)