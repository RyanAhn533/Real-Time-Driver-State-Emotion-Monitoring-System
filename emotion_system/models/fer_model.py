"""
AU-Based FER Model (Single Backbone Architecture)
===================================================

Architecture:
  Input Image (224×224)
      ↓
  MobileViTv2 Backbone (1x forward)
      ↓
  Feature Map [B, C, h, w]
      ├── Global Avg Pool → [B, C] (Global token)
      └── AU RoI Extract  → [B, K, C] (AU tokens)
      ↓
  [CLS] + [Global] + [AU_1, ..., AU_K]
      ↓
  Cross-Attention Fusion (1-2 layers)
      ↓
  CLS token → FER Head → num_classes logits

기존 대비 개선점:
  - Backbone 1회 forward (기존 321회 → 1회)
  - Feature map 레벨 AU 추출 (spatial context 보존)
  - Pre-norm Cross → Self attention 구조
  - Per-dim gated residual
"""

import torch
import torch.nn as nn

from .backbones import MobileViTBackbone
from .fusion import AURoIExtractor, CrossAttentionFusion
from .heads import FERHead, ExpressionMagnitudeScorer


class AUFERModel(nn.Module):
    """
    Single-backbone AU-based Facial Expression Recognition model.
    """

    def __init__(self,
                 backbone_name: str = "mobilevitv2_100",
                 pretrained: bool = True,
                 num_au: int = 6,
                 num_classes: int = 7,
                 d_emb: int = 384,
                 n_heads: int = 8,
                 n_fusion_layers: int = 1,
                 roi_mode: str = "bilinear",
                 roi_spatial: int = 1,
                 dropout: float = 0.1,
                 head_dropout: float = 0.2,
                 gate_init: float = 0.0,
                 img_size: int = 224):
        super().__init__()

        self.img_size = img_size
        self.num_au = num_au

        # ── Backbone (shared, single forward pass) ──
        self.backbone = MobileViTBackbone(backbone_name, pretrained)
        feat_dim = self.backbone.feat_dim

        # Project backbone dim to d_emb if different
        self.proj = nn.Identity()
        if feat_dim != d_emb:
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, d_emb),
                nn.LayerNorm(d_emb),
            )
            # Feature map projection (1x1 conv for spatial features)
            self.feat_proj = nn.Sequential(
                nn.Conv2d(feat_dim, d_emb, 1, bias=False),
                nn.BatchNorm2d(d_emb),
            )
        else:
            self.feat_proj = nn.Identity()

        # ── AU RoI Extractor ──
        self.au_roi = AURoIExtractor(
            feat_dim=d_emb, num_au=num_au,
            mode=roi_mode, roi_spatial=roi_spatial
        )

        # ── CLS token ──
        self.cls_token = nn.Parameter(torch.zeros(1, d_emb))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Cross-Attention Fusion ──
        self.fusion = CrossAttentionFusion(
            d_model=d_emb, n_heads=n_heads,
            n_layers=n_fusion_layers, dropout=dropout,
            gate_init=gate_init
        )

        # ── Classification Head ──
        self.head = FERHead(d_model=d_emb, num_classes=num_classes,
                            dropout=head_dropout)

        # ── Expression Magnitude Scorer (optional, for inference) ──
        self.expr_scorer = ExpressionMagnitudeScorer(d_model=d_emb)

    def forward(self, images: torch.Tensor, au_coords: torch.Tensor,
                return_features: bool = False):
        """
        Args:
            images:    [B, 3, H, W] normalized face images
            au_coords: [B, K, 2] AU center pixel coordinates (img_size 기준)
            return_features: if True, also return global_feat for expr magnitude

        Returns:
            logits: [B, num_classes]
            (optional) global_feat: [B, d_emb]
        """
        B = images.shape[0]

        # ── 1. Single backbone forward ──
        feat_map, global_feat = self.backbone(images)
        # feat_map: [B, C_backbone, h, w], global_feat: [B, C_backbone]

        # ── 2. Project to d_emb ──
        feat_map_proj = self.feat_proj(feat_map)      # [B, d_emb, h, w]
        global_feat_proj = self.proj(global_feat)      # [B, d_emb]

        # ── 3. AU RoI extraction from feature map ──
        au_tokens = self.au_roi(feat_map_proj, au_coords, self.img_size)
        # au_tokens: [B, K, d_emb]

        # ── 4. CLS token ──
        cls = self.cls_token.expand(B, -1)  # [B, d_emb]

        # ── 5. Cross-Attention Fusion ──
        tokens = self.fusion(cls, global_feat_proj, au_tokens)
        # tokens: [B, K+2, d_emb]

        # ── 6. Classification ──
        logits = self.head(tokens)  # [B, num_classes]

        if return_features:
            return logits, global_feat_proj
        return logits

    def get_expr_magnitude(self, images: torch.Tensor) -> torch.Tensor:
        """
        Backbone only → expression magnitude score.
        For peak frame selection at inference time.
        Does NOT run attention/head (fast).
        """
        _, global_feat = self.backbone(images)
        global_feat_proj = self.proj(global_feat)
        return self.expr_scorer(global_feat_proj)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_param_groups(self, base_lr: float, backbone_lr_scale: float = 0.1):
        """
        Differentiated learning rates:
        - backbone: base_lr * backbone_lr_scale
        - everything else: base_lr
        """
        backbone_params = set(id(p) for p in self.backbone.parameters())
        groups = [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad],
             "lr": base_lr * backbone_lr_scale, "name": "backbone"},
            {"params": [p for p in self.parameters()
                        if p.requires_grad and id(p) not in backbone_params],
             "lr": base_lr, "name": "head"},
        ]
        return [g for g in groups if len(g["params"]) > 0]
