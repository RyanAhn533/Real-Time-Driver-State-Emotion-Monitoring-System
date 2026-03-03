"""
MobileViTv2/v3 backbone wrapper via timm.

핵심 변경: 기존 구조는 Patch Encoder + Global Encoder로 backbone을 2번 돌렸음.
새 구조는 backbone 1번만 돌리고, feature map에서 AU RoI를 추출함.

Available timm models (MobileViT family):
  - mobilevitv2_050, mobilevitv2_075, mobilevitv2_100, mobilevitv2_125, mobilevitv2_150
  - mobilevitv2_175, mobilevitv2_200
  
  Jetson 추천: mobilevitv2_100 (d=384, ~5M params, 224 input)
  고성능:      mobilevitv2_150 (d=576)  또는 mobilevitv2_200 (d=768)
"""

import torch
import torch.nn as nn
import timm


class MobileViTBackbone(nn.Module):
    """
    Single backbone that outputs:
      - feature_map: [B, C, H, W] for AU RoI extraction
      - global_feat: [B, C] via global average pooling
    """

    def __init__(self, model_name: str = "mobilevitv2_100", pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # timm의 forward_features → feature map 반환
        # mobilevitv2_100: output = [B, 384, 7, 7] for 224 input
        self.feat_dim = self._get_feat_dim()

        # Resolve normalization params
        data_cfg = timm.data.resolve_model_data_config(self.model)
        self.norm_mean = data_cfg.get("mean", (0.485, 0.456, 0.406))
        self.norm_std = data_cfg.get("std", (0.229, 0.224, 0.225))

    def _get_feat_dim(self) -> int:
        """Get feature dimension from model."""
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            feat = self.model.forward_features(dummy)
        return feat.shape[1]

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W] normalized image
        Returns:
            feat_map: [B, C, h, w] spatial feature map
            global_feat: [B, C] global average pooled feature
        """
        feat_map = self.model.forward_features(x)  # [B, C, h, w]
        global_feat = feat_map.mean(dim=[-2, -1])   # [B, C]
        return feat_map, global_feat

    def get_spatial_size(self, input_size: int = 224) -> int:
        """Feature map spatial size for given input."""
        dummy = torch.randn(1, 3, input_size, input_size)
        with torch.no_grad():
            feat = self.model.forward_features(dummy)
        return feat.shape[-1]  # assumes square
