"""
Expression Magnitude Scorer
=============================
Backbone feature만으로 표정 강도를 추정.
Neutral prototype과의 거리를 expression magnitude로 사용.
Peak frame selection에 활용 (inference time).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpressionMagnitudeScorer(nn.Module):
    """
    학습된 neutral centroid와의 거리로 expression magnitude 계산.
    Training 중에는 neutral class feature를 EMA로 업데이트.
    Inference 시에는 frozen centroid 사용.
    """

    def __init__(self, d_model: int = 384, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        # Neutral centroid (EMA updated during training)
        self.register_buffer("neutral_centroid", torch.zeros(d_model))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update_centroid(self, features: torch.Tensor, labels: torch.Tensor,
                        neutral_id: int):
        """
        Training step에서 호출. Neutral class features의 EMA 업데이트.

        Args:
            features: [B, d] global features
            labels: [B] class labels
            neutral_id: neutral class의 label id
        """
        mask = labels == neutral_id
        if mask.sum() == 0:
            return

        neutral_feats = features[mask].mean(dim=0)  # [d]

        if not self.initialized:
            self.neutral_centroid.copy_(neutral_feats)
            self.initialized.fill_(True)
        else:
            self.neutral_centroid.mul_(self.momentum).add_(
                neutral_feats, alpha=1 - self.momentum
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, d] global features from backbone
        Returns:
            magnitude: [B] expression magnitude scores (higher = more expressive)
        """
        # L2 distance from neutral centroid
        diff = features - self.neutral_centroid.unsqueeze(0)
        magnitude = torch.norm(diff, dim=-1)  # [B]
        return magnitude
