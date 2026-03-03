"""
Loss Functions
===============
Focal Loss: class imbalance를 자연스럽게 처리.
기존 CE + class_weight + label_smoothing 조합 대체.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma > 0: 쉬운 샘플의 loss를 줄여서 어려운 샘플에 집중.
    alpha: per-class weighting (optional).
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] raw logits
            targets: [B] class indices
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # Gather target class probabilities
        targets_onehot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        p_t = (probs * targets_onehot).sum(dim=-1)       # [B]
        log_p_t = (log_probs * targets_onehot).sum(dim=-1)  # [B]

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma  # [B]

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # [B]
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_p_t  # [B]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(csv_path: str, label2id: Dict[str, int]) -> torch.Tensor:
    """
    Inverse-frequency class weights, normalized to mean=1.
    """
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts()
    maxn = counts.max()
    weights = []
    for lb in sorted(label2id.keys(), key=lambda x: label2id[x]):
        n = counts.get(lb, 1)
        weights.append(maxn / n)
    w = torch.tensor(weights, dtype=torch.float32)
    w = w * (len(w) / w.sum())  # normalize mean=1
    return w
