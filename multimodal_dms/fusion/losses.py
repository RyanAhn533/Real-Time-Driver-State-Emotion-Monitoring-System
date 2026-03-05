"""
K-MER Loss Functions
=====================
1. Uncertainty-weighted MTL (Kendall et al., CVPR 2018)
2. KD losses (feature-level + logit-level)
3. Individual task losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class UncertaintyWeightedMTL(nn.Module):
    """
    Uncertainty-weighted multi-task loss (Kendall et al., CVPR 2018).
    L_total = Σ (1/(2σ²_i)) * L_i + log(σ_i)
    Learns σ_i (uncertainty) per task automatically.
    """

    def __init__(self, task_names=("arousal", "valence", "drowsy"),
                 initial_log_var: float = 0.0):
        super().__init__()
        self.task_names = task_names
        # Learnable log variance per task
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(initial_log_var))
            for name in task_names
        })

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            losses: dict of {task_name: scalar loss}

        Returns:
            total weighted loss (scalar)
        """
        total = 0.0
        for name in self.task_names:
            if name in losses:
                log_var = self.log_vars[name]
                precision = torch.exp(-log_var)
                total = total + precision * losses[name] + log_var
        return total

    def get_weights(self) -> Dict[str, float]:
        """Current task weights (1/(2σ²))."""
        return {
            name: float(0.5 * torch.exp(-self.log_vars[name]).item())
            for name in self.task_names
        }


class BinaryFocalLoss(nn.Module):
    """Binary focal loss for imbalanced classification."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 1) sigmoid output [0, 1]
            target: (B,) binary labels {0, 1}

        Returns:
            scalar focal loss
        """
        pred = pred.squeeze(-1).clamp(1e-6, 1 - 1e-6)
        target = target.float()

        bce = F.binary_cross_entropy(pred, target, reduction='none')

        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma

        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)

        loss = alpha_t * focal_weight * bce
        return loss.mean()


class KDLoss(nn.Module):
    """
    Knowledge Distillation loss components:
      1. Feature-level MSE (normalized repr alignment)
      2. Logit-level KL (soft label matching)
    """

    def __init__(self, temperature: float = 4.0, feat_weight: float = 0.3,
                 logit_weight: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.feat_weight = feat_weight
        self.logit_weight = logit_weight

    def feature_kd(self, student_repr: torch.Tensor,
                   teacher_repr: torch.Tensor) -> torch.Tensor:
        """MSE between L2-normalized representations."""
        s = F.normalize(student_repr, p=2, dim=-1)
        t = F.normalize(teacher_repr, p=2, dim=-1)
        return F.mse_loss(s, t)

    def logit_kd(self, student_logit: torch.Tensor,
                 teacher_logit: torch.Tensor) -> torch.Tensor:
        """KL divergence with temperature scaling."""
        T = self.temperature
        s = F.log_softmax(student_logit / T, dim=-1)
        t = F.softmax(teacher_logit / T, dim=-1)
        return F.kl_div(s, t, reduction='batchmean') * (T ** 2)

    def forward(self, student_repr, teacher_repr,
                student_logit=None, teacher_logit=None) -> torch.Tensor:
        loss = self.feat_weight * self.feature_kd(student_repr, teacher_repr)
        if student_logit is not None and teacher_logit is not None:
            loss = loss + self.logit_weight * self.logit_kd(student_logit, teacher_logit)
        return loss


class KMERLoss(nn.Module):
    """
    Complete K-MER loss: uncertainty-weighted MTL + optional KD.

    Tasks:
      - arousal:  Binary focal loss (PRIMARY)
      - valence:  Binary focal loss (OPTIONAL)
      - drowsy:   CrossEntropy (3-class)
    """

    def __init__(self,
                 focal_gamma: float = 2.0,
                 focal_alpha_arousal: float = 0.5,
                 focal_alpha_valence: float = 0.5,
                 use_valence: bool = True,
                 use_drowsy: bool = True,
                 use_kd: bool = False,
                 kd_temperature: float = 4.0):
        super().__init__()

        task_names = ["arousal"]
        if use_valence:
            task_names.append("valence")
        if use_drowsy:
            task_names.append("drowsy")

        self.arousal_loss = BinaryFocalLoss(focal_gamma, focal_alpha_arousal)

        if use_valence:
            self.valence_loss = BinaryFocalLoss(focal_gamma, focal_alpha_valence)

        if use_drowsy:
            self.drowsy_loss = nn.CrossEntropyLoss()

        self.mtl = UncertaintyWeightedMTL(task_names)

        self.use_valence = use_valence
        self.use_drowsy = use_drowsy
        self.use_kd = use_kd

        if use_kd:
            self.kd_loss = KDLoss(temperature=kd_temperature)

    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                teacher_cache: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: model outputs (arousal, valence, drowsy, fused_repr)
            targets: {arousal: (B,), valence: (B,), drowsy: (B,)}
            teacher_cache: {fused_repr: (B, d), arousal_logit: (B, 1)} (optional)

        Returns:
            {total: scalar, arousal: scalar, valence: scalar, drowsy: scalar, ...}
        """
        losses = {}

        # Arousal (PRIMARY)
        losses["arousal"] = self.arousal_loss(outputs["arousal"], targets["arousal"])

        # Valence (OPTIONAL)
        if self.use_valence and "valence" in outputs and "valence" in targets:
            losses["valence"] = self.valence_loss(outputs["valence"], targets["valence"])

        # Drowsy
        if self.use_drowsy and "drowsy" in outputs and "drowsy" in targets:
            losses["drowsy"] = self.drowsy_loss(outputs["drowsy"], targets["drowsy"])

        # Uncertainty-weighted total
        total = self.mtl(losses)

        # KD (optional)
        if self.use_kd and teacher_cache is not None:
            kd = self.kd_loss(
                outputs["fused_repr"],
                teacher_cache["fused_repr"],
                outputs.get("arousal"),
                teacher_cache.get("arousal_logit"),
            )
            total = total + kd
            losses["kd"] = kd

        losses["total"] = total
        losses["mtl_weights"] = self.mtl.get_weights()

        return losses
