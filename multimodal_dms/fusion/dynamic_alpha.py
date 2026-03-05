"""
Dynamic Alpha Ensemble: Neural + LightGBM Hybrid
==================================================
Quality signals → α ∈ [0, 1].
final_pred = α · neural_pred + (1 - α) · lgbm_pred

Two modes:
  1. Learned: Small MLP gate (7d → 16 → 1, sigmoid)
  2. Rule-based: Hand-tuned heuristic (for when data is insufficient)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class DynamicAlpha(nn.Module):
    """
    Learned gate: quality signals → α ∈ [0, 1].
    Input: 7 quality signals.
    Output: α scalar per sample.
    """

    def __init__(self, in_dim: int = 7, hidden: int = 16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, quality_signals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            quality_signals: (B, 7) =
                [kfer_entropy, face_valid, audio_valid, bio_valid,
                 perclos_conf, neural_entropy, lgbm_margin]

        Returns:
            alpha: (B, 1) ∈ [0, 1]
        """
        return self.gate(quality_signals)


def alpha_rule_based(face_valid: np.ndarray,
                     audio_valid: np.ndarray,
                     neural_entropy: np.ndarray,
                     lgbm_margin: np.ndarray) -> np.ndarray:
    """
    Rule-based α computation (fallback when data insufficient).

    Args:
        face_valid:     (B,) bool
        audio_valid:    (B,) bool
        neural_entropy: (B,) float — entropy of neural prediction
        lgbm_margin:    (B,) float — margin between top-2 LGBM predictions

    Returns:
        alpha: (B,) float ∈ [0.1, 0.9]
    """
    B = len(face_valid)
    alpha = np.full(B, 0.5, dtype=np.float32)

    # Cross-modal available → neural gets more weight
    cross = face_valid & audio_valid
    alpha[cross] += 0.15

    # Neural uncertain → reduce neural weight
    high_ent = neural_entropy > 1.5
    alpha[high_ent] -= 0.2

    # LGBM confident → reduce neural weight
    high_margin = lgbm_margin > 0.4
    alpha[high_margin] -= 0.1

    # Clamp
    alpha = np.clip(alpha, 0.1, 0.9)

    return alpha


def build_quality_signals(kfer_entropy: torch.Tensor,
                          face_valid: torch.Tensor,
                          audio_valid: torch.Tensor,
                          bio_valid: torch.Tensor,
                          perclos_conf: torch.Tensor,
                          neural_pred: torch.Tensor,
                          lgbm_pred: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Build the 7-dim quality signal vector for DynamicAlpha.

    Args:
        kfer_entropy: (B,) K-FER prediction entropy
        face_valid:   (B,) bool
        audio_valid:  (B,) bool
        bio_valid:    (B,) bool
        perclos_conf: (B,) PERCLOS confidence
        neural_pred:  (B, 1) neural arousal prediction
        lgbm_pred:    (B,) optional LGBM prediction

    Returns:
        (B, 7) quality signals
    """
    B = kfer_entropy.shape[0]
    device = kfer_entropy.device

    # Neural entropy: entropy of the arousal prediction
    # For binary, entropy = -p*log(p) - (1-p)*log(1-p)
    p = neural_pred.squeeze(-1).clamp(1e-6, 1 - 1e-6)
    neural_ent = -(p * p.log() + (1 - p) * (1 - p).log())

    # LGBM margin: |lgbm_pred - 0.5| * 2 (how confident)
    if lgbm_pred is not None:
        lgbm_margin = (lgbm_pred - 0.5).abs() * 2
    else:
        lgbm_margin = torch.zeros(B, device=device)

    signals = torch.stack([
        kfer_entropy,
        face_valid.float(),
        audio_valid.float(),
        bio_valid.float(),
        perclos_conf,
        neural_ent,
        lgbm_margin,
    ], dim=1)  # (B, 7)

    return signals


class HybridPredictor:
    """
    Combines neural (KMERFusion) and LightGBM predictions.
    Supports both learned and rule-based alpha.
    """

    def __init__(self, mode: str = "rule_based", alpha_model: Optional[DynamicAlpha] = None):
        self.mode = mode
        self.alpha_model = alpha_model

    def predict(self,
                neural_pred: np.ndarray,
                lgbm_pred: np.ndarray,
                alpha: Optional[np.ndarray] = None,
                quality_signals: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Combine predictions.

        Args:
            neural_pred: (B,) neural arousal prediction [0, 1]
            lgbm_pred:   (B,) LGBM prediction [0, 1]
            alpha:       (B,) pre-computed alpha (if provided)
            quality_signals: (B, 7) for learned mode

        Returns:
            (B,) final prediction [0, 1]
        """
        if alpha is None:
            if self.mode == "learned" and self.alpha_model is not None:
                import torch
                with torch.no_grad():
                    q = torch.from_numpy(quality_signals).float()
                    alpha = self.alpha_model(q).squeeze(-1).numpy()
            else:
                alpha = np.full_like(neural_pred, 0.5)

        return alpha * neural_pred + (1 - alpha) * lgbm_pred
