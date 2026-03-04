"""
FER Expert Wrapper
====================
Pipeline Layer 2.1.

Wraps the trained AUFERModel to output standardized signals:
  - p_FER(e): emotion class probabilities [B, C]
  - VA_FER: face-based arousal/valence estimate [B, 2] (optional)
  - q_face: face quality score [B] (blur, occlusion, face_ratio)
  - u_FER: uncertainty (softmax entropy) [B]

This module does NOT modify the original AUFERModel.
It wraps it and adds quality/uncertainty computation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

from models.fer_model import AUFERModel


class FERExpert:
    """
    Stateless wrapper around trained AUFERModel.
    Adds quality estimation and uncertainty computation.
    """

    def __init__(self, model: AUFERModel, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        self.num_classes = model.head.head[-1].out_features

    @torch.no_grad()
    def predict(self, images: torch.Tensor,
                au_coords: torch.Tensor,
                face_detected: Optional[torch.Tensor] = None,
                blur_score: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Full FER Expert prediction with quality and uncertainty.

        Args:
            images: [B, 3, H, W]
            au_coords: [B, K, 2]
            face_detected: [B] bool, whether face was detected (for quality)
            blur_score: [B] laplacian variance (higher = sharper)

        Returns dict:
            emotion_logits: [B, C]
            emotion_probs: [B, C]
            quality: [B] face quality score in [0, 1]
            uncertainty: [B] normalized entropy in [0, 1]
        """
        B = images.size(0)

        logits = self.model(images.to(self.device),
                            au_coords.to(self.device))  # [B, C]

        probs = F.softmax(logits, dim=-1)

        # Uncertainty: normalized softmax entropy
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)  # [B]
        max_entropy = np.log(self.num_classes)
        uncertainty = entropy / max_entropy  # [0, 1]

        # Quality estimation
        quality = self._compute_quality(
            au_coords, face_detected, blur_score, B
        )

        return {
            "emotion_logits": logits,     # [B, C]
            "emotion_probs": probs,       # [B, C]
            "quality": quality,           # [B]
            "uncertainty": uncertainty,    # [B]
        }

    def _compute_quality(self,
                         au_coords: torch.Tensor,
                         face_detected: Optional[torch.Tensor],
                         blur_score: Optional[torch.Tensor],
                         B: int) -> torch.Tensor:
        """
        Heuristic face quality score based on:
        1. AU coordinate validity (all within image bounds)
        2. Face detection success
        3. Image sharpness (blur score)

        Returns: [B] quality score in [0, 1]
        """
        scores = torch.ones(B, device=self.device)

        # Check AU coordinates are within valid range
        valid_coords = (au_coords >= 0).all(dim=-1).all(dim=-1).float()
        coords_in_range = (au_coords <= 224).all(dim=-1).all(dim=-1).float()
        scores = scores * valid_coords * coords_in_range

        # Face detection
        if face_detected is not None:
            scores = scores * face_detected.float().to(self.device)

        # Blur score: normalize to [0, 1] with heuristic threshold
        if blur_score is not None:
            blur_q = (blur_score.to(self.device) / 500.0).clamp(0, 1)
            scores = scores * blur_q

        return scores

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: str = "cuda",
                        **model_kwargs) -> "FERExpert":
        """Load FER Expert from saved checkpoint."""
        dev = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

        cfg = ckpt.get("config", {})
        mcfg = cfg.get("model", {})

        model = AUFERModel(
            backbone_name=mcfg.get("backbone", "mobilevitv2_100"),
            pretrained=False,
            num_au=model_kwargs.get("num_au", 8),
            num_classes=mcfg.get("num_classes", 7),
            d_emb=mcfg.get("d_emb", 384),
            n_heads=mcfg.get("n_heads", 8),
            n_fusion_layers=mcfg.get("n_fusion_layers", 1),
            roi_mode=mcfg.get("roi_mode", "bilinear"),
            roi_spatial=mcfg.get("roi_spatial", 1),
            dropout=mcfg.get("dropout", 0.1),
            head_dropout=mcfg.get("head_dropout", 0.2),
            gate_init=mcfg.get("gate_init", 0.0),
        )

        state = ckpt.get("model_state_dict", ckpt.get("model", {}))
        model.load_state_dict(state, strict=False)
        model.to(dev)

        return cls(model, dev)
