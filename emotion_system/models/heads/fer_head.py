"""FER Classification Head - extracts CLS token and classifies."""

import torch
import torch.nn as nn


class FERHead(nn.Module):
    """
    CLS token → emotion class logits.
    Pre-norm + 2-layer MLP.
    """

    def __init__(self, d_model: int = 384, num_classes: int = 7,
                 dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, K+2, d] fused tokens (CLS at index 0)
        Returns:
            logits: [B, num_classes]
        """
        cls_token = tokens[:, 0, :]  # [B, d]
        return self.head(cls_token)
