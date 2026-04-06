"""
Cross-Label Knowledge Distillation: HSEmotion(8cls) → K-FER(7cls)
==================================================================
Different label sets → soft mapping matrix M ∈ R^(8×7).

HSEmotion 8cls     →  K-FER 7cls      근거
Angry              →  angry            직접
Contempt           →  hurt(0.7)+angry(0.3)  경멸≈상처
Disgust            →  angry(0.7)+hurt(0.3)  혐오→분노
Fear               →  anxious          공포≈불안
Happy              →  happy            직접
Neutral            →  neutral          직접
Sad                →  sad              직접
Surprise           →  surprised        직접

KD: teacher_7cls = hse_probs @ M → KL(kfer || teacher_7cls/T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Soft mapping matrix M: (8, 7)
# HSEmotion labels: [Angry, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise]
# K-FER labels:     [angry, anxious, happy, hurt, neutral, sad, surprised]
#                    [  0  ,   1   ,   2  ,  3  ,   4   ,  5 ,    6    ]
SOFT_MAPPING_MATRIX = torch.tensor([
    #   angry  anxious  happy  hurt  neutral  sad  surprised
    [1.0,    0.0,    0.0,   0.0,   0.0,   0.0,   0.0],  # Angry→angry
    [0.3,    0.0,    0.0,   0.7,   0.0,   0.0,   0.0],  # Contempt→hurt+angry
    [0.7,    0.0,    0.0,   0.3,   0.0,   0.0,   0.0],  # Disgust→angry+hurt
    [0.0,    1.0,    0.0,   0.0,   0.0,   0.0,   0.0],  # Fear→anxious
    [0.0,    0.0,    1.0,   0.0,   0.0,   0.0,   0.0],  # Happy→happy
    [0.0,    0.0,    0.0,   0.0,   1.0,   0.0,   0.0],  # Neutral→neutral
    [0.0,    0.0,    0.0,   0.0,   0.0,   1.0,   0.0],  # Sad→sad
    [0.0,    0.0,    0.0,   0.0,   0.0,   0.0,   1.0],  # Surprise→surprised
], dtype=torch.float32)


class CrossLabelKD(nn.Module):
    """
    Cross-label knowledge distillation.
    Maps HSEmotion 8-class probs to K-FER 7-class space via soft mapping.
    Computes KL divergence loss for alignment.
    """

    def __init__(self, temperature: float = 4.0,
                 mapping_matrix: torch.Tensor = None):
        super().__init__()
        M = mapping_matrix if mapping_matrix is not None else SOFT_MAPPING_MATRIX
        self.register_buffer("M", M)  # (8, 7)
        self.temperature = temperature

    def map_to_kfer_space(self, hse_probs: torch.Tensor) -> torch.Tensor:
        """
        Map HSEmotion 8-class probs to K-FER 7-class distribution.

        Args:
            hse_probs: (B, 8) — HSEmotion softmax probabilities

        Returns:
            teacher_7cls: (B, 7) — soft teacher distribution in K-FER space
        """
        # hse_probs @ M → (B, 7)
        teacher = hse_probs @ self.M
        # Renormalize to ensure valid distribution
        teacher = teacher / (teacher.sum(dim=-1, keepdim=True) + 1e-8)
        return teacher

    def forward(self, kfer_logits: torch.Tensor,
                hse_probs: torch.Tensor) -> torch.Tensor:
        """
        Cross-label KD loss.

        Args:
            kfer_logits: (B, 7) — K-FER logits (pre-softmax)
            hse_probs:   (B, 8) — HSEmotion softmax probs

        Returns:
            KL divergence loss (scalar)
        """
        T = self.temperature

        # Map HSE → K-FER space
        teacher_7cls = self.map_to_kfer_space(hse_probs)

        # Temperature-scaled distributions
        student = F.log_softmax(kfer_logits / T, dim=-1)
        teacher = (teacher_7cls + 1e-8).log()  # Already a distribution

        return F.kl_div(student, teacher.exp(), reduction='batchmean') * (T ** 2)


def verify_mapping_matrix():
    """Verify M has correct properties."""
    M = SOFT_MAPPING_MATRIX
    # Each row should sum to 1 (complete mapping)
    row_sums = M.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(8), atol=1e-5), \
        f"Row sums should be 1.0: {row_sums}"

    # All values >= 0
    assert (M >= 0).all(), "M should be non-negative"

    # Test mapping
    kd = CrossLabelKD()
    uniform_hse = torch.ones(1, 8) / 8
    teacher = kd.map_to_kfer_space(uniform_hse)
    assert teacher.shape == (1, 7), f"Expected (1,7), got {teacher.shape}"
    assert torch.allclose(teacher.sum(), torch.tensor(1.0), atol=1e-5)

    print("[CrossLabelKD] Mapping matrix verified ✓")
    return True
