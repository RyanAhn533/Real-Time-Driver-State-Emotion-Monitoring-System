"""
PERCLOS-based Drowsiness Detection
====================================
FaceMesh AU extraction 단계에서 EAR (Eye Aspect Ratio) 계산.
PERCLOS = 1초 윈도우에서 눈 감긴 비율.

이 모듈은 두 가지 모드:
  1. Feature extraction: FaceMesh landmarks → EAR → PERCLOS (preprocess time)
  2. Classification: PERCLOS + optional Arousal → drowsiness level (inference time)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


# MediaPipe FaceMesh Eye Landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.21   # Below this = eyes closed
PERCLOS_THRESHOLD = 0.8  # Above this in 1-sec window = drowsy


def compute_ear(landmarks, eye_indices: List[int], w: int, h: int) -> float:
    """
    Eye Aspect Ratio (EAR) from FaceMesh landmarks.
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * w, lm.y * h))

    # p1=pts[0], p2=pts[1], p3=pts[2], p4=pts[3], p5=pts[4], p6=pts[5]
    def dist(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    vertical_1 = dist(pts[1], pts[5])
    vertical_2 = dist(pts[2], pts[4])
    horizontal = dist(pts[0], pts[3])

    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_perclos(ear_sequence: List[float], threshold: float = EAR_THRESHOLD) -> float:
    """
    PERCLOS: fraction of time eyes are closed in a window.
    ear_sequence: list of EAR values over time (e.g., 30 frames = 1 sec at 30fps)
    """
    if len(ear_sequence) == 0:
        return 0.0
    closed = sum(1 for ear in ear_sequence if ear < threshold)
    return closed / len(ear_sequence)


class DrowsinessClassifier(nn.Module):
    """
    Simple classifier: PERCLOS (+ optional Arousal) → drowsiness level.

    Levels: 0=alert, 1=drowsy, 2=sleeping

    두 가지 모드:
      - rule_based: 단순 threshold 기반 (딥러닝 불필요)
      - learned: 작은 MLP로 분류 (Arousal 포함 시)
    """

    def __init__(self, use_arousal: bool = False, mode: str = "rule_based"):
        super().__init__()
        self.use_arousal = use_arousal
        self.mode = mode

        if mode == "learned":
            in_dim = 2 if use_arousal else 1  # PERCLOS + optional Arousal
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 3),  # 3 levels
            )

    def forward_rule(self, perclos: float, arousal: Optional[float] = None) -> int:
        """
        Rule-based drowsiness detection.
        Returns: 0=alert, 1=drowsy, 2=sleeping
        """
        # Arousal이 낮으면 drowsiness threshold를 낮춤
        threshold_drowsy = 0.4
        threshold_sleeping = 0.8

        if arousal is not None and arousal < 0.3:
            threshold_drowsy *= 0.7   # 더 민감하게
            threshold_sleeping *= 0.8

        if perclos >= threshold_sleeping:
            return 2
        elif perclos >= threshold_drowsy:
            return 1
        return 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Learned mode.
        features: [B, 1] PERCLOS or [B, 2] PERCLOS + Arousal
        Returns: [B, 3] logits
        """
        if self.mode == "rule_based":
            raise RuntimeError("Use forward_rule() for rule-based mode")
        return self.mlp(features)
