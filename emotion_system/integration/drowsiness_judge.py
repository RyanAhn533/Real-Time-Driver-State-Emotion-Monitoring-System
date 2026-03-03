"""
Drowsiness Judge: PERCLOS + Arousal → Drowsiness Level
========================================================
PERCLOS: FaceMesh 전처리 단계에서 추출된 EAR 기반 눈 감김 비율.
Arousal: 음성/생체 센서에서 추출 (optional).

Levels: 0=alert, 1=drowsy, 2=sleeping
"""

from typing import Optional
from models.drowsiness.perclos import compute_perclos, DrowsinessClassifier


class DrowsinessJudge:
    """
    PERCLOS + Arousal로 졸음 수준 판단.

    사용:
      judge = DrowsinessJudge()
      # 매 프레임 EAR 추가
      judge.push_ear(0.25)
      judge.push_ear(0.18)
      ...
      # 1초 윈도우 후 판단
      level = judge.judge(arousal=0.3)
    """

    def __init__(self, window_size: int = 30, ear_threshold: float = 0.21,
                 use_arousal: bool = False):
        self.window_size = window_size
        self.ear_threshold = ear_threshold
        self.use_arousal = use_arousal
        self.ear_buffer = []
        self.classifier = DrowsinessClassifier(
            use_arousal=use_arousal, mode="rule_based"
        )

    def push_ear(self, ear: float):
        """EAR 값 추가 (매 프레임 호출)."""
        self.ear_buffer.append(ear)
        # Keep only latest window
        if len(self.ear_buffer) > self.window_size * 3:
            self.ear_buffer = self.ear_buffer[-self.window_size * 2:]

    def judge(self, arousal: Optional[float] = None) -> int:
        """
        현재 윈도우의 PERCLOS로 졸음 판단.
        Returns: 0=alert, 1=drowsy, 2=sleeping
        """
        window = self.ear_buffer[-self.window_size:]
        if len(window) < self.window_size // 2:
            return 0  # not enough data

        perclos = compute_perclos(window, self.ear_threshold)
        return self.classifier.forward_rule(perclos, arousal)

    def get_perclos(self) -> float:
        """현재 윈도우 PERCLOS 반환."""
        window = self.ear_buffer[-self.window_size:]
        if not window:
            return 0.0
        return compute_perclos(window, self.ear_threshold)

    def reset(self):
        self.ear_buffer.clear()
