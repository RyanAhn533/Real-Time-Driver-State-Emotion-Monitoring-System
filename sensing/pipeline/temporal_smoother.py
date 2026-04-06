"""
Multimodal Temporal Smoother
=============================
KMERFusion 출력에 대한 temporal smoothing.

- 이산 출력 (emotion_code, drowsy, compound_id): majority vote (sliding window)
- 연속 출력 (arousal, valence): EMA (exponential moving average)

Usage:
    smoother = MultimodalTemporalSmoother(
        emotion_window=7,
        drowsy_window=5,
        compound_window=7,
        ema_alpha=0.3,
    )
    smoothed = smoother.smooth(result_dict)
"""

from collections import Counter, deque
from typing import Dict, Optional


class MajorityVoteSmoother:
    """Sliding window majority vote for discrete values."""

    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def push(self, value) -> object:
        """값 추가 후 majority vote 결과 반환."""
        self.buffer.append(value)
        counts = Counter(self.buffer)
        return counts.most_common(1)[0][0]

    def get_confidence(self) -> float:
        """현재 윈도우에서 majority의 비율."""
        if not self.buffer:
            return 0.0
        counts = Counter(self.buffer)
        return counts.most_common(1)[0][1] / len(self.buffer)

    def reset(self):
        self.buffer.clear()


class EMASmoother:
    """Exponential Moving Average for continuous values."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value: Optional[float] = None

    def push(self, x: float) -> float:
        """새 값 입력 후 smoothed 값 반환."""
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value

    def get(self) -> Optional[float]:
        return self.value

    def reset(self):
        self.value = None


class MultimodalTemporalSmoother:
    """
    KMERFusion 출력 전체에 대한 temporal smoothing.

    이산 출력: majority vote
    연속 출력: EMA
    """

    def __init__(
        self,
        emotion_window: int = 7,
        drowsy_window: int = 5,
        compound_window: int = 7,
        ema_alpha: float = 0.3,
    ):
        # Majority vote smoothers
        self.emotion_smoother = MajorityVoteSmoother(emotion_window)
        self.drowsy_smoother = MajorityVoteSmoother(drowsy_window)
        self.compound_smoother = MajorityVoteSmoother(compound_window)

        # EMA smoothers
        self.arousal_smoother = EMASmoother(ema_alpha)
        self.valence_smoother = EMASmoother(ema_alpha)

    def smooth(self, result: Dict) -> Dict:
        """
        KMERInferencer.forward() 출력을 smoothing.

        Args:
            result: KMERInferencer.forward()의 반환값 dict
                Keys: arousal, valence, drowsy, compound_id, compound_label,
                      kfer_emotion, kfer_confidence, face_detected

        Returns:
            smoothed result dict (원본 키 유지 + smoothed_ 접두어 키 추가)
        """
        smoothed = dict(result)  # 원본 보존

        # 얼굴 미검출 시 smoothing skip
        if not result.get("face_detected", False):
            return smoothed

        # K-FER emotion (이산) — kfer_emotion은 문자열
        kfer_emo = result.get("kfer_emotion")
        if kfer_emo is not None:
            smoothed["smoothed_kfer_emotion"] = self.emotion_smoother.push(kfer_emo)
            smoothed["smoothed_emotion_confidence"] = self.emotion_smoother.get_confidence()

        # Arousal (연속)
        arousal = result.get("arousal")
        if arousal is not None:
            smoothed["smoothed_arousal"] = self.arousal_smoother.push(float(arousal))

        # Valence (연속)
        valence = result.get("valence")
        if valence is not None:
            smoothed["smoothed_valence"] = self.valence_smoother.push(float(valence))

        # Drowsy (이산: 0, 1, 2)
        drowsy = result.get("drowsy")
        if drowsy is not None:
            smoothed["smoothed_drowsy"] = self.drowsy_smoother.push(int(drowsy))

        # Compound emotion (이산: 0~12)
        compound_id = result.get("compound_id")
        compound_label = result.get("compound_label")
        if compound_id is not None:
            smoothed["smoothed_compound_id"] = self.compound_smoother.push(compound_id)
        if compound_label is not None:
            # compound_label도 majority vote (문자열)
            # compound_id와 동기화를 위해 compound_id 기반으로 smoothing
            pass  # compound_id → label 매핑은 E2E pipeline에서 처리

        return smoothed

    def reset(self):
        """모든 smoother 초기화."""
        self.emotion_smoother.reset()
        self.drowsy_smoother.reset()
        self.compound_smoother.reset()
        self.arousal_smoother.reset()
        self.valence_smoother.reset()


# ── Self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== MultimodalTemporalSmoother Test ===\n")

    smoother = MultimodalTemporalSmoother(
        emotion_window=5, drowsy_window=3, ema_alpha=0.3
    )

    # 시뮬레이션: 연속 프레임
    test_results = [
        {"face_detected": True, "kfer_emotion": "happy", "arousal": 0.7,
         "valence": 0.8, "drowsy": 0, "compound_id": 2, "compound_label": "happy"},
        {"face_detected": True, "kfer_emotion": "happy", "arousal": 0.65,
         "valence": 0.75, "drowsy": 0, "compound_id": 2, "compound_label": "happy"},
        {"face_detected": True, "kfer_emotion": "neutral", "arousal": 0.5,
         "valence": 0.6, "drowsy": 0, "compound_id": 0, "compound_label": "neutral"},
        {"face_detected": True, "kfer_emotion": "happy", "arousal": 0.72,
         "valence": 0.82, "drowsy": 0, "compound_id": 2, "compound_label": "happy"},
        {"face_detected": True, "kfer_emotion": "happy", "arousal": 0.68,
         "valence": 0.78, "drowsy": 1, "compound_id": 2, "compound_label": "happy"},
    ]

    for i, r in enumerate(test_results):
        s = smoother.smooth(r)
        print(f"Frame {i}: raw_emo={r['kfer_emotion']:>8s} → "
              f"smoothed={s.get('smoothed_kfer_emotion', '-'):>8s} | "
              f"arousal={r['arousal']:.2f} → {s.get('smoothed_arousal', 0):.2f} | "
              f"drowsy={r['drowsy']} → {s.get('smoothed_drowsy', '-')}")
