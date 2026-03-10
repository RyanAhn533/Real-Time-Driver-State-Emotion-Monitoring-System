"""
Fatigue Tracker
================
운전자 만성 피로/저각성 상태 판정.

판정 조건 (OR):
  1. compound_label ∈ {"depressed", "calm"} 연속 30초 이상
  2. arousal < 0.3 연속 30초 이상
  3. drowsy 상태 연속 10초 이상

Sensor missing 처리:
  - Bio missing → 조건 2 불가 → 조건 1, 3만으로 판정
  - Cam missing → 조건 1, 3 불가 → False

Usage:
    tracker = FatigueTracker(
        duration_sec=30, arousal_thresh=0.3, drowsy_sec=10, hz=10
    )
    is_fatigued = tracker.update(compound_label, arousal, drowsy)
"""

import sys
from pathlib import Path
from typing import Optional

# sensing/ 디렉토리를 sys.path에 추가 (직접 실행 지원)
_SENSING_DIR = str(Path(__file__).resolve().parent.parent)
if _SENSING_DIR not in sys.path:
    sys.path.insert(0, _SENSING_DIR)

from core.logger import get_logger

logger = get_logger("kmer.pipeline.fatigue")


class FatigueTracker:
    """
    지속적 저각성/우울 상태 기반 Fatigue 판정기.

    10-class 출력의 10번째 항목 (Fatigue)을 담당.
    """

    def __init__(
        self,
        duration_sec: float = 30.0,
        arousal_thresh: float = 0.3,
        drowsy_sec: float = 10.0,
        hz: float = 10.0,
    ):
        """
        Args:
            duration_sec: compound/arousal 지속 시간 (초)
            arousal_thresh: 저각성 판정 threshold
            drowsy_sec: drowsy → fatigue 전환 시간 (초)
            hz: 추론 루프 주기
        """
        self.duration_frames = int(duration_sec * hz)
        self.drowsy_frames = int(drowsy_sec * hz)
        self.arousal_thresh = arousal_thresh

        # 연속 카운터
        self.compound_counter = 0
        self.arousal_counter = 0
        self.drowsy_counter = 0

        # 이전 상태
        self._prev_fatigue = False

        # Fatigue 대상 compound labels
        self.FATIGUE_COMPOUNDS = {"depressed", "calm"}

    def update(
        self,
        compound_label: Optional[str],
        arousal: Optional[float],
        drowsy: Optional[int],
    ) -> bool:
        """
        한 프레임의 상태를 입력받아 Fatigue 여부 반환.

        Args:
            compound_label: CompoundEmotionMapper 결과 (13-class 문자열)
            arousal: KMERFusion arousal 출력 [0, 1] (bio missing이면 None)
            drowsy: KMERFusion drowsy 출력 (0=alert, 1=drowsy, 2=sleeping)

        Returns:
            True if fatigue detected
        """
        # 조건 1: depressed/calm 연속
        if compound_label is not None and compound_label in self.FATIGUE_COMPOUNDS:
            self.compound_counter += 1
        else:
            self.compound_counter = 0

        # 조건 2: 저각성 연속
        if arousal is not None and arousal < self.arousal_thresh:
            self.arousal_counter += 1
        elif arousal is not None:
            self.arousal_counter = 0
        # arousal=None (bio missing)이면 카운터 유지 (증가하지도 리셋하지도 않음)

        # 조건 3: drowsy 연속
        if drowsy is not None and drowsy >= 1:
            self.drowsy_counter += 1
        elif drowsy is not None:
            self.drowsy_counter = 0
        # drowsy=None (cam missing)이면 카운터 유지

        # 최종 판정
        is_fatigue = (
            self.compound_counter >= self.duration_frames
            or self.arousal_counter >= self.duration_frames
            or self.drowsy_counter >= self.drowsy_frames
        )

        # 상태 전환 시 로그
        if is_fatigue and not self._prev_fatigue:
            reason = []
            if self.compound_counter >= self.duration_frames:
                reason.append(f"compound({compound_label}) {self.compound_counter}/{self.duration_frames}frames")
            if self.arousal_counter >= self.duration_frames:
                reason.append(f"arousal({arousal:.2f}<{self.arousal_thresh}) {self.arousal_counter}/{self.duration_frames}frames")
            if self.drowsy_counter >= self.drowsy_frames:
                reason.append(f"drowsy({drowsy}) {self.drowsy_counter}/{self.drowsy_frames}frames")
            logger.warning("FATIGUE detected: %s", " | ".join(reason))
        elif not is_fatigue and self._prev_fatigue:
            logger.info("FATIGUE cleared")

        self._prev_fatigue = is_fatigue
        return is_fatigue

    def get_state(self) -> dict:
        """현재 Fatigue tracker 상태 반환 (디버깅용)."""
        return {
            "compound_counter": self.compound_counter,
            "compound_threshold": self.duration_frames,
            "arousal_counter": self.arousal_counter,
            "arousal_threshold": self.duration_frames,
            "drowsy_counter": self.drowsy_counter,
            "drowsy_threshold": self.drowsy_frames,
            "is_fatigue": self._prev_fatigue,
        }

    def reset(self):
        """카운터 초기화."""
        self.compound_counter = 0
        self.arousal_counter = 0
        self.drowsy_counter = 0
        self._prev_fatigue = False


# ── Self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== FatigueTracker Test ===\n")

    # 10Hz, 5초 duration (테스트용)
    tracker = FatigueTracker(duration_sec=5, arousal_thresh=0.3, drowsy_sec=3, hz=10)

    # 시나리오: 5초간 depressed 상태 → fatigue
    print("[Test 1] Depressed 50 frames (5초)")
    for i in range(55):
        fatigue = tracker.update("depressed", arousal=0.2, drowsy=0)
        if i % 10 == 0 or fatigue:
            state = tracker.get_state()
            print(f"  frame {i:3d}: fatigue={fatigue}, "
                  f"compound={state['compound_counter']}/{state['compound_threshold']}")

    tracker.reset()
    print()

    # 시나리오: drowsy 3초 → fatigue
    print("[Test 2] Drowsy 30 frames (3초)")
    for i in range(35):
        fatigue = tracker.update("neutral", arousal=0.5, drowsy=1)
        if i % 10 == 0 or fatigue:
            state = tracker.get_state()
            print(f"  frame {i:3d}: fatigue={fatigue}, "
                  f"drowsy={state['drowsy_counter']}/{state['drowsy_threshold']}")

    tracker.reset()
    print()

    # 시나리오: bio missing (arousal=None)
    print("[Test 3] Bio missing, compound=calm 50 frames")
    for i in range(55):
        fatigue = tracker.update("calm", arousal=None, drowsy=0)
        if i % 10 == 0 or fatigue:
            state = tracker.get_state()
            print(f"  frame {i:3d}: fatigue={fatigue}, "
                  f"compound={state['compound_counter']}/{state['compound_threshold']}, "
                  f"arousal_counter={state['arousal_counter']}")
