"""
K-MER 실증 데모 파이프라인 v2 (모트렉스 13-byte 프로토콜)
=========================================================
센서 입력 → AI 모델 → 패킷 인코딩 → 게이트웨이 전송.

변경사항 (v1 → v2):
  - 8-byte (0xAA..0xFE, CRC8) → 13-byte (0x02..0x03, XOR checksum)
  - 모트렉스 규격 DEST/COMMAND/DATA_LENGTH 추가
  - 표시 메시지 + 사운드 규격 포함

Usage:
    pipeline = DemoPipeline(checkpoint_path="emotion_system/result/best.pth")

    result = pipeline.process_frame(frame_bgr)
    if result:
        packet = result["packet"]        # 13-byte 모트렉스 패킷
        emotion = result["emotion_ko"]   # "행복/기쁨", "분노" 등
        message = result["display_message"]  # "안정적으로 주행 중입니다"
        drowsy = result["drowsy"]        # True/False
"""

import sys
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

# ── Project paths ──
PROJECT_ROOT = Path(__file__).parent.parent
EMO_SYS = PROJECT_ROOT / "emotion_system"

sys.path.insert(0, str(PROJECT_ROOT / "multimodal_dms"))
sys.path.insert(0, str(EMO_SYS))

from .packet_encoder import (
    PacketEncoder, KFER_TO_PROTOCOL, PROTOCOL_NAMES, PROTOCOL_MESSAGES,
    STATUS_MESSAGES, KFER_LABELS
)


class TemporalSmoother:
    """Sliding window majority vote for temporal smoothing."""

    def __init__(self, window_size: int = 7, n_classes: int = 7):
        self.window_size = window_size
        self.n_classes = n_classes
        self.buffer: List[int] = []

    def push(self, pred_id: int) -> int:
        self.buffer.append(pred_id)
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]
        counts = Counter(self.buffer)
        return counts.most_common(1)[0][0]

    def get_confidence(self) -> float:
        if not self.buffer:
            return 0.0
        counts = Counter(self.buffer)
        return counts.most_common(1)[0][1] / len(self.buffer)

    def reset(self):
        self.buffer.clear()


class DemoPipeline:
    """
    K-MER 실증 데모 E2E 파이프라인 (모트렉스 13-byte).

    frame_bgr → K-FER → temporal smoothing → status flags → 13-byte packet → gateway
                 ↓
               EAR/PERCLOS → drowsiness
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        window_size: int = 7,
        device: str = "cuda",
    ):
        self.window_size = window_size
        self.device = device

        self.encoder = PacketEncoder()
        self.smoother = TemporalSmoother(window_size=window_size)

        self.ear_buffer: List[float] = []
        self.ear_window = 30

        self._fer = None
        self._checkpoint_path = checkpoint_path or str(EMO_SYS / "result" / "best.pth")
        self._bio_processor = None
        self._audio_processor = None

    def _get_fer(self):
        if self._fer is None:
            try:
                sys.path.insert(0, str(PROJECT_ROOT / "sensing"))
                from fer_inferencer import FERInferencer
                self._fer = FERInferencer(self._checkpoint_path, device=self.device)
            except ImportError:
                raise RuntimeError(
                    "FERInferencer not found. "
                    "Ensure sensing/fer_inferencer.py is accessible."
                )
        return self._fer

    def _compute_perclos(self) -> float:
        if len(self.ear_buffer) < self.ear_window // 2:
            return 0.0
        window = self.ear_buffer[-self.ear_window:]
        closed = sum(1 for ear in window if ear < 0.21)
        return closed / len(window)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        bio_features: Optional[Dict] = None,
        audio_features: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """단일 프레임 처리 → 13-byte 모트렉스 패킷 생성."""
        fer = self._get_fer()
        result = fer.predict(frame_bgr)
        if result is None:
            return None

        kfer_id = result["emotion_id"]
        confidence = result["confidence"]

        if "ear" in result:
            ear_left, ear_right = result["ear"]
            avg_ear = (ear_left + ear_right) / 2
            self.ear_buffer.append(avg_ear)
            if len(self.ear_buffer) > self.ear_window * 3:
                self.ear_buffer = self.ear_buffer[-self.ear_window * 2:]

        perclos = self._compute_perclos()

        arousal = None
        if bio_features and "arousal" in bio_features:
            arousal = bio_features["arousal"]

        smoothed_id = self.smoother.push(kfer_id)
        smooth_conf = self.smoother.get_confidence()

        packet = self.encoder.encode(
            kfer_emotion_id=smoothed_id,
            emotion_confidence=smooth_conf,
            arousal=arousal,
            perclos=perclos,
        )

        decoded = self.encoder.decode(packet)
        proto_code = KFER_TO_PROTOCOL.get(smoothed_id, 0x0F)

        return {
            # Raw K-FER
            "kfer_emotion": KFER_LABELS[kfer_id],
            "kfer_confidence": confidence,
            # Smoothed
            "emotion_ko": PROTOCOL_NAMES.get(proto_code, ""),
            "emotion_code": proto_code,
            "smooth_confidence": smooth_conf,
            # Status flags
            "stress": decoded["stress"],
            "low_attention": decoded["low_attention"],
            "drowsy": decoded["drowsy"],
            "perclos": perclos,
            # Packet (13-byte 모트렉스)
            "packet": packet,
            "packet_hex": decoded["raw_hex"],
            "packet_bytes": decoded["raw_bytes"],
            # Intensity
            "emotion_intensity": decoded["emotion_intensity"],
            "state_intensity": decoded["state_intensity"],
            # 표시 메시지 (PPTX 규격)
            "display_message": decoded["display_message"],
        }

    def reset(self):
        self.smoother.reset()
        self.ear_buffer.clear()

    def register_bio_processor(self, processor):
        self._bio_processor = processor

    def register_audio_processor(self, processor):
        self._audio_processor = processor


# ── Self-test (without actual model) ──
if __name__ == "__main__":
    from packet_encoder import PacketEncoder, KFER_LABELS, PROTOCOL_NAMES, KFER_TO_PROTOCOL

    print("=" * 70)
    print("  DemoPipeline v2 — 모트렉스 13-byte 프로토콜 테스트")
    print("=" * 70)

    # Temporal smoother test
    smoother = TemporalSmoother(window_size=5)
    preds = [2, 2, 2, 0, 2, 2, 1, 2, 2, 2]
    print("\nTemporal Smoother (window=5):")
    for p in preds:
        smoothed = smoother.push(p)
        conf = smoother.get_confidence()
        print(f"  input={KFER_LABELS[p]:>10s} → smoothed={KFER_LABELS[smoothed]:>10s} "
              f"(conf={conf:.2f})")

    # Packet encoding flow
    enc = PacketEncoder()
    scenarios = [
        ("정상 운전 (행복)", 2, 0.9, 0.4, 0.05),
        ("분노 + 스트레스",  0, 0.85, 0.8, 0.1),
        ("졸음 전조",       4, 0.6, 0.3, 0.3),
        ("졸음 운전",       4, 0.5, 0.2, 0.55),
    ]

    print("\nGateway Packet scenarios (13-byte 모트렉스):")
    for name, emo, conf, arousal, perclos in scenarios:
        pkt = enc.encode(emo, conf, arousal, perclos)
        info = enc.decode(pkt)
        print(f"\n  [{name}]")
        print(f"    → {info['emotion_name']}(code={info['emotion_code']:X}), "
              f"stress={info['stress']}, attn={info['low_attention']}, drowsy={info['drowsy']}")
        print(f"    Packet: {info['raw_bytes']}")
        print(f"    표시: \"{info['display_message']}\"")

    print("\n" + "=" * 70)
