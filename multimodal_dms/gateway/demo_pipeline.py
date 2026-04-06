"""
K-MER 실증 데모 파이프라인
==========================
센서 입력 → AI 모델 → 패킷 인코딩 → 게이트웨이 전송.

현재 구현:
  - K-FER (얼굴 감정 7-class) ← sensing/fer_inferencer.py
  - PERCLOS/EAR (졸음 감지)   ← emotion_system/integration/drowsiness_judge.py
  - Temporal Smoothing (majority vote)
  - Packet Encoding → 8-byte USB

향후 추가 (센싱 코드 올리면 연결):
  - E4 Wristband (BVP, EDA, TEMP, HR) → Bio expert
  - Microphone (16kHz) → Audio expert (emotion2vec)
  - K-MER Fusion (멀티모달 퓨전)

Usage:
    pipeline = DemoPipeline(checkpoint_path="emotion_system/result/best.pth")

    # 프레임 단위 처리
    result = pipeline.process_frame(frame_bgr)
    if result:
        packet = result["packet"]        # 8-byte USB 패킷
        emotion = result["emotion_ko"]   # "행복", "분노" 등
        drowsy = result["drowsy"]        # True/False
"""

import sys
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

# ── Project paths ──
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Jetson_thor/
EMO_SYS = PROJECT_ROOT / "emotion_system"

# Add paths for imports
sys.path.insert(0, str(PROJECT_ROOT / "multimodal_dms"))
sys.path.insert(0, str(EMO_SYS))

from gateway.packet_encoder import (
    PacketEncoder, KFER_TO_PROTOCOL, PROTOCOL_NAMES, KFER_LABELS
)


class TemporalSmoother:
    """
    Sliding window majority vote for temporal smoothing.

    AU/FACS 기반 K-FER은 프레임 간 예측이 안정적이므로
    majority vote로 노이즈 제거 효과가 높음.
    """

    def __init__(self, window_size: int = 7, n_classes: int = 7):
        self.window_size = window_size
        self.n_classes = n_classes
        self.buffer: List[int] = []

    def push(self, pred_id: int) -> int:
        """
        예측 추가 후 majority vote 결과 반환.

        Args:
            pred_id: K-FER class index (0~6)

        Returns:
            smoothed prediction (majority vote)
        """
        self.buffer.append(pred_id)
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]

        counts = Counter(self.buffer)
        return counts.most_common(1)[0][0]

    def get_confidence(self) -> float:
        """현재 윈도우에서 majority class의 비율."""
        if not self.buffer:
            return 0.0
        counts = Counter(self.buffer)
        majority_count = counts.most_common(1)[0][1]
        return majority_count / len(self.buffer)

    def reset(self):
        self.buffer.clear()


class DemoPipeline:
    """
    K-MER 실증 데모 end-to-end 파이프라인.

    구조:
        frame_bgr → K-FER → temporal smoothing → status flags → packet → gateway
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

        # Packet encoder
        self.encoder = PacketEncoder()

        # Temporal smoother
        self.smoother = TemporalSmoother(window_size=window_size)

        # PERCLOS buffer (EAR values)
        self.ear_buffer: List[float] = []
        self.ear_window = 30  # 1 second at 30fps

        # K-FER inferencer (lazy init — Jetson에서만 실제 로드)
        self._fer = None
        self._checkpoint_path = checkpoint_path or str(EMO_SYS / "result" / "best.pth")

        # Bio/Audio (placeholder — 센싱 코드 올리면 연결)
        self._bio_processor = None
        self._audio_processor = None

    def _get_fer(self):
        """Lazy-load FER inferencer."""
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
        """현재 EAR buffer에서 PERCLOS 계산."""
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
        """
        단일 프레임 처리 → 패킷 생성.

        Args:
            frame_bgr: BGR image from camera
            bio_features: (future) E4 Wristband features
            audio_features: (future) audio emotion features

        Returns:
            dict with packet, emotion info, status flags
            or None if no face detected
        """
        # 1. K-FER inference
        fer = self._get_fer()
        result = fer.predict(frame_bgr)
        if result is None:
            return None

        kfer_id = result["emotion_id"]
        confidence = result["confidence"]

        # 2. EAR → PERCLOS (졸음 감지)
        # FERInferencer에서 EAR도 추출 가능
        if "ear" in result:
            ear_left, ear_right = result["ear"]
            avg_ear = (ear_left + ear_right) / 2
            self.ear_buffer.append(avg_ear)
            if len(self.ear_buffer) > self.ear_window * 3:
                self.ear_buffer = self.ear_buffer[-self.ear_window * 2:]

        perclos = self._compute_perclos()

        # 3. Arousal (현재: None, 향후 멀티모달 퓨전 결과 사용)
        arousal = None
        if bio_features and "arousal" in bio_features:
            arousal = bio_features["arousal"]

        # 4. Temporal smoothing
        smoothed_id = self.smoother.push(kfer_id)
        smooth_conf = self.smoother.get_confidence()

        # 5. Packet encoding (smoothed prediction 사용)
        packet = self.encoder.encode(
            kfer_emotion_id=smoothed_id,
            emotion_confidence=smooth_conf,
            arousal=arousal,
            perclos=perclos,
        )

        # 6. Decode for debug info
        decoded = self.encoder.decode(packet)

        proto_code = KFER_TO_PROTOCOL.get(smoothed_id, 6)

        return {
            # Raw K-FER result
            "kfer_emotion": KFER_LABELS[kfer_id],
            "kfer_confidence": confidence,
            # Smoothed result
            "emotion_ko": PROTOCOL_NAMES.get(proto_code, "중립"),
            "emotion_code": proto_code,
            "smooth_confidence": smooth_conf,
            # Status flags
            "stress": decoded["stress"],
            "low_attention": decoded["low_attention"],
            "drowsy": decoded["drowsy"],
            "perclos": perclos,
            # Packet
            "packet": packet,
            "packet_hex": decoded["raw_hex"],
            # Intensity
            "emotion_intensity": decoded["emotion_intensity"],
            "state_intensity": decoded["state_intensity"],
        }

    def process_end_event(self) -> bytes:
        """이벤트 종료 패킷 전송."""
        return self.encoder.encode(
            kfer_emotion_id=4,  # neutral
            end_flag=True,
        )

    def reset(self):
        """파이프라인 상태 초기화."""
        self.smoother.reset()
        self.ear_buffer.clear()
        self.encoder.seq = 0

    # ── Future: Bio/Audio hooks ──

    def register_bio_processor(self, processor):
        """
        E4 Wristband 센서 프로세서 등록.
        processor.process(bvp, eda, temp, hr) → {"arousal": float, "features": ndarray}
        """
        self._bio_processor = processor

    def register_audio_processor(self, processor):
        """
        마이크 오디오 프로세서 등록.
        processor.process(audio_chunk) → {"emotion_probs": ndarray, "features": ndarray}
        """
        self._audio_processor = processor


# ── Self-test (without actual model) ──
if __name__ == "__main__":
    print("=== DemoPipeline Structure Test ===")
    print()

    # Test temporal smoother
    smoother = TemporalSmoother(window_size=5)
    preds = [2, 2, 2, 0, 2, 2, 1, 2, 2, 2]  # mostly happy
    print("Temporal Smoother test (window=5):")
    for p in preds:
        smoothed = smoother.push(p)
        conf = smoother.get_confidence()
        print(f"  input={KFER_LABELS[p]:>10s} → smoothed={KFER_LABELS[smoothed]:>10s} "
              f"(conf={conf:.2f})")

    print()

    # Test packet encoding flow
    enc = PacketEncoder()
    scenarios = [
        ("정상 운전", 2, 0.9, 0.4, 0.05),    # happy, low arousal, alert
        ("분노 운전", 0, 0.85, 0.8, 0.1),     # angry, high arousal
        ("졸음 전조", 4, 0.6, 0.3, 0.3),      # neutral, moderate PERCLOS
        ("졸음 운전", 4, 0.5, 0.2, 0.55),     # neutral, high PERCLOS
    ]

    print("Gateway Packet scenarios:")
    for name, emo, conf, arousal, perclos in scenarios:
        pkt = enc.encode(emo, conf, arousal, perclos)
        info = enc.decode(pkt)
        print(f"  [{name}] → {info['emotion_name']}(code={info['emotion_code']}), "
              f"stress={info['stress']}, attn={info['low_attention']}, "
              f"drowsy={info['drowsy']}, pkt={info['raw_hex']}")

    print()
    print("Pipeline ready for sensor integration.")
    print("  - Register bio:  pipeline.register_bio_processor(processor)")
    print("  - Register audio: pipeline.register_audio_processor(processor)")
