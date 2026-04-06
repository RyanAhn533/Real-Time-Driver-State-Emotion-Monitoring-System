"""
K-MER E2E Pipeline v2 (모트렉스 13-byte 프로토콜)
==================================================
Sensors → KMERInferencer → TemporalSmoother → 10-class → PacketEncoder v2 → GatewaySender

v1 대비 변경:
  - 8-byte (0xAA..0xFE, CRC8) → 13-byte (0x02..0x03, XOR)
  - DEST/COMMAND/DATA_LENGTH 추가
  - neutral → Reserved (표시 안 함)
  - 슬픔/혐오 분리
  - 표시 메시지 + 사운드 규격 포함
  - seq/end_flag/neg_emo 삭제

Usage:
    from gateway_v2.e2e_pipeline import E2EPipeline

    pipeline = E2EPipeline()
    pipeline.initialize()

    result = pipeline.process_cycle(frame, audio, ppg, eda, temp)
    # result["packet"]           → 13-byte 모트렉스 패킷
    # result["display_message"]  → "안정적으로 주행 중입니다"
    # result["packet_hex"]       → "0201000010030000F040E84803"

    pipeline.shutdown()
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path
from collections import Counter, deque
from typing import Dict, List, Optional, Any

from .packet_encoder import (
    PacketEncoder, KFER_TO_PROTOCOL, PROTOCOL_NAMES, PROTOCOL_MESSAGES,
    STATUS_MESSAGES, KFER_LABELS, PACKET_SIZE,
    compute_xor_checksum,
)
from .gateway_sender import GatewaySender, NullGatewaySender

logger = logging.getLogger("kmer.pipeline.e2e_v2")

# ── Project paths ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_SENSING_DIR = _PROJECT_ROOT / "sensing"
_EMO_SYS_ROOT = _PROJECT_ROOT / "emotion_system"
_MM_DMS_ROOT = _PROJECT_ROOT / "multimodal_dms"


# ── Temporal Smoother (self-contained, no external dependency) ──

class MajorityVoteSmoother:
    def __init__(self, window_size: int = 7):
        self.buffer = deque(maxlen=window_size)

    def push(self, value):
        self.buffer.append(value)
        return Counter(self.buffer).most_common(1)[0][0]

    def get_confidence(self) -> float:
        if not self.buffer:
            return 0.0
        return Counter(self.buffer).most_common(1)[0][1] / len(self.buffer)

    def reset(self):
        self.buffer.clear()


class EMASmoother:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value: Optional[float] = None

    def push(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


class TemporalSmoother:
    """KMERFusion 출력 smoothing: 이산=majority vote, 연속=EMA."""

    def __init__(self, emotion_window=7, drowsy_window=5, compound_window=7, ema_alpha=0.3):
        self.emotion = MajorityVoteSmoother(emotion_window)
        self.drowsy = MajorityVoteSmoother(drowsy_window)
        self.compound = MajorityVoteSmoother(compound_window)
        self.arousal = EMASmoother(ema_alpha)
        self.valence = EMASmoother(ema_alpha)

    def smooth(self, result: Dict) -> Dict:
        out = dict(result)
        if not result.get("face_detected", False):
            return out

        if result.get("kfer_emotion") is not None:
            out["smoothed_kfer_emotion"] = self.emotion.push(result["kfer_emotion"])
            out["smoothed_emotion_confidence"] = self.emotion.get_confidence()

        if result.get("arousal") is not None:
            out["smoothed_arousal"] = self.arousal.push(float(result["arousal"]))

        if result.get("valence") is not None:
            out["smoothed_valence"] = self.valence.push(float(result["valence"]))

        if result.get("drowsy") is not None:
            out["smoothed_drowsy"] = self.drowsy.push(int(result["drowsy"]))

        if result.get("compound_id") is not None:
            out["smoothed_compound_id"] = self.compound.push(result["compound_id"])

        return out

    def reset(self):
        self.emotion.reset()
        self.drowsy.reset()
        self.compound.reset()
        self.arousal.reset()
        self.valence.reset()


# ── Fatigue Tracker (self-contained) ──

class FatigueTracker:
    """운전자 만성 피로 판정. compound ∈ {depressed, calm} 30s / arousal < 0.3 30s / drowsy 10s."""
    FATIGUE_COMPOUNDS = {"depressed", "calm"}

    def __init__(self, duration_sec=30.0, arousal_thresh=0.3, drowsy_sec=10.0, hz=10.0):
        self.duration_frames = int(duration_sec * hz)
        self.drowsy_frames = int(drowsy_sec * hz)
        self.arousal_thresh = arousal_thresh
        self.compound_counter = 0
        self.arousal_counter = 0
        self.drowsy_counter = 0

    def update(self, compound_label=None, arousal=None, drowsy=None) -> bool:
        if compound_label in self.FATIGUE_COMPOUNDS:
            self.compound_counter += 1
        else:
            self.compound_counter = 0

        if arousal is not None and arousal < self.arousal_thresh:
            self.arousal_counter += 1
        elif arousal is not None:
            self.arousal_counter = 0

        if drowsy is not None and drowsy >= 1:
            self.drowsy_counter += 1
        elif drowsy is not None:
            self.drowsy_counter = 0

        return (
            self.compound_counter >= self.duration_frames
            or self.arousal_counter >= self.duration_frames
            or self.drowsy_counter >= self.drowsy_frames
        )

    def reset(self):
        self.compound_counter = 0
        self.arousal_counter = 0
        self.drowsy_counter = 0


# ── Degraded Mode ──

class DegradedMode:
    FULL = "FULL"
    CAM_MIC = "CAM+MIC"
    CAM_BIO = "CAM+BIO"
    CAM_ONLY = "CAM_ONLY"
    NO_CAM = "NO_CAM"


def detect_mode(frame, audio, ppg, eda, temp) -> str:
    has_cam = frame is not None
    has_mic = audio is not None and len(audio) > 0
    has_bio = any(x is not None and len(x) > 0 for x in [ppg, eda, temp])
    if not has_cam:
        return DegradedMode.NO_CAM
    if has_mic and has_bio:
        return DegradedMode.FULL
    if has_mic:
        return DegradedMode.CAM_MIC
    if has_bio:
        return DegradedMode.CAM_BIO
    return DegradedMode.CAM_ONLY


KFER_LABEL_TO_ID = {label: idx for idx, label in enumerate(KFER_LABELS)}

COMPOUND_LABELS = {
    0: "neutral", 1: "calm", 2: "happy", 3: "positive_engaged",
    4: "excited", 5: "sad", 6: "depressed", 7: "anxious",
    8: "stressed", 9: "angry", 10: "hurt", 11: "surprised",
    12: "drowsy",
}


# ── E2E Pipeline v2 ──

class E2EPipeline:
    """
    K-MER E2E Pipeline v2: 모트렉스 13-byte 프로토콜.

    Sensors → KMERInferencer → TemporalSmoother → 10-class → PacketEncoder v2 → Gateway
    """

    def __init__(
        self,
        gateway_port: str = "/dev/ttyUSB0",
        gateway_baudrate: int = 115200,
        gateway_enabled: bool = True,
        emotion_window: int = 7,
        drowsy_window: int = 5,
        ema_alpha: float = 0.3,
        fatigue_duration_sec: float = 30.0,
        fatigue_arousal_thresh: float = 0.3,
        fatigue_drowsy_sec: float = 10.0,
        inference_hz: float = 10.0,
        arousal_stress_thresh: float = 0.6,
        perclos_low_attn: float = 0.2,
        perclos_drowsy: float = 0.4,
        kfer_ckpt: Optional[str] = None,
        kmer_ckpt: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = device
        self.kfer_ckpt = kfer_ckpt or str(_EMO_SYS_ROOT / "result" / "best.pth")
        self.kmer_ckpt = kmer_ckpt
        self.arousal_stress_thresh = arousal_stress_thresh
        self.perclos_low_attn = perclos_low_attn
        self.perclos_drowsy = perclos_drowsy

        # Components
        self.encoder = PacketEncoder()
        self.smoother = TemporalSmoother(emotion_window, drowsy_window, ema_alpha=ema_alpha)
        self.fatigue_tracker = FatigueTracker(fatigue_duration_sec, fatigue_arousal_thresh, fatigue_drowsy_sec, inference_hz)

        if gateway_enabled:
            self.sender = GatewaySender(port=gateway_port, baudrate=gateway_baudrate)
        else:
            self.sender = NullGatewaySender()

        self._inferencer = None
        self._initialized = False
        self._cycle_count = 0
        self._mode = DegradedMode.NO_CAM

    def initialize(self):
        """모델 로드 + 게이트웨이 연결."""
        if self._initialized:
            return

        # sys.path setup for imports
        for p in [str(_SENSING_DIR), str(_EMO_SYS_ROOT), str(_MM_DMS_ROOT)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        # KMERInferencer 로드
        logger.info("Loading KMERInferencer...")
        try:
            from kmer_inferencer_v2 import KMERInferencer
            self._inferencer = KMERInferencer(
                kfer_ckpt=self.kfer_ckpt,
                kmer_ckpt=self.kmer_ckpt,
                device=self.device,
            )
            logger.info("KMERInferencer loaded")
        except Exception as e:
            logger.error("KMERInferencer failed: %s", e)
            raise

        # Gateway 연결
        self.sender.connect()
        self._initialized = True
        logger.info("E2EPipeline v2 initialized (13-byte Motrex protocol)")

    def process_cycle(
        self,
        frame: Optional[np.ndarray],
        audio: Optional[np.ndarray] = None,
        ppg: Optional[List] = None,
        eda: Optional[List] = None,
        temp: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        단일 추론 사이클 → 13-byte 모트렉스 패킷 생성 + 전송.

        Returns:
            dict with all results including packet, display_message, etc.
        """
        if not self._initialized:
            self.initialize()

        self._cycle_count += 1
        self._mode = detect_mode(frame, audio, ppg, eda, temp)

        if self._mode == DegradedMode.NO_CAM:
            return self._fallback_result()

        # 1. KMERInferencer forward
        raw = self._inferencer.forward(
            frame_bgr=frame,
            audio_1d=audio if audio is not None else np.zeros(0, dtype=np.float32),
            ppg=ppg or [],
            eda=eda or [],
            temp=temp or [],
        )

        if not raw.get("face_detected", False):
            return self._fallback_result()

        # 2. Temporal smoothing
        smoothed = self.smoother.smooth(raw)

        # 3. 10-class 판정
        ten_class = self._compute_ten_class(smoothed)

        # 4. Fatigue
        ten_class["fatigue"] = self.fatigue_tracker.update(
            compound_label=smoothed.get("compound_label"),
            arousal=smoothed.get("smoothed_arousal", smoothed.get("arousal")),
            drowsy=smoothed.get("smoothed_drowsy", smoothed.get("drowsy")),
        )

        # 5. PacketEncoder v2 → 13-byte 패킷
        packet_info = self._encode_packet(ten_class, smoothed)

        # 6. Gateway 전송
        if packet_info.get("packet") is not None:
            self.sender.send(packet_info["packet"])

        # 7. 결과 합성
        result = {}
        result.update(raw)
        result.update(smoothed)
        result.update(ten_class)
        result.update(packet_info)
        result["mode"] = self._mode
        result["cycle"] = self._cycle_count
        return result

    def _compute_ten_class(self, smoothed: Dict) -> Dict:
        """10-class 판정: Emotion 6종 + Status 4종."""
        kfer_emotion = smoothed.get("smoothed_kfer_emotion", smoothed.get("kfer_emotion", "neutral"))
        kfer_id = KFER_LABEL_TO_ID.get(kfer_emotion, 4)
        emo_code = KFER_TO_PROTOCOL.get(kfer_id, 0x0F)

        arousal = smoothed.get("smoothed_arousal", smoothed.get("arousal"))

        # Stress
        is_negative = kfer_id in {0, 1}  # angry, anxious
        stress = is_negative and (arousal is None or arousal > self.arousal_stress_thresh)

        # PERCLOS (from face output if available)
        perclos = 0.0
        face_out = smoothed.get("_face_out", {})
        if "perclos_ear" in face_out:
            perclos = face_out["perclos_ear"][0]

        # Low Attention / Drowsy
        low_attn = self.perclos_low_attn <= perclos < self.perclos_drowsy
        drowsy_raw = smoothed.get("smoothed_drowsy", smoothed.get("drowsy", 0))
        drowsy = (drowsy_raw >= 1) or (perclos >= self.perclos_drowsy)

        return {
            "emotion_code": emo_code,
            "emotion_name_ko": PROTOCOL_NAMES.get(emo_code, ""),
            "emotion_kfer_id": kfer_id,
            "stress": stress,
            "low_attention": low_attn,
            "drowsy": drowsy,
            "drowsy_level": drowsy_raw,
            "perclos": perclos,
            "fatigue": False,
        }

    def _encode_packet(self, ten_class: Dict, smoothed: Dict) -> Dict:
        """PacketEncoder v2 → 13-byte 모트렉스 패킷."""
        try:
            kfer_id = ten_class.get("emotion_kfer_id", 4)
            confidence = smoothed.get("smoothed_emotion_confidence",
                                      smoothed.get("kfer_confidence", 0.5))
            arousal = smoothed.get("smoothed_arousal", smoothed.get("arousal"))
            perclos = ten_class.get("perclos", 0.0)

            packet = self.encoder.encode(
                kfer_emotion_id=kfer_id,
                emotion_confidence=float(confidence),
                arousal=arousal,
                perclos=perclos,
            )

            # Fatigue bit → BODY[1] D0
            if ten_class.get("fatigue", False):
                packet = self._add_fatigue_bit(packet)

            decoded = self.encoder.decode(packet)

            return {
                "packet": packet,
                "packet_hex": decoded["raw_hex"],
                "packet_bytes": decoded["raw_bytes"],
                "display_message": decoded["display_message"],
                "packet_decoded": decoded,
            }
        except Exception as e:
            logger.warning("Packet encode error: %s", e)
            return {"packet": None, "packet_hex": None, "display_message": "", "packet_bytes": None}

    def _add_fatigue_bit(self, packet: bytes) -> bytes:
        """BODY[1] (Byte10) D0에 fatigue flag, XOR checksum 재계산."""
        pkt = bytearray(packet)
        pkt[10] = pkt[10] | 0x01  # BODY[1] D0 = fatigue
        pkt[11] = compute_xor_checksum(bytes(pkt[:11]))  # recalc checksum
        return bytes(pkt)

    def _fallback_result(self) -> Dict:
        return {
            "arousal": None, "valence": None, "drowsy": 0,
            "compound_id": 0, "compound_label": "neutral",
            "kfer_emotion": "neutral", "kfer_confidence": 0.0,
            "face_detected": False,
            "emotion_code": 0x0F, "emotion_name_ko": "",
            "emotion_kfer_id": 4,
            "stress": False, "low_attention": False, "drowsy": False,
            "drowsy_level": 0, "fatigue": False, "perclos": 0.0,
            "packet": None, "packet_hex": None, "packet_bytes": None,
            "display_message": "",
            "mode": self._mode, "cycle": self._cycle_count,
        }

    def shutdown(self):
        self.sender.close()
        logger.info("E2EPipeline v2 shutdown (cycles=%d)", self._cycle_count)

    def reset(self):
        self.smoother.reset()
        self.fatigue_tracker.reset()
        self._cycle_count = 0
