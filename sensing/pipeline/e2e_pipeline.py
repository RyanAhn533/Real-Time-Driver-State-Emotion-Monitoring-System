"""
K-MER End-to-End Pipeline
==========================
센서 → KMERFusion → Temporal Smoothing → 10-class 판정 → PacketEncoder → Gateway

동작 보존 원칙:
  - KMERInferencer를 원본 그대로 import하여 사용 (수정 없음)
  - PacketEncoder를 기존 gateway 코드에서 그대로 import
  - 기존 로직 변경 없이 wrapper로 추가 기능만 붙임

Usage:
    from pipeline.e2e_pipeline import E2EPipeline
    from config import load_config

    cfg = load_config()
    pipeline = E2EPipeline(cfg)

    # 추론 사이클 (10Hz)
    result = pipeline.process_cycle(frame, audio, ppg, eda, temp)
    # result: {
    #   "arousal", "valence", "drowsy", "compound_label",     # KMERFusion 원본
    #   "smoothed_arousal", "smoothed_drowsy", ...            # Smoothed
    #   "emotion_code", "stress", "low_attention", "fatigue", # 10-class
    #   "packet", "packet_hex",                               # Gateway
    # }
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

# sensing/ 디렉토리를 sys.path에 추가 (직접 실행 지원)
# sensing/ 디렉토리를 sys.path[0]에 확보 (emotion_system/pipeline/ 충돌 방지)
_SENSING_DIR = Path(__file__).resolve().parent.parent
_sensing_str = str(_SENSING_DIR)
if _sensing_str in sys.path:
    sys.path.remove(_sensing_str)
sys.path.insert(0, _sensing_str)

from core.logger import get_logger
from pipeline.temporal_smoother import MultimodalTemporalSmoother
from pipeline.fatigue_tracker import FatigueTracker
from pipeline.gateway_sender import GatewaySender, NullGatewaySender

logger = get_logger("kmer.pipeline.e2e")
_PROJECT_ROOT = _SENSING_DIR.parent  # Jetson_thor/

# 기존 코드 경로
_RAW_SENSING_DIR = (
    _SENSING_DIR
    / "raw_sensing_code"
    / "Real-Time-Driver-State-Emotion-Monitoring-System-ysh_sensing_260305"
    / "sensing"
)
_EMO_SYS_ROOT = _PROJECT_ROOT / "emotion_system"
_MM_DMS_ROOT = _PROJECT_ROOT / "multimodal_dms"

# sys.path에 추가 (기존 코드 import 가능하도록)
for _p in [str(_RAW_SENSING_DIR), str(_EMO_SYS_ROOT), str(_MM_DMS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 핵심: sensing/ 디렉토리를 다시 sys.path[0]에 보장
# emotion_system/pipeline/ 이 sensing/pipeline/ 을 가리는 것을 방지
if sys.path[0] != _sensing_str:
    if _sensing_str in sys.path:
        sys.path.remove(_sensing_str)
    sys.path.insert(0, _sensing_str)


# ── Degraded Mode 정의 ──────────────────────────────────────────────────

class DegradedMode:
    """센서 가용 상태에 따른 운영 모드."""
    FULL = "FULL"           # Cam + Mic + Bio
    CAM_MIC = "CAM+MIC"     # Cam + Mic (Bio 없음)
    CAM_BIO = "CAM+BIO"     # Cam + Bio (Mic 없음)
    CAM_ONLY = "CAM_ONLY"   # Cam만
    NO_CAM = "NO_CAM"       # 카메라 없음 (추론 불가)


def detect_mode(
    frame: Optional[np.ndarray],
    audio: Optional[np.ndarray],
    ppg: Optional[List],
    eda: Optional[List],
    temp: Optional[List],
) -> str:
    """현재 센서 데이터 가용성으로 mode 판별."""
    has_cam = frame is not None
    has_mic = audio is not None and len(audio) > 0
    has_bio = (ppg is not None and len(ppg) > 0) or \
              (eda is not None and len(eda) > 0) or \
              (temp is not None and len(temp) > 0)

    if not has_cam:
        return DegradedMode.NO_CAM
    if has_mic and has_bio:
        return DegradedMode.FULL
    if has_mic:
        return DegradedMode.CAM_MIC
    if has_bio:
        return DegradedMode.CAM_BIO
    return DegradedMode.CAM_ONLY


# ── Compound ID → Label 매핑 (CompoundEmotionMapper 참조) ────────────────

COMPOUND_LABELS = {
    0: "neutral", 1: "calm", 2: "happy", 3: "positive_engaged",
    4: "excited", 5: "sad", 6: "depressed", 7: "anxious",
    8: "stressed", 9: "angry", 10: "hurt", 11: "surprised",
    12: "drowsy",
}


# ── E2E Pipeline ─────────────────────────────────────────────────────────

class E2EPipeline:
    """
    K-MER End-to-End Pipeline.

    구조:
        Sensors → KMERInferencer → TemporalSmoother → 10-class → PacketEncoder → Gateway
    """

    def __init__(self, config):
        """
        Args:
            config: SensingConfig dataclass (from config.load_config())
        """
        self.config = config
        self._initialized = False
        self._inferencer = None
        self._mode = DegradedMode.NO_CAM
        self._cycle_count = 0
        self._last_mode_log = 0.0

        # Temporal Smoother
        ts = config.temporal_smoothing
        self.smoother = MultimodalTemporalSmoother(
            emotion_window=ts.emotion_window,
            drowsy_window=ts.drowsy_window,
            compound_window=ts.compound_window,
            ema_alpha=ts.ema_alpha,
        )

        # Fatigue Tracker
        th = config.thresholds
        self.fatigue_tracker = FatigueTracker(
            duration_sec=th.fatigue_duration_sec,
            arousal_thresh=th.fatigue_arousal,
            drowsy_sec=th.fatigue_drowsy_sec,
            hz=config.inference.hz,
        )

        # PacketEncoder (기존 코드 그대로 import)
        self._encoder = None

        # Gateway Sender
        if config.gateway.enabled:
            self.sender = GatewaySender(
                port=config.gateway.port,
                baudrate=config.gateway.baudrate,
                timeout=config.gateway.timeout,
                retry_interval=config.gateway.retry_interval,
            )
        else:
            self.sender = NullGatewaySender()

        logger.info("E2EPipeline created (shadow_mode=%s, gateway=%s)",
                     config.shadow_mode.enabled, config.gateway.enabled)

    def _lazy_init(self):
        """Lazy initialization — 첫 호출 시 모델 로드."""
        if self._initialized:
            return

        logger.info("Initializing KMERInferencer...")
        try:
            from kmer_inferencer import KMERInferencer
            self._inferencer = KMERInferencer(
                kfer_ckpt=self.config.inference.kfer_ckpt,
                kmer_ckpt=self.config.inference.kmer_ckpt,
                device=self.config.inference.device,
                enable_audio_experts=self.config.inference.enable_audio_experts,
                enable_face_expert=self.config.inference.enable_face_expert,
                audio_src_sr=self.config.inference.audio_sr,
            )
            logger.info("KMERInferencer loaded successfully")
        except Exception as e:
            logger.error("KMERInferencer init failed: %s", e)
            raise

        # PacketEncoder
        try:
            from gateway.packet_encoder import (
                PacketEncoder, KFER_TO_PROTOCOL, PROTOCOL_NAMES, KFER_LABELS
            )
            self._encoder = PacketEncoder()
            self._kfer_to_protocol = KFER_TO_PROTOCOL
            self._protocol_names = PROTOCOL_NAMES
            self._kfer_labels = KFER_LABELS
            logger.info("PacketEncoder loaded successfully")
        except Exception as e:
            logger.error("PacketEncoder init failed: %s", e)
            self._encoder = None

        # Gateway 연결
        self.sender.connect()

        self._initialized = True

    def process_cycle(
        self,
        frame: Optional[np.ndarray],
        audio: Optional[np.ndarray],
        ppg: Optional[List] = None,
        eda: Optional[List] = None,
        temp: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        단일 추론 사이클.

        Args:
            frame: BGR image from camera (H, W, 3) or None
            audio: mono float32 audio (48kHz) or None
            ppg: [(ts, d1, d2), ...] or None
            eda: [(ts, real), ...] or None
            temp: [(ts, skin_c), ...] or None

        Returns:
            10-class 판정 결과 + 패킷 정보 dict
        """
        self._lazy_init()
        self._cycle_count += 1

        # 1. Degraded mode 판별
        self._mode = detect_mode(frame, audio, ppg, eda, temp)

        # 모드 변경 시 로그 (10초마다)
        now = time.time()
        if now - self._last_mode_log >= 10.0:
            logger.info("Operating mode: %s (cycle=%d)", self._mode, self._cycle_count)
            self._last_mode_log = now

        # 2. NO_CAM → fallback 결과
        if self._mode == DegradedMode.NO_CAM:
            return self._make_fallback_result()

        # 3. KMERInferencer.forward() 호출 (기존 로직 그대로)
        raw_result = self._inferencer.forward(
            frame_bgr=frame,
            audio_1d=audio if audio is not None else np.zeros(0, dtype=np.float32),
            ppg=ppg or [],
            eda=eda or [],
            temp=temp or [],
        )

        # face_detected=False인 경우
        if not raw_result.get("face_detected", False):
            return self._make_fallback_result()

        # 4. Temporal Smoothing
        smoothed = self.smoother.smooth(raw_result)

        # 5. 10-class 판정
        ten_class = self._compute_ten_class(smoothed)

        # 6. Fatigue 판정
        fatigue = self.fatigue_tracker.update(
            compound_label=smoothed.get("compound_label"),
            arousal=smoothed.get("smoothed_arousal", smoothed.get("arousal")),
            drowsy=smoothed.get("smoothed_drowsy", smoothed.get("drowsy")),
        )
        ten_class["fatigue"] = fatigue

        # 7. PacketEncoder → 8-byte 패킷
        packet_info = self._encode_packet(ten_class, smoothed)

        # 8. Gateway 전송
        if packet_info.get("packet") is not None:
            self.sender.send(packet_info["packet"])

        # 9. 결과 합성
        result = {}
        result.update(raw_result)       # 원본 KMERFusion 결과
        result.update(smoothed)         # Smoothed 값들
        result.update(ten_class)        # 10-class 판정
        result.update(packet_info)      # 패킷 정보
        result["mode"] = self._mode
        result["cycle"] = self._cycle_count

        return result

    def _compute_ten_class(self, smoothed: Dict) -> Dict:
        """
        10-class 판정.

        Emotion 6: smoothed K-FER → KFER_TO_PROTOCOL 매핑
        Driver State 4: Stress, Low Attention, Drowsy, Fatigue
        """
        result = {}

        # ── Emotion (6개) ──
        kfer_emotion = smoothed.get("smoothed_kfer_emotion", smoothed.get("kfer_emotion"))
        kfer_label_to_id = {label: idx for idx, label in enumerate(self._kfer_labels)}
        kfer_id = kfer_label_to_id.get(kfer_emotion, 4)  # default: neutral(4)
        emotion_code = self._kfer_to_protocol.get(kfer_id, 5)  # default: neutral(5)
        emotion_name = self._protocol_names.get(emotion_code, "중립")

        result["emotion_code"] = emotion_code
        result["emotion_name_ko"] = emotion_name
        result["emotion_kfer_id"] = kfer_id

        # ── Stress ──
        arousal = smoothed.get("smoothed_arousal", smoothed.get("arousal"))
        is_negative = kfer_id in {0, 1}  # angry, anxious
        if is_negative:
            if arousal is None or arousal > self.config.thresholds.arousal_stress:
                result["stress"] = True
            else:
                result["stress"] = False
        else:
            result["stress"] = False

        # ── Low Attention ──
        # PERCLOS는 KMERInferencer에서 facs_aux로 계산됨
        # smoothed result에 직접 perclos가 없으므로 raw에서 가져옴
        perclos = None
        raw_face_out = smoothed.get("_face_out", {})
        if "perclos_ear" in raw_face_out:
            perclos = raw_face_out["perclos_ear"][0]
        # 또는 별도로 전달된 perclos
        if perclos is None:
            perclos = 0.0  # fallback

        th = self.config.thresholds
        result["low_attention"] = th.perclos_low_attn <= perclos < th.perclos_drowsy

        # ── Drowsy ──
        drowsy = smoothed.get("smoothed_drowsy", smoothed.get("drowsy", 0))
        result["drowsy"] = drowsy >= 1 or perclos >= th.perclos_drowsy
        result["drowsy_level"] = drowsy  # 원본 3-class 보존

        # ── Fatigue ──
        # 호출자 (_process_cycle)에서 설정
        result["fatigue"] = False  # placeholder

        return result

    def _encode_packet(self, ten_class: Dict, smoothed: Dict) -> Dict:
        """PacketEncoder로 8-byte 패킷 생성."""
        if self._encoder is None:
            return {"packet": None, "packet_hex": None}

        try:
            # 기존 PacketEncoder.encode() 호출
            kfer_id = ten_class.get("emotion_kfer_id", 4)
            confidence = smoothed.get("smoothed_emotion_confidence",
                                     smoothed.get("kfer_confidence", 0.5))
            arousal = smoothed.get("smoothed_arousal", smoothed.get("arousal"))
            perclos = 0.0  # 추후 연결

            packet = self._encoder.encode(
                kfer_emotion_id=kfer_id,
                emotion_confidence=float(confidence),
                arousal=arousal,
                perclos=perclos,
            )

            # Fatigue bit 추가 (Byte5 bit0)
            if ten_class.get("fatigue", False):
                packet = self._add_fatigue_bit(packet)

            decoded = self._encoder.decode(packet)

            return {
                "packet": packet,
                "packet_hex": decoded.get("raw_hex", packet.hex().upper()),
                "packet_decoded": decoded,
            }
        except Exception as e:
            logger.warning("Packet encode error: %s", e)
            return {"packet": None, "packet_hex": None}

    def _add_fatigue_bit(self, packet: bytes) -> bytes:
        """
        기존 패킷의 Byte5 bit0에 Fatigue flag 설정.

        Byte5를 수정하면 CRC도 재계산해야 함.
        """
        try:
            from gateway.packet_encoder import compute_crc8, SOF, EOF
            pkt = bytearray(packet)

            # Byte5 bit0 = Fatigue
            pkt[5] = pkt[5] | 0x01

            # CRC 재계산 (Byte1~Byte5)
            pkt[6] = compute_crc8(bytes(pkt[1:6]))

            return bytes(pkt)
        except Exception:
            return packet  # 실패 시 원본 반환

    def _make_fallback_result(self) -> Dict:
        """카메라 없을 때 또는 얼굴 미검출 시 fallback 결과."""
        return {
            # KMERFusion 원본 (없음)
            "arousal": None,
            "valence": None,
            "drowsy": 0,
            "compound_id": 0,
            "compound_label": "neutral",
            "kfer_emotion": "neutral",
            "kfer_confidence": 0.0,
            "face_detected": False,
            # 10-class (모두 기본값)
            "emotion_code": 5,       # neutral (Protocol code 5 = 중립)
            "emotion_name_ko": "중립",
            "emotion_kfer_id": 4,
            "stress": False,
            "low_attention": False,
            "drowsy": False,
            "drowsy_level": 0,
            "fatigue": False,
            # Packet
            "packet": None,
            "packet_hex": None,
            # Meta
            "mode": self._mode,
            "cycle": self._cycle_count,
        }

    def shutdown(self):
        """파이프라인 정리."""
        self.sender.close()
        logger.info("E2EPipeline shutdown (total cycles=%d)", self._cycle_count)

    def reset(self):
        """상태 초기화."""
        self.smoother.reset()
        self.fatigue_tracker.reset()
        if self._encoder is not None:
            self._encoder.seq = 0
        self._cycle_count = 0


# ── Self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== E2EPipeline Structure Test ===")
    print()
    print("DegradedMode:")
    print(f"  FULL     = {DegradedMode.FULL}")
    print(f"  CAM_MIC  = {DegradedMode.CAM_MIC}")
    print(f"  CAM_BIO  = {DegradedMode.CAM_BIO}")
    print(f"  CAM_ONLY = {DegradedMode.CAM_ONLY}")
    print(f"  NO_CAM   = {DegradedMode.NO_CAM}")
    print()

    # Mode detection test
    test_cases = [
        ("FULL", np.zeros((480, 640, 3)), np.zeros(48000), [(0, 1.0, 2.0)], [], []),
        ("CAM+MIC", np.zeros((480, 640, 3)), np.zeros(48000), [], [], []),
        ("CAM+BIO", np.zeros((480, 640, 3)), None, [(0, 1.0, 2.0)], [], []),
        ("CAM_ONLY", np.zeros((480, 640, 3)), None, [], [], []),
        ("NO_CAM", None, None, [], [], []),
    ]

    for expected, frame, audio, ppg, eda, temp in test_cases:
        mode = detect_mode(frame, audio, ppg, eda, temp)
        status = "OK" if mode == expected else "FAIL"
        print(f"  [{status}] expected={expected:>10s}, got={mode:>10s}")

    print()
    print("Pipeline ready for integration.")
