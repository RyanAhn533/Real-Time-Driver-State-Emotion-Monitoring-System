"""
차량 게이트웨이 USB 패킷 인코더
================================
K-MER 시스템 출력 → 8-byte USB 패킷으로 변환.

Packet Structure (8 bytes):
  Byte0: SOF = 0xAA (Start Of Frame)
  Byte1: TYPE = 1 (감정/상태 패킷)
  Byte2: SEQ = 0~255 (순번)
  Byte3: LEN = 2 (Payload: Byte4+Byte5)
  Byte4: [Emotion Code (4bit)] [Stress (1)] [LowAttn (1)] [Drowsy (1)] [END_FLAG (1)]
  Byte5: [Emotion Intensity (3bit)] [State Intensity (3bit)] [Reserved (2bit)]
  Byte6: CRC8 (Byte1~Byte5)
  Byte7: EOF = 0xFE (End Of Frame)
"""

import struct
from typing import Dict, Optional, Tuple

# ── Constants ──
SOF = 0xAA
EOF = 0xFE
TYPE_EMOTION = 1
PAYLOAD_LEN = 2

# ── K-FER 7-class → Protocol Emotion Code ──
# K-FER labels (alphabetical): angry(0), anxious(1), happy(2), hurt(3), neutral(4), sad(5), surprised(6)
KFER_LABELS = ["angry", "anxious", "happy", "hurt", "neutral", "sad", "surprised"]

KFER_TO_PROTOCOL = {
    0: 2,   # angry    → Code 2 (분노)
    1: 0,   # anxious  → Code 0 (공포)
    2: 4,   # happy    → Code 4 (행복)
    3: 5,   # hurt     → Code 5 (혐오)
    4: 6,   # neutral  → Code 6 (중립, Reserved)
    5: 3,   # sad      → Code 3 (슬픔)
    6: 1,   # surprised→ Code 1 (놀람)
}

# Protocol Code → Korean name
PROTOCOL_NAMES = {
    0: "공포",
    1: "놀람",
    2: "분노",
    3: "슬픔",
    4: "행복",
    5: "혐오",
    6: "중립",
}

# Reverse mapping: Protocol Code → K-FER index
PROTOCOL_TO_KFER = {v: k for k, v in KFER_TO_PROTOCOL.items()}


# ── CRC8 (polynomial 0x07, init 0x00) ──
def compute_crc8(data: bytes, poly: int = 0x07, init: int = 0x00) -> int:
    """CRC8 계산 (Byte1~Byte5 기반)."""
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


# ── Intensity Quantization ──
def quantize_intensity(value: float, levels: int = 8) -> int:
    """
    [0, 1] float → 0~7 정수 (3bit).
    value=0 → 0, value=1 → 7.
    """
    v = max(0.0, min(1.0, value))
    return min(int(v * levels), levels - 1)


# ── Status Flag Detectors ──
class StressDetector:
    """
    Stress 판정: 고각성 + 부정감정 → Stress flag.

    조건: arousal > threshold AND emotion ∈ {angry, anxious, stressed}
    """

    NEGATIVE_EMOTIONS = {0, 1}  # K-FER: angry(0), anxious(1)

    def __init__(self, arousal_threshold: float = 0.6):
        self.arousal_threshold = arousal_threshold

    def detect(self, kfer_emotion_id: int, arousal: Optional[float] = None) -> bool:
        if kfer_emotion_id in self.NEGATIVE_EMOTIONS:
            if arousal is None or arousal > self.arousal_threshold:
                return True
        return False


class AttentionDetector:
    """
    Low Attention 판정: PERCLOS 중간 단계 또는 졸음 전 단계.

    PERCLOS ∈ [0.2, 0.4) → 주의 분산 상태.
    """

    def __init__(self, perclos_low: float = 0.2, perclos_high: float = 0.4):
        self.perclos_low = perclos_low
        self.perclos_high = perclos_high

    def detect(self, perclos: float) -> bool:
        return self.perclos_low <= perclos < self.perclos_high


class DrowsyDetector:
    """
    Drowsy 판정: PERCLOS ≥ threshold.

    NHTSA/FHWA 표준 (Dinges & Grace, 1998).
    """

    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold

    def detect(self, perclos: float) -> bool:
        return perclos >= self.threshold


# ── Packet Encoder ──
class PacketEncoder:
    """
    K-MER 시스템 출력 → 8-byte USB 패킷 인코더.

    Usage:
        encoder = PacketEncoder()

        # K-FER 결과로 패킷 생성
        packet = encoder.encode(
            kfer_emotion_id=2,       # happy
            emotion_confidence=0.92, # softmax max prob
            arousal=0.7,             # arousal level [0,1]
            perclos=0.1,             # PERCLOS
        )

        # 디코딩 (디버그)
        info = encoder.decode(packet)
    """

    def __init__(self):
        self.seq = 0
        self.stress_det = StressDetector()
        self.attn_det = AttentionDetector()
        self.drowsy_det = DrowsyDetector()

    def encode(
        self,
        kfer_emotion_id: int,
        emotion_confidence: float = 0.5,
        arousal: Optional[float] = None,
        perclos: float = 0.0,
        end_flag: bool = False,
    ) -> bytes:
        """
        K-MER 시스템 출력 → 8-byte packet.

        Args:
            kfer_emotion_id: K-FER top-1 class index (0~6)
            emotion_confidence: K-FER softmax max probability [0,1]
            arousal: predicted arousal [0,1] or None
            perclos: PERCLOS value [0,1]
            end_flag: True if event ended

        Returns:
            8-byte packet as bytes
        """
        # Emotion Code (upper 4 bits of Byte4)
        emo_code = KFER_TO_PROTOCOL.get(kfer_emotion_id, 6)  # default: neutral

        # Status flags
        stress = self.stress_det.detect(kfer_emotion_id, arousal)
        low_attn = self.attn_det.detect(perclos)
        drowsy = self.drowsy_det.detect(perclos)

        # Byte4: [emo_code(4)] [stress(1)] [low_attn(1)] [drowsy(1)] [end_flag(1)]
        byte4 = (
            ((emo_code & 0x0F) << 4) |
            (int(stress) << 3) |
            (int(low_attn) << 2) |
            (int(drowsy) << 1) |
            int(end_flag)
        )

        # Byte5: [emo_intensity(3)] [state_intensity(3)] [reserved(2)]
        emo_intensity = quantize_intensity(emotion_confidence)
        state_intensity = quantize_intensity(arousal if arousal is not None else 0.5)
        byte5 = (
            ((emo_intensity & 0x07) << 5) |
            ((state_intensity & 0x07) << 2)
        )

        # CRC8 over Byte1~Byte5
        payload = bytes([TYPE_EMOTION, self.seq & 0xFF, PAYLOAD_LEN, byte4, byte5])
        crc = compute_crc8(payload)

        # Assemble 8-byte packet
        packet = bytes([SOF]) + payload + bytes([crc, EOF])

        # Increment sequence
        self.seq = (self.seq + 1) & 0xFF

        return packet

    def decode(self, packet: bytes) -> Dict:
        """패킷 디코딩 (디버그/검증용)."""
        if len(packet) != 8:
            raise ValueError(f"Packet must be 8 bytes, got {len(packet)}")
        if packet[0] != SOF or packet[7] != EOF:
            raise ValueError(f"Invalid SOF/EOF: 0x{packet[0]:02X}/0x{packet[7]:02X}")

        ptype = packet[1]
        seq = packet[2]
        plen = packet[3]
        byte4 = packet[4]
        byte5 = packet[5]
        crc_recv = packet[6]

        # Verify CRC
        crc_calc = compute_crc8(packet[1:6])
        crc_ok = crc_recv == crc_calc

        # Decode Byte4
        emo_code = (byte4 >> 4) & 0x0F
        stress = bool((byte4 >> 3) & 1)
        low_attn = bool((byte4 >> 2) & 1)
        drowsy = bool((byte4 >> 1) & 1)
        end_flag = bool(byte4 & 1)

        # Decode Byte5
        emo_intensity = (byte5 >> 5) & 0x07
        state_intensity = (byte5 >> 2) & 0x07

        # Map back to K-FER
        kfer_id = PROTOCOL_TO_KFER.get(emo_code, -1)
        kfer_label = KFER_LABELS[kfer_id] if 0 <= kfer_id < len(KFER_LABELS) else "unknown"

        return {
            "type": ptype,
            "seq": seq,
            "len": plen,
            "emotion_code": emo_code,
            "emotion_name": PROTOCOL_NAMES.get(emo_code, f"reserved_{emo_code}"),
            "emotion_kfer": kfer_label,
            "stress": stress,
            "low_attention": low_attn,
            "drowsy": drowsy,
            "end_flag": end_flag,
            "emotion_intensity": emo_intensity,
            "state_intensity": state_intensity,
            "crc_ok": crc_ok,
            "raw_hex": packet.hex().upper(),
        }

    def encode_raw(
        self,
        emotion_code: int,
        stress: bool = False,
        low_attn: bool = False,
        drowsy: bool = False,
        end_flag: bool = False,
        emo_intensity: int = 0,
        state_intensity: int = 0,
    ) -> bytes:
        """
        프로토콜 코드 직접 지정하여 패킷 생성 (테스트용).
        """
        byte4 = (
            ((emotion_code & 0x0F) << 4) |
            (int(stress) << 3) |
            (int(low_attn) << 2) |
            (int(drowsy) << 1) |
            int(end_flag)
        )
        byte5 = (
            ((emo_intensity & 0x07) << 5) |
            ((state_intensity & 0x07) << 2)
        )
        payload = bytes([TYPE_EMOTION, self.seq & 0xFF, PAYLOAD_LEN, byte4, byte5])
        crc = compute_crc8(payload)
        packet = bytes([SOF]) + payload + bytes([crc, EOF])
        self.seq = (self.seq + 1) & 0xFF
        return packet


# ── Self-test ──
if __name__ == "__main__":
    enc = PacketEncoder()

    print("=== K-FER → Protocol Emotion Code Mapping ===")
    for kfer_id, label in enumerate(KFER_LABELS):
        proto_code = KFER_TO_PROTOCOL[kfer_id]
        proto_name = PROTOCOL_NAMES[proto_code]
        print(f"  K-FER {kfer_id} ({label:>10s}) → Code {proto_code} ({proto_name})")

    print("\n=== Packet Encode/Decode Test ===")
    test_cases = [
        {"kfer_emotion_id": 2, "emotion_confidence": 0.95, "arousal": 0.3, "perclos": 0.05},
        {"kfer_emotion_id": 0, "emotion_confidence": 0.85, "arousal": 0.8, "perclos": 0.1},
        {"kfer_emotion_id": 4, "emotion_confidence": 0.60, "arousal": 0.5, "perclos": 0.35},
        {"kfer_emotion_id": 4, "emotion_confidence": 0.40, "arousal": 0.2, "perclos": 0.55},
    ]

    for tc in test_cases:
        pkt = enc.encode(**tc)
        info = enc.decode(pkt)
        print(f"\n  Input: kfer={tc['kfer_emotion_id']}({KFER_LABELS[tc['kfer_emotion_id']]}), "
              f"conf={tc['emotion_confidence']}, A={tc['arousal']}, PERCLOS={tc['perclos']}")
        print(f"  Packet: {info['raw_hex']}")
        print(f"  Decoded: {info['emotion_name']}(code={info['emotion_code']}), "
              f"stress={info['stress']}, attn={info['low_attention']}, "
              f"drowsy={info['drowsy']}, emo_int={info['emotion_intensity']}, "
              f"state_int={info['state_intensity']}, CRC={'OK' if info['crc_ok'] else 'FAIL'}")
