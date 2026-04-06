"""
복합감정인지 USB 통신 프로토콜 v2 (모트렉스 규격, 2026-04-03 변경)
===================================================================
모트렉스 박정현 위원 메일 기준 패킷 인코더.

Packet Structure (12 bytes):
  Byte 0:  STX          = 0x02
  Byte 1:  VERSION      = 0x01
  Byte 2:  RESERVED_H   = 0x00
  Byte 3:  RESERVED_L   = 0x00
  Byte 4:  DEST         = 0x10 (슈퍼게이트→모트렉스) / 0x00 (모트렉스→슈퍼게이트)
  Byte 5:  DATA_LEN[0]  = 0x03 (LSB) — COMMAND(1) + BODY(2) = 3
  Byte 6:  DATA_LEN[1]  = 0x00
  Byte 7:  COMMAND      = 0xF0
  Byte 8:  BODY[0]      = Emotion Code (0x00=공포, 0x01=놀람, 0x02=분노, 0x03=슬픔, 0x04=행복, 0x05=혐오)
  Byte 9:  BODY[1]      = [Emo Intensity D7~D5] [State Intensity D4~D2] [Reserved D1~D0]
  Byte10:  CHECKSUM     = XOR(Byte0 ~ Byte9)
  Byte11:  ETX          = 0x03

변경 이력:
  2026-03-30  초기 13-byte (DATA_LEN 3바이트, BODY[0] 비트패킹)
  2026-04-03  12-byte로 변경 (DATA_LEN 2바이트, BODY[0] 감정코드 full byte)
              모트렉스 박정현 위원 메일 기준

예시 (공포, 감정7, 상태7):
  슈퍼게이트→모트렉스: 02 01 00 00 10 03 00 F0 00 FC 1C 03
  모트렉스→슈퍼게이트: 02 01 00 00 00 03 00 F0 00 FC 0C 03
"""

from typing import Dict, Optional

# ── Protocol Constants (모트렉스 규격) ──
STX = 0x02
ETX = 0x03
VERSION = 0x01
RESERVED_H = 0x00
RESERVED_L = 0x00
DEST_MOTREX = 0x10       # 슈퍼게이트 → 모트렉스 제어기
DEST_SUPERGATE = 0x00    # 모트렉스 → 슈퍼게이트 제어기
COMMAND = 0xF0
BODY_LEN = 2
DATA_LENGTH_VALUE = BODY_LEN + 1  # COMMAND(1) + BODY(2) = 3

PACKET_SIZE = 12  # STX(1)+VER(1)+RSV(2)+DEST(1)+DLEN(2)+CMD(1)+BODY(2)+CHKSUM(1)+ETX(1)

# ── K-FER 7-class → Protocol Emotion Code ──
KFER_LABELS = ["angry", "anxious", "happy", "hurt", "neutral", "sad", "surprised"]

# Protocol: 0=공포, 1=놀람, 2=분노, 3=슬픔, 4=행복/기쁨, 5=혐오
KFER_TO_PROTOCOL = {
    0: 2,     # angry    → Code 2 (분노)
    1: 0,     # anxious  → Code 0 (공포)
    2: 4,     # happy    → Code 4 (행복/기쁨)
    3: 5,     # hurt     → Code 5 (혐오)
    4: 0xFF,  # neutral  → Reserved (표시 안 함)
    5: 3,     # sad      → Code 3 (슬픔)
    6: 1,     # surprised→ Code 1 (놀람)
}

PROTOCOL_NAMES = {
    0: "공포",
    1: "놀람",
    2: "분노",
    3: "슬픔",
    4: "행복/기쁨",
    5: "혐오",
}

# 표시 메시지 (복합감정인지 표시 및 사운드__v1.pptx)
PROTOCOL_MESSAGES = {
    0: "안전하게 함께 가고 있어요",
    1: "괜찮습니다, 안정적으로 주행 중입니다",
    2: "편안한 주행을 도와드릴게요",
    3: "편안한 주행을 도와드릴게요",
    4: "안정적으로 주행 중입니다",
    5: "차량이 안정적으로 유지하고 있습니다",
}

STATUS_MESSAGES = {
    "stress": "편안한 주행을 도와드릴게요",
    "low_attention": "주행에 집중해 주세요",
    "drowsy": "잠시 휴식이 필요해 보여요",
}

PROTOCOL_TO_KFER = {0: 1, 1: 6, 2: 0, 3: 5, 4: 2, 5: 3}


# ── XOR Checksum ──
def compute_xor_checksum(data: bytes) -> int:
    result = 0
    for b in data:
        result ^= b
    return result


def quantize_intensity(value: float, levels: int = 8) -> int:
    """[0, 1] float → 0~7 정수 (3bit)."""
    v = max(0.0, min(1.0, value))
    return min(int(v * levels), levels - 1)


# ── Packet Encoder (모트렉스 12-byte 규격) ──
class PacketEncoder:
    """
    K-MER → 모트렉스 12-byte USB 패킷 인코더.

    Usage:
        encoder = PacketEncoder()
        packet = encoder.encode(kfer_emotion_id=2, emotion_confidence=0.92, arousal=0.7, perclos=0.1)
        info = encoder.decode(packet)
    """

    def __init__(self, dest: int = DEST_MOTREX):
        self.dest = dest

    def _build_header(self) -> bytes:
        """STX + VERSION + RESERVED(2) + DEST + DATA_LENGTH(2, LSB first) + COMMAND."""
        dl_bytes = DATA_LENGTH_VALUE.to_bytes(2, byteorder='little')
        return bytes([STX, VERSION, RESERVED_H, RESERVED_L, self.dest]) + dl_bytes + bytes([COMMAND])

    def encode(
        self,
        kfer_emotion_id: int,
        emotion_confidence: float = 0.5,
        arousal: Optional[float] = None,
        perclos: float = 0.0,
    ) -> bytes:
        """K-MER 출력 → 12-byte 모트렉스 패킷."""
        # BODY[0]: emotion code (full byte)
        emo_code = KFER_TO_PROTOCOL.get(kfer_emotion_id, 0xFF)
        body0 = emo_code & 0xFF

        # BODY[1]: [emo_intensity D7~D5] [state_intensity D4~D2] [reserved D1~D0]
        emo_intensity = quantize_intensity(emotion_confidence)
        state_intensity = quantize_intensity(arousal if arousal is not None else 0.5)
        body1 = (
            ((emo_intensity & 0x07) << 5) |
            ((state_intensity & 0x07) << 2)
        )

        pre_checksum = self._build_header() + bytes([body0, body1])
        checksum = compute_xor_checksum(pre_checksum)

        return pre_checksum + bytes([checksum, ETX])

    def encode_raw(
        self,
        emotion_code: int,
        emo_intensity: int = 0,
        state_intensity: int = 0,
    ) -> bytes:
        """프로토콜 코드 직접 지정 (테스트용)."""
        body0 = emotion_code & 0xFF
        body1 = (
            ((emo_intensity & 0x07) << 5) |
            ((state_intensity & 0x07) << 2)
        )
        pre_checksum = self._build_header() + bytes([body0, body1])
        checksum = compute_xor_checksum(pre_checksum)
        return pre_checksum + bytes([checksum, ETX])

    def decode(self, packet: bytes) -> Dict:
        """패킷 디코딩."""
        if len(packet) != PACKET_SIZE:
            raise ValueError(f"Packet must be {PACKET_SIZE} bytes, got {len(packet)}")
        if packet[0] != STX:
            raise ValueError(f"Invalid STX: 0x{packet[0]:02X}")
        if packet[-1] != ETX:
            raise ValueError(f"Invalid ETX: 0x{packet[-1]:02X}")

        version = packet[1]
        reserved = (packet[2] << 8) | packet[3]
        dest = packet[4]
        data_len = int.from_bytes(packet[5:7], byteorder='little')
        command = packet[7]
        body0 = packet[8]
        body1 = packet[9]
        checksum_recv = packet[10]

        checksum_calc = compute_xor_checksum(packet[:10])
        checksum_ok = checksum_recv == checksum_calc

        emo_code = body0
        emo_intensity = (body1 >> 5) & 0x07
        state_intensity = (body1 >> 2) & 0x07

        kfer_id = PROTOCOL_TO_KFER.get(emo_code, -1)
        kfer_label = KFER_LABELS[kfer_id] if 0 <= kfer_id < len(KFER_LABELS) else "reserved"

        display_msg = PROTOCOL_MESSAGES.get(emo_code, "")

        return {
            "version": version,
            "dest": dest,
            "dest_name": "모트렉스" if dest == DEST_MOTREX else "슈퍼게이트",
            "data_length": data_len,
            "command": f"0x{command:02X}",
            "emotion_code": emo_code,
            "emotion_name": PROTOCOL_NAMES.get(emo_code, f"reserved_{emo_code}"),
            "emotion_kfer": kfer_label,
            "emotion_intensity": emo_intensity,
            "state_intensity": state_intensity,
            "display_message": display_msg,
            "checksum_ok": checksum_ok,
            "raw_hex": packet.hex().upper(),
            "raw_bytes": " ".join(f"0x{b:02X}" for b in packet),
        }


# ── Self-test ──
if __name__ == "__main__":
    enc = PacketEncoder()

    print("=" * 65)
    print("  복합감정인지 USB 통신 프로토콜 v2 (모트렉스 12-byte)")
    print("=" * 65)

    print("\n[구조] STX(1)+VER(1)+RSV(2)+DEST(1)+DLEN(2)+CMD(1)+BODY(2)+CHKSUM(1)+ETX(1) = 12B")

    # 모트렉스 예시 검증
    print("\n=== 모트렉스 예시 검증 ===")
    pkt = enc.encode_raw(emotion_code=0, emo_intensity=7, state_intensity=7)
    info = enc.decode(pkt)
    print(f"  공포(감정7,상태7): {info['raw_bytes']}")
    print(f"  기대값:            0x02 0x01 0x00 0x00 0x10 0x03 0x00 0xF0 0x00 0xFC 0x1C 0x03")
    print(f"  CHKSUM: {'OK' if info['checksum_ok'] else 'FAIL'}")

    print("\n=== Emotion Code Mapping ===")
    for kfer_id, label in enumerate(KFER_LABELS):
        proto_code = KFER_TO_PROTOCOL[kfer_id]
        proto_name = PROTOCOL_NAMES.get(proto_code, "Reserved")
        print(f"  K-FER {kfer_id} ({label:>10s}) → Code 0x{proto_code:02X} ({proto_name})")

    print("\n=== Encode/Decode Test ===")
    test_cases = [
        ("행복",    {"kfer_emotion_id": 2, "emotion_confidence": 0.95, "arousal": 0.3, "perclos": 0.05}),
        ("분노",    {"kfer_emotion_id": 0, "emotion_confidence": 0.85, "arousal": 0.8, "perclos": 0.1}),
        ("슬픔",    {"kfer_emotion_id": 5, "emotion_confidence": 0.70, "arousal": 0.4, "perclos": 0.05}),
        ("공포",    {"kfer_emotion_id": 1, "emotion_confidence": 0.90, "arousal": 0.7, "perclos": 0.0}),
        ("놀람",    {"kfer_emotion_id": 6, "emotion_confidence": 0.82, "arousal": 0.6, "perclos": 0.03}),
        ("혐오",    {"kfer_emotion_id": 3, "emotion_confidence": 0.70, "arousal": 0.5, "perclos": 0.08}),
        ("중립",    {"kfer_emotion_id": 4, "emotion_confidence": 0.60, "arousal": 0.5, "perclos": 0.05}),
    ]

    for label, tc in test_cases:
        pkt = enc.encode(**tc)
        info = enc.decode(pkt)
        print(f"\n  [{label}] {info['raw_bytes']}")
        print(f"    → {info['emotion_name']}(0x{info['emotion_code']:02X}), "
              f"emo_int={info['emotion_intensity']}/7, state_int={info['state_intensity']}/7, "
              f"CHKSUM={'OK' if info['checksum_ok'] else 'FAIL'}")

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)
