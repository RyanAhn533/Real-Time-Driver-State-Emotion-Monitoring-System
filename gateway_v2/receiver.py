#!/usr/bin/env python3
"""
K-MER 패킷 수신기 (로컬 PC에서 실행)
======================================
Jetson에서 TCP로 보낸 13-byte 모트렉스 패킷을 수신 + 실시간 디코딩 표시.

Usage (네 로컬 PC에서):
    python3 receiver.py --port 9000

그 다음 Jetson에서:
    python3 gateway_v2/demo_usb_send.py --simulate --tcp <네PC_IP>:9000
    python3 gateway_v2/demo_usb_send.py --live --tcp <네PC_IP>:9000
"""

import socket
import sys
import time
import argparse
from datetime import datetime

# ── 프로토콜 상수 (packet_encoder.py에서 복사, 단독 실행용) ──
STX = 0x02
ETX = 0x03
PACKET_SIZE = 13

PROTOCOL_NAMES = {0: "공포", 1: "놀람", 2: "분노", 3: "슬픔", 4: "행복/기쁨", 5: "혐오"}

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


def xor_checksum(data: bytes) -> int:
    r = 0
    for b in data:
        r ^= b
    return r


def decode_packet(packet: bytes) -> dict:
    if len(packet) != PACKET_SIZE:
        return {"error": f"size={len(packet)}"}
    if packet[0] != STX or packet[-1] != ETX:
        return {"error": f"STX=0x{packet[0]:02X} ETX=0x{packet[-1]:02X}"}

    body0 = packet[9]
    body1 = packet[10]
    checksum_ok = packet[11] == xor_checksum(packet[:11])

    emo_code = (body0 >> 4) & 0x0F
    stress = bool((body0 >> 3) & 1)
    low_attn = bool((body0 >> 2) & 1)
    drowsy = bool((body0 >> 1) & 1)
    emo_int = (body1 >> 5) & 0x07
    state_int = (body1 >> 2) & 0x07

    # 표시 메시지
    msg = ""
    if drowsy:
        msg = STATUS_MESSAGES["drowsy"]
    elif low_attn:
        msg = STATUS_MESSAGES["low_attention"]
    elif stress:
        msg = STATUS_MESSAGES["stress"]
    elif emo_code in PROTOCOL_MESSAGES:
        msg = PROTOCOL_MESSAGES[emo_code]

    return {
        "emotion_code": emo_code,
        "emotion_name": PROTOCOL_NAMES.get(emo_code, f"Reserved(0x{emo_code:X})"),
        "stress": stress,
        "low_attention": low_attn,
        "drowsy": drowsy,
        "emo_intensity": emo_int,
        "state_intensity": state_int,
        "display_message": msg,
        "checksum_ok": checksum_ok,
        "raw_hex": " ".join(f"{b:02X}" for b in packet),
    }


# ── 표시 색상 (ANSI) ──
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def colorize_status(info: dict) -> str:
    if info["drowsy"]:
        return f"{RED}{BOLD}DROWSY{RESET}"
    elif info["low_attention"]:
        return f"{YELLOW}{BOLD}LOW_ATTN{RESET}"
    elif info["stress"]:
        return f"{RED}STRESS{RESET}"
    else:
        return f"{GREEN}NORMAL{RESET}"


def print_header():
    print(f"\n{BOLD}{'='*78}{RESET}")
    print(f"{BOLD}  K-MER 복합감정인지 패킷 수신기 (모트렉스 13-byte){RESET}")
    print(f"{BOLD}{'='*78}{RESET}")
    print(f"  {CYAN}STX=0x02  ETX=0x03  COMMAND=0xF0  PACKET={PACKET_SIZE}B{RESET}")
    print(f"{BOLD}{'='*78}{RESET}\n")


def print_packet_live(info: dict, count: int):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    status = colorize_status(info)
    chk = f"{GREEN}OK{RESET}" if info["checksum_ok"] else f"{RED}FAIL{RESET}"

    emo = info["emotion_name"]
    if info["emotion_code"] > 5:
        emo = f"(표시안함)"

    print(f"  {CYAN}[{count:04d}]{RESET} {ts}  {info['raw_hex']}")
    print(f"         감정: {BOLD}{emo}{RESET} (code={info['emotion_code']:X}) "
          f" 강도: emo={info['emo_intensity']}/7 state={info['state_intensity']}/7")
    print(f"         상태: {status}  CHKSUM: {chk}")

    msg = info["display_message"]
    if msg:
        print(f"         {YELLOW}▶ \"{msg}\"{RESET}")
    print()


def run_server(bind_port: int):
    print_header()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", bind_port))
    server.listen(1)

    print(f"  대기 중... 0.0.0.0:{bind_port}")
    print(f"  Jetson에서 연결하세요:")
    print(f"    python3 gateway_v2/demo_usb_send.py --simulate --tcp <이PC_IP>:{bind_port}")
    print(f"    python3 gateway_v2/demo_usb_send.py --live --tcp <이PC_IP>:{bind_port}")
    print()

    try:
        while True:
            conn, addr = server.accept()
            print(f"  {GREEN}연결됨: {addr[0]}:{addr[1]}{RESET}\n")

            count = 0
            buf = b""

            try:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break

                    buf += data

                    # 13-byte 패킷 단위로 처리
                    while len(buf) >= PACKET_SIZE:
                        # STX 찾기
                        stx_pos = buf.find(bytes([STX]))
                        if stx_pos < 0:
                            buf = b""
                            break
                        if stx_pos > 0:
                            buf = buf[stx_pos:]

                        if len(buf) < PACKET_SIZE:
                            break

                        pkt = buf[:PACKET_SIZE]
                        buf = buf[PACKET_SIZE:]

                        if pkt[-1] != ETX:
                            # ETX 안 맞으면 1바이트 건너뛰기
                            buf = pkt[1:] + buf
                            continue

                        count += 1
                        info = decode_packet(pkt)
                        if "error" in info:
                            print(f"  [ERROR] {info['error']}")
                        else:
                            print_packet_live(info, count)

            except ConnectionResetError:
                pass
            finally:
                conn.close()
                print(f"  연결 종료. 수신 패킷: {count}개\n")
                print(f"  다시 대기 중...")

    except KeyboardInterrupt:
        print(f"\n  수신기 종료.")
    finally:
        server.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-MER 패킷 수신기")
    parser.add_argument("--port", type=int, default=9000, help="수신 포트 (default: 9000)")
    args = parser.parse_args()
    run_server(args.port)
