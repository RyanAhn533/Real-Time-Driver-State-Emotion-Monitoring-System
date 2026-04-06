#!/usr/bin/env python3
"""
K-MER USB 프로토콜 시연 스크립트
=================================
모트렉스 13-byte 패킷을 USB 시리얼 또는 TCP 소켓으로 전송.

모드:
  1. --live     : 카메라 실시간 → KMERInferencer → 13-byte 패킷 전송 (E2E)
  2. --simulate : 모델 없이 시나리오 시퀀스 전송 (통신 테스트)
  3. --loopback : 전송 + 수신 확인 (TX→RX 루프백)
  4. --dry-run  : 패킷만 생성, 전송 안 함 (프로토콜 검증)

전송 방식:
  --port /dev/ttyUSB0   : USB Serial
  --tcp <HOST>:<PORT>   : TCP 소켓 (네트워크 테스트, 로컬PC 수신)

Usage:
    # 드라이런
    python3 gateway_v2/demo_usb_send.py --dry-run

    # TCP로 시뮬레이션 전송 (로컬PC에서 receiver.py 실행 후)
    python3 gateway_v2/demo_usb_send.py --simulate --tcp 192.168.1.10:9000

    # TCP로 카메라 라이브 전송
    python3 gateway_v2/demo_usb_send.py --live --tcp 192.168.1.10:9000

    # USB Serial 전송
    python3 gateway_v2/demo_usb_send.py --simulate --port /dev/ttyUSB0
"""

import argparse
import time
import sys
import os
from pathlib import Path

# 프로젝트 루트를 path에 추가
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from gateway_v2.packet_encoder import (
    PacketEncoder, KFER_LABELS, PROTOCOL_NAMES, PROTOCOL_MESSAGES,
    STATUS_MESSAGES, KFER_TO_PROTOCOL, PACKET_SIZE,
)
from gateway_v2.gateway_sender import GatewaySender, TcpGatewaySender, NullGatewaySender


def make_sender(args):
    """args에서 TCP 또는 USB sender 생성."""
    if hasattr(args, 'tcp') and args.tcp:
        host, port = args.tcp.split(":")
        return TcpGatewaySender(host=host, port=int(port))
    else:
        return GatewaySender(port=args.port, baudrate=args.baudrate)


def print_packet(info: dict, label: str = ""):
    """패킷 디코딩 결과를 예쁘게 출력."""
    prefix = f"[{label}] " if label else ""
    emo = info['emotion_name']
    if info['emotion_code'] > 5:
        emo = f"Reserved(0x{info['emotion_code']:X})"

    flags = []
    if info['stress']:
        flags.append("STRESS")
    if info['low_attention']:
        flags.append("LOW_ATTN")
    if info['drowsy']:
        flags.append("DROWSY")
    flags_str = " | ".join(flags) if flags else "NORMAL"

    print(f"{prefix}{info['raw_bytes']}")
    print(f"  감정: {emo} (code={info['emotion_code']:X}), "
          f"강도: emo={info['emotion_intensity']}/7, state={info['state_intensity']}/7")
    print(f"  상태: {flags_str}")
    print(f"  표시: \"{info['display_message']}\"")
    print(f"  DEST: {info['dest_name']} (0x{info['dest']:02X}), "
          f"CHKSUM: {'OK' if info['checksum_ok'] else 'FAIL'}")


# ── 시뮬레이션 시나리오 ──

SCENARIOS = [
    ("정상 운전 (행복)",       2, 0.90, 0.4, 0.05),   # happy
    ("행복 유지",             2, 0.92, 0.35, 0.03),   # happy
    ("중립 전환",             4, 0.70, 0.5, 0.05),    # neutral → reserved
    ("공포 감지",             1, 0.80, 0.6, 0.08),    # anxious
    ("공포 + 스트레스",       1, 0.85, 0.75, 0.05),   # anxious + stress
    ("분노 감지",             0, 0.88, 0.8, 0.10),    # angry + stress
    ("분노 지속",             0, 0.90, 0.82, 0.08),   # angry + stress
    ("슬픔 감지",             5, 0.75, 0.35, 0.05),   # sad
    ("혐오 감지",             3, 0.70, 0.45, 0.05),   # hurt
    ("주의 분산",             4, 0.60, 0.5, 0.30),    # neutral + low attention
    ("졸음 전조",             4, 0.55, 0.3, 0.35),    # neutral + low attention
    ("졸음 감지!",            4, 0.50, 0.2, 0.55),    # neutral + drowsy
    ("졸음 위험!!",           4, 0.40, 0.15, 0.70),   # neutral + drowsy severe
    ("놀람 (이벤트)",         6, 0.95, 0.8, 0.05),    # surprised
    ("정상 복귀",             2, 0.85, 0.5, 0.05),    # happy
]


def run_dry_run():
    """패킷만 생성, 전송 없음."""
    print("=" * 70)
    print("  K-MER 모트렉스 프로토콜 v2 — Dry Run (13-byte)")
    print("=" * 70)
    print(f"  패킷 크기: {PACKET_SIZE} bytes")
    print(f"  STX=0x02, ETX=0x03, COMMAND=0xF0, DEST=0x10")
    print()

    enc = PacketEncoder()

    for label, kfer_id, conf, arousal, perclos in SCENARIOS:
        pkt = enc.encode(kfer_id, conf, arousal, perclos)
        info = enc.decode(pkt)
        print(f"--- {label} ---")
        print_packet(info)
        print()


def run_simulate(args, interval: float):
    """시나리오 시퀀스를 USB 또는 TCP로 전송."""
    sender = make_sender(args)
    target = args.tcp if args.tcp else f"{args.port} @ {args.baudrate}"
    print("=" * 70)
    print(f"  K-MER 시뮬레이션 전송 — {target}")
    print("=" * 70)

    enc = PacketEncoder()

    if not sender.connect():
        print(f"ERROR: 연결 실패 ({target}). --dry-run 으로 테스트하세요.")
        return

    print(f"  연결 성공. {len(SCENARIOS)}개 시나리오를 {interval}초 간격으로 전송합니다.")
    print()

    try:
        for i, (label, kfer_id, conf, arousal, perclos) in enumerate(SCENARIOS):
            pkt = enc.encode(kfer_id, conf, arousal, perclos)
            info = enc.decode(pkt)

            ok = sender.send(pkt)
            status = "TX OK" if ok else "TX FAIL"

            print(f"[{i+1:02d}/{len(SCENARIOS):02d}] {label} — {status}")
            print_packet(info, "  ")
            print()

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        stats = sender.stats
        print(f"\n전송 완료: sent={stats['send_count']}, errors={stats['error_count']}")
        sender.close()


def run_loopback(port: str, baudrate: int):
    """TX → RX 루프백 테스트 (TX/RX 핀 연결 필요)."""
    print("=" * 70)
    print(f"  K-MER 루프백 테스트 — {port}")
    print("=" * 70)

    try:
        import serial
    except ImportError:
        print("ERROR: pyserial 필요. pip install pyserial")
        return

    enc = PacketEncoder()

    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=1.0)
    except Exception as e:
        print(f"ERROR: 포트 열기 실패: {e}")
        return

    print("  TX/RX 루프백 핀을 연결하세요.")
    print()

    test_cases = [
        (2, 0.90, 0.4, 0.05),   # happy
        (0, 0.85, 0.8, 0.10),   # angry + stress
        (4, 0.50, 0.2, 0.55),   # drowsy
    ]

    passed = 0
    for kfer_id, conf, arousal, perclos in test_cases:
        pkt = enc.encode(kfer_id, conf, arousal, perclos)

        # TX
        ser.write(pkt)
        ser.flush()

        # RX
        rx = ser.read(PACKET_SIZE)

        if rx == pkt:
            result = "PASS"
            passed += 1
        else:
            result = "FAIL"

        info = enc.decode(pkt)
        print(f"  [{result}] {info['emotion_name']:>8s} | TX={pkt.hex().upper()} | RX={rx.hex().upper() if rx else 'EMPTY'}")

    ser.close()
    print(f"\n결과: {passed}/{len(test_cases)} passed")


def run_live(args, hz: float):
    """카메라 실시간 → E2E → USB/TCP 전송."""
    target = args.tcp if args.tcp else f"{args.port} @ {args.baudrate}"
    print("=" * 70)
    print(f"  K-MER Live E2E — {target}, {hz}Hz")
    print("=" * 70)

    from gateway_v2.e2e_pipeline import E2EPipeline

    # E2E 파이프라인은 gateway 비활성화 (직접 sender 사용)
    pipeline = E2EPipeline(
        gateway_enabled=False,
        device="cuda",
    )
    sender = make_sender(args)

    print("  모델 로딩 중...")
    pipeline.initialize()

    if not sender.connect():
        print(f"WARNING: 전송 연결 실패 ({target}). 패킷 생성만 합니다.")
    print("  모델 로드 완료. 카메라 시작...")

    # 카메라
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # RealSense fallback
        try:
            import pyrealsense2 as rs
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipe.start(cfg)
            use_rs = True
            print("  RealSense 카메라 사용")
        except Exception:
            print("ERROR: 카메라 없음")
            return
    else:
        use_rs = False
        print("  USB/UVC 카메라 사용")

    interval = 1.0 / hz
    cycle = 0

    try:
        while True:
            t0 = time.time()

            # 프레임 획득
            if use_rs:
                frames = pipe.wait_for_frames(1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                import numpy as np
                frame = np.asarray(color.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    continue

            # E2E 처리
            result = pipeline.process_cycle(frame)
            cycle += 1

            # 패킷 전송
            pkt = result.get("packet")
            if pkt and sender.is_connected:
                sender.send(pkt)

            # 출력
            pkt_hex = result.get("packet_bytes", "N/A")
            emo = result.get("emotion_name_ko", "")
            msg = result.get("display_message", "")
            mode = result.get("mode", "?")

            stress = "S" if result.get("stress") else "-"
            attn = "A" if result.get("low_attention") else "-"
            drowsy = "D" if result.get("drowsy") else "-"
            fatigue = "F" if result.get("fatigue") else "-"

            latency_ms = (time.time() - t0) * 1000

            if cycle % 5 == 0 or cycle <= 3:
                print(f"[{cycle:04d}] {latency_ms:5.1f}ms | {mode:>8s} | "
                      f"{emo:>6s} [{stress}{attn}{drowsy}{fatigue}] | "
                      f"{msg}")
                if pkt_hex and pkt_hex != "N/A":
                    print(f"         PKT: {pkt_hex}")

            # Hz 맞추기
            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    except KeyboardInterrupt:
        print("\n종료...")
    finally:
        sender.close()
        pipeline.shutdown()
        if use_rs:
            pipe.stop()
        else:
            cap.release()
        stats = sender.stats
        print(f"완료. 전송: {stats['send_count']}개, 에러: {stats['error_count']}개")


def main():
    parser = argparse.ArgumentParser(description="K-MER 모트렉스 USB 프로토콜 시연")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--live", action="store_true", help="카메라 실시간 E2E + USB 전송")
    mode_group.add_argument("--simulate", action="store_true", help="시나리오 시퀀스 USB 전송")
    mode_group.add_argument("--loopback", action="store_true", help="TX→RX 루프백 테스트")
    mode_group.add_argument("--dry-run", action="store_true", help="패킷 생성만 (전송 안 함)")

    parser.add_argument("--port", default="/dev/ttyUSB0", help="USB 시리얼 포트 (default: /dev/ttyUSB0)")
    parser.add_argument("--baudrate", type=int, default=115200, help="시리얼 통신 속도")
    parser.add_argument("--tcp", default=None, help="TCP 전송: HOST:PORT (예: 192.168.1.10:9000)")
    parser.add_argument("--interval", type=float, default=1.0, help="시뮬레이션 전송 간격 (초)")
    parser.add_argument("--hz", type=float, default=10.0, help="라이브 모드 추론 Hz")

    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
    elif args.simulate:
        run_simulate(args, args.interval)
    elif args.loopback:
        run_loopback(args.port, args.baudrate)
    elif args.live:
        run_live(args, args.hz)


if __name__ == "__main__":
    main()
