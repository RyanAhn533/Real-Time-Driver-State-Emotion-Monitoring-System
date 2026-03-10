"""
K-MER Sensor Test Utility
==========================
각 센서를 독립적으로 테스트하고 데이터 수신 상태를 확인.

기존 test_sensors.py의 로직을 보존하면서 config 시스템 연동 추가.

Usage:
    python test_sensors.py                     # 전체 센서 테스트
    python test_sensors.py --test camera       # 카메라만
    python test_sensors.py --test mic          # 마이크만
    python test_sensors.py --test watch        # 워치만
    python test_sensors.py --no_audio          # 마이크 제외
    python test_sensors.py --no_watch          # 워치 제외
"""

import sys
import argparse
import threading
import time
import numpy as np
from pathlib import Path

# ── 프로젝트 경로 설정 ──
_SENSING_DIR = Path(__file__).resolve().parent
_RAW_SENSING_DIR = (
    _SENSING_DIR
    / "raw_sensing_code"
    / "Real-Time-Driver-State-Emotion-Monitoring-System-ysh_sensing_260305"
    / "sensing"
)

for _p in [str(_RAW_SENSING_DIR), str(_SENSING_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 기존 버퍼 클래스 (동작 보존)
from sensing_main import Ring1D, BioQueues

# Config (선택적)
try:
    from config import load_config
    _cfg = load_config()
except Exception:
    _cfg = None

# ── 공유 상태 ────────────────────────────────────────────────────────────

audio_ring = Ring1D(maxlen=48000 * 2)
bio_queues = BioQueues(maxlen=512)

audio_count = 0
ppg_count = 0
eda_count = 0
temp_count = 0
frame_count = 0
count_lock = threading.Lock()


# ── 콜백 ─────────────────────────────────────────────────────────────────

def on_frame_main(ts_ms, frame_bgr):
    global frame_count
    with count_lock:
        frame_count += 1


def on_audio_chunk(ts_ms, mono, sr):
    global audio_count
    audio_ring.push(ts_ms, mono)
    with count_lock:
        audio_count += 1


def on_ppg(ts_ms, d1, d2):
    global ppg_count
    bio_queues.push_ppg(ts_ms, d1, d2)
    with count_lock:
        ppg_count += 1


def on_eda(ts_ms, real):
    global eda_count
    bio_queues.push_eda(ts_ms, real)
    with count_lock:
        eda_count += 1


def on_temp(ts_ms, skin_c):
    global temp_count
    bio_queues.push_temp(ts_ms, skin_c)
    with count_lock:
        temp_count += 1


# ── 모니터 루프 ──────────────────────────────────────────────────────────

def monitor_loop(shutdown: threading.Event, test_mode: str, interval: float = 1.0):
    global audio_count, ppg_count, eda_count, temp_count, frame_count
    prev = dict(audio=0, ppg=0, eda=0, temp=0, frame=0)

    while not shutdown.is_set():
        time.sleep(interval)

        with count_lock:
            cur = dict(audio=audio_count, ppg=ppg_count,
                       eda=eda_count, temp=temp_count, frame=frame_count)

        delta = {k: cur[k] - prev[k] for k in cur}
        prev = cur

        parts = [f"[{time.strftime('%H:%M:%S')}]"]

        if test_mode in ("all", "camera"):
            parts.append(f"CAM +{delta['frame']:3d}fps (total={cur['frame']})")

        if test_mode in ("all", "mic"):
            audio_ts, audio_buf = audio_ring.snapshot()
            rms = float(np.sqrt(np.mean(audio_buf ** 2))) if audio_buf.size > 0 else 0.0
            parts.append(
                f"AUDIO +{delta['audio']:3d}chunks "
                f"buf={audio_buf.size:6d}samples RMS={rms:.5f}"
            )

        if test_mode in ("all", "watch"):
            ppg_list, eda_list, temp_list = bio_queues.snapshot()
            ppg_str = f"d1={ppg_list[-1][1]:.1f} d2={ppg_list[-1][2]:.1f}" if ppg_list else "no data"
            eda_str = f"real={eda_list[-1][1]:.4f}" if eda_list else "no data"
            temp_str = f"skin={temp_list[-1][1]:.2f}C" if temp_list else "no data"
            parts.append(
                f"PPG +{delta['ppg']:3d} ({ppg_str}) | "
                f"EDA +{delta['eda']:3d} ({eda_str}) | "
                f"TEMP +{delta['temp']:3d} ({temp_str})"
            )

        print(" | ".join(parts))


# ── Safe sensor runner ───────────────────────────────────────────────────

def _safe_run(name, func, shutdown, **kwargs):
    """센서 실행, 실패 시 에러 출력 후 대기."""
    try:
        func(shutdown_event=shutdown, **kwargs)
    except Exception as e:
        import traceback
        print(f"[{name}][ERROR] {e}")
        traceback.print_exc()
        while not shutdown.is_set():
            time.sleep(1.0)


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser("K-MER Sensor Test")
    ap.add_argument("--test", choices=["all", "camera", "mic", "watch"],
                    default="all", help="테스트할 센서")
    ap.add_argument("--no_audio", action="store_true", help="마이크 비활성화")
    ap.add_argument("--no_watch", action="store_true", help="워치 비활성화")
    ap.add_argument("--no_camera", action="store_true", help="카메라 비활성화")
    args = ap.parse_args()

    shutdown = threading.Event()
    threads = []
    test_mode = args.test

    # 센서별 설정 (config 또는 기본값)
    rs_serial = _cfg.hardware.realsense_main.serial if _cfg else "021222070391"
    rs_width = _cfg.hardware.realsense_main.width if _cfg else 1280
    rs_height = _cfg.hardware.realsense_main.height if _cfg else 720
    rs_fps = _cfg.hardware.realsense_main.fps if _cfg else 30

    # ── Camera ──
    if test_mode in ("all", "camera") and not args.no_camera:
        from realsense import run_realsense
        threads.append(threading.Thread(
            target=_safe_run,
            args=("RS_MAIN", run_realsense, shutdown),
            kwargs=dict(
                device_serial=rs_serial,
                is_main_cam=True,
                on_frame=on_frame_main,
                width=rs_width, height=rs_height, fps=rs_fps,
            ),
            daemon=True, name="RS_MAIN",
        ))

    # ── Microphone ──
    if test_mode in ("all", "mic") and not args.no_audio:
        from rode import run_rode
        threads.append(threading.Thread(
            target=_safe_run,
            args=("RODE", run_rode, shutdown),
            kwargs=dict(on_audio_chunk=on_audio_chunk),
            daemon=True, name="RODE",
        ))

    # ── Watch ──
    if test_mode in ("all", "watch") and not args.no_watch:
        from watch import run_watch
        threads.append(threading.Thread(
            target=_safe_run,
            args=("WATCH", run_watch, shutdown),
            kwargs=dict(on_ppg=on_ppg, on_eda=on_eda, on_temp=on_temp),
            daemon=True, name="WATCH",
        ))

    # ── Monitor ──
    threads.append(threading.Thread(
        target=monitor_loop,
        args=(shutdown, test_mode),
        daemon=True, name="MONITOR",
    ))

    if len(threads) <= 1:
        print("[TEST] 테스트할 센서가 없습니다.")
        sys.exit(1)

    for t in threads:
        t.start()

    print(f"[TEST] 실행 중 (mode={test_mode})... Ctrl+C로 종료")
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[TEST] 종료 중...")
    finally:
        shutdown.set()
        for t in threads:
            t.join(timeout=3.0)
        print("[TEST] 완료")
