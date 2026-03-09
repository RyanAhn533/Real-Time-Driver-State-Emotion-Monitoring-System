"""
마이크(RØDE) + 워치 센서 수신 확인용 테스트 스크립트.
실행: python test_sensors.py [--no_audio] [--no_watch]
"""
import argparse
import threading
import time
import numpy as np
from sensing_main import BioQueues, Ring1D

# ── 공유 상태 ──────────────────────────────────────────────────────────────────
audio_ring  = Ring1D(maxlen=48000 * 2)   # 2초
bio_queues  = BioQueues(maxlen=512)

audio_count  = 0   # 수신된 청크 수
ppg_count    = 0
eda_count    = 0
temp_count   = 0
count_lock   = threading.Lock()

# ── 콜백 ───────────────────────────────────────────────────────────────────────
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

# ── 출력 루프 ──────────────────────────────────────────────────────────────────
def monitor_loop(shutdown: threading.Event, interval: float = 1.0):
    global audio_count, ppg_count, eda_count, temp_count
    prev = dict(audio=0, ppg=0, eda=0, temp=0)

    while not shutdown.is_set():
        time.sleep(interval)

        with count_lock:
            cur = dict(audio=audio_count, ppg=ppg_count,
                       eda=eda_count,    temp=temp_count)

        delta = {k: cur[k] - prev[k] for k in cur}
        prev  = cur

        audio_ts, audio_buf = audio_ring.snapshot()
        ppg_list, eda_list, temp_list = bio_queues.snapshot()

        # 오디오 RMS
        rms = float(np.sqrt(np.mean(audio_buf ** 2))) if audio_buf.size > 0 else 0.0

        # 최신 생체 신호 값
        ppg_str  = f"d1={ppg_list[-1][1]:.1f} d2={ppg_list[-1][2]:.1f}" if ppg_list  else "no data"
        eda_str  = f"real={eda_list[-1][1]:.4f}"                          if eda_list  else "no data"
        temp_str = f"skin={temp_list[-1][1]:.2f}°C"                       if temp_list else "no data"

        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"AUDIO +{delta['audio']:3d}chunks buf={audio_buf.size:6d}samples RMS={rms:.5f} | "
            f"PPG +{delta['ppg']:3d} ({ppg_str}) | "
            f"EDA +{delta['eda']:3d} ({eda_str}) | "
            f"TEMP +{delta['temp']:3d} ({temp_str})"
        )

# ── 메인 ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Sensor data check")
    ap.add_argument("--no_audio", action="store_true", help="마이크 비활성화")
    ap.add_argument("--no_watch", action="store_true", help="워치 비활성화")
    args = ap.parse_args()

    shutdown = threading.Event()
    threads  = []

    if not args.no_audio:
        from rode import run_rode
        threads.append(threading.Thread(
            target=run_rode,
            kwargs=dict(shutdown_event=shutdown, on_audio_chunk=on_audio_chunk),
            daemon=True, name="RODE",
        ))

    if not args.no_watch:
        from watch import run_watch
        def run_watch_safe(**kwargs):
            try:
                run_watch(**kwargs)
            except Exception as e:
                import traceback
                print(f"[WATCH][ERROR] {e}")
                traceback.print_exc()
        threads.append(threading.Thread(
            target=run_watch_safe,
            kwargs=dict(shutdown_event=shutdown,
                        on_ppg=on_ppg, on_eda=on_eda, on_temp=on_temp),
            daemon=True, name="WATCH",
        ))

    threads.append(threading.Thread(
        target=monitor_loop,
        args=(shutdown,),
        daemon=True, name="MONITOR",
    ))

    for t in threads:
        t.start()

    print("[TEST] 실행 중... Ctrl+C로 종료")
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
