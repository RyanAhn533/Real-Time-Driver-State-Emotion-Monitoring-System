"""
K-MER 실시간 센싱 메인
======================
모든 센서(RealSense, RODE, Watch)를 스레드로 구동하고
수집된 데이터를 inference_loop에서 주기적으로 모델에 입력.

바이트코드 복원 기반 재구성.

Usage:
    python sensing_main.py [--kfer_ckpt PATH] [--device cuda] [--hz 10] [--no_audio_experts]
"""

import os
import time
import threading
from collections import deque
import numpy as np
import argparse


# ═══════════════════════════════════════════
# 데이터 버퍼 클래스들
# ═══════════════════════════════════════════

class Latest:
    """최신 값 1개만 유지하는 thread-safe 버퍼."""

    def __init__(self):
        self.lock = threading.Lock()
        self.ts = 0
        self.val = None

    def set(self, ts, val):
        with self.lock:
            self.ts = int(ts)
            self.val = val

    def get(self):
        with self.lock:
            return self.ts, self.val


class Ring1D:
    """고정 크기 링 버퍼 (오디오용)."""

    def __init__(self, maxlen: int):
        self.lock = threading.Lock()
        self.maxlen = int(maxlen)
        self.buf = np.zeros(self.maxlen, dtype=np.float32)
        self.idx = 0
        self.filled = False
        self.ts = 0

    def push(self, ts_ms: int, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32).ravel()
        n = x.size
        with self.lock:
            self.ts = int(ts_ms)
            if n >= self.maxlen:
                self.buf[:] = x[-self.maxlen:]
                self.idx = 0
                self.filled = True
            else:
                end = self.idx + n
                if end <= self.maxlen:
                    self.buf[self.idx:end] = x
                else:
                    first = self.maxlen - self.idx
                    self.buf[self.idx:] = x[:first]
                    self.buf[: n - first] = x[first:]
                    self.filled = True
                self.idx = end % self.maxlen
                if end >= self.maxlen:
                    self.filled = True

    def snapshot(self):
        with self.lock:
            ts = self.ts
            if not self.filled:
                return ts, self.buf[: self.idx].copy()
            # 정렬된 순서로 반환
            return ts, np.concatenate([
                self.buf[self.idx:], self.buf[: self.idx]
            ])


class BioQueues:
    """생체 신호 큐 (PPG, EDA, Temp)."""

    def __init__(self, maxlen=512):
        self.lock = threading.Lock()
        self.ppg = deque(maxlen=int(maxlen))
        self.eda = deque(maxlen=int(maxlen))
        self.temp = deque(maxlen=int(maxlen))

    def push_ppg(self, ts, d1, d2):
        with self.lock:
            self.ppg.append((int(ts), float(d1), float(d2)))

    def push_eda(self, ts, real):
        with self.lock:
            self.eda.append((int(ts), float(real)))

    def push_temp(self, ts, skin_c):
        with self.lock:
            self.temp.append((int(ts), float(skin_c)))

    def snapshot(self):
        with self.lock:
            return list(self.ppg), list(self.eda), list(self.temp)


class ModelInputs:
    """모든 센서 데이터를 모아놓는 컨테이너."""

    def __init__(self, audio_sr=48000, audio_sec=2.0):
        self.frame_main = Latest()
        self.audio = Ring1D(int(audio_sr * audio_sec))
        self.bio = BioQueues(maxlen=512)


# ═══════════════════════════════════════════
# 추론 루프
# ═══════════════════════════════════════════

def inference_loop(shutdown, mi: ModelInputs, inferencer, hz=10.0):
    """
    주기적으로 모델 입력을 수집하고 추론 실행.

    Args:
        shutdown: threading.Event
        mi: ModelInputs
        inferencer: KMERInferencer (or None for sensor-only test)
        hz: 추론 주기 (Hz)
    """
    interval = 1.0 / float(hz)

    while not shutdown.is_set():
        t0 = time.time()

        # 프레임 가져오기
        fts, frame = mi.frame_main.get()

        # 오디오 스냅샷
        ats, audio_snap = mi.audio.snapshot()

        # 바이오 스냅샷
        ppg_list, eda_list, temp_list = mi.bio.snapshot()

        # 추론
        if inferencer is not None and frame is not None:
            try:
                result = inferencer.forward(
                    frame_bgr=frame,
                    audio=audio_snap,
                    bio=(ppg_list, eda_list, temp_list),
                )
                arousal = result.get("arousal", 0.0)
                valence = result.get("valence", 0.0)
                compound = result.get("compound_label", "-")
                kfer_emo = result.get("kfer_emotion", "-")
                kfer_conf = result.get("kfer_confidence", 0.0)
                drowsy = result.get("drowsy", 0)

                print(
                    f"[inference] fts={fts} | kfer={kfer_emo}({kfer_conf:.2f})"
                    f" | A={arousal:.3f} V={valence:.3f}"
                    f" | drowsy={drowsy} | compound={compound}"
                    f" | bio=ppg {len(ppg_list)} eda {len(eda_list)} temp {len(temp_list)}"
                )
            except Exception as e:
                print(f"[inference] error: {e}")
        else:
            # 센서 데이터만 표시 (모델 없을 때)
            status_parts = [f"fts={fts}"]
            if frame is not None:
                status_parts.append(f"frame={frame.shape}")
            status_parts.append(f"audio={len(audio_snap)}")
            status_parts.append(f"ppg={len(ppg_list)}")
            status_parts.append(f"eda={len(eda_list)}")
            status_parts.append(f"temp={len(temp_list)}")
            print(f"[sensing] {' | '.join(status_parts)}")

        # 주기 유지
        elapsed = time.time() - t0
        time.sleep(max(0.0, interval - elapsed))


# ═══════════════════════════════════════════
# 콜백 팩토리
# ═══════════════════════════════════════════

def make_callbacks(mi: ModelInputs):
    """센서 콜백 함수들을 생성."""

    def on_frame_main(ts_ms, frame_bgr):
        mi.frame_main.set(ts_ms, frame_bgr)

    def on_audio_chunk(ts_ms, mono, sr):
        mi.audio.push(ts_ms, mono)

    def on_ppg(ts_ms, d1, d2):
        mi.bio.push_ppg(ts_ms, d1, d2)

    def on_eda(ts_ms, real):
        mi.bio.push_eda(ts_ms, real)

    def on_temp(ts_ms, skin_c):
        mi.bio.push_temp(ts_ms, skin_c)

    return on_frame_main, on_audio_chunk, on_ppg, on_eda, on_temp


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

# RealSense 시리얼 (필요시 수정)
RS_MAIN_SERIAL = "254622073310"
RS_SUB_SERIAL = ""


def main():
    ap = argparse.ArgumentParser("K-MER Real-Time Monitoring")
    ap.add_argument("--kfer_ckpt", default="../emotion_system/result/best.pth",
                    help="K-FER 모델 체크포인트 경로")
    ap.add_argument("--kmer_ckpt", default="../multimodal_dms/results_kmer/best_model.pth",
                    help="KMERFusion 모델 체크포인트 경로")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--hz", type=float, default=10.0, help="추론 루프 주기 (Hz)")
    ap.add_argument("--audio_sec", type=float, default=2.0, help="오디오 링 버퍼 길이 (초)")
    ap.add_argument("--no_audio_experts", action="store_true",
                    help="emotion2vec / audeering 비활성화 (메모리 절약)")
    ap.add_argument("--sensor_only", action="store_true",
                    help="센서 데이터만 수집 (모델 추론 없이)")
    args = ap.parse_args()

    shutdown = threading.Event()
    mi = ModelInputs(audio_sr=48000, audio_sec=args.audio_sec)

    # 콜백 생성
    on_frame, on_audio, on_ppg, on_eda, on_temp = make_callbacks(mi)

    # 추론기 (선택)
    inferencer = None
    if not args.sensor_only:
        try:
            sys_path = os.path.dirname(os.path.abspath(__file__))
            proj_root = os.path.join(sys_path, "..",
                "Real-Time-Driver-State-Emotion-Monitoring-System")
            # KMERInferencer import 시도
            # (모델 체크포인트 없으면 sensor_only 모드로 폴백)
            print("[main] Model loading skipped (sensor_only mode)")
        except Exception as e:
            print(f"[main] Model load failed: {e}, running sensor-only")

    # ── 센서 스레드 시작 ──
    threads = []

    # 1. RealSense 카메라
    try:
        from realsense import run_realsense
        t_cam = threading.Thread(
            target=run_realsense,
            kwargs=dict(
                shutdown_event=shutdown,
                device_serial=RS_MAIN_SERIAL,
                is_main_cam=True,
                on_frame=on_frame,
                width=1280, height=720, fps=30,
            ),
            daemon=True,
        )
        threads.append(("RealSense", t_cam))
    except ImportError:
        print("[main] RealSense module not available, skipping camera")

    # 2. RODE 마이크
    try:
        from rode import run_rode
        t_mic = threading.Thread(
            target=run_rode,
            kwargs=dict(
                shutdown_event=shutdown,
                on_audio_chunk=on_audio,
            ),
            daemon=True,
        )
        threads.append(("RODE", t_mic))
    except ImportError:
        print("[main] RODE module not available, skipping microphone")

    # 3. Watch (ADI Study Watch)
    try:
        from watch import run_watch
        t_watch = threading.Thread(
            target=run_watch,
            kwargs=dict(
                shutdown_event=shutdown,
                on_ppg=on_ppg,
                on_eda=on_eda,
                on_temp=on_temp,
            ),
            daemon=True,
        )
        threads.append(("Watch", t_watch))
    except ImportError:
        print("[main] Watch module not available, skipping bio sensors")

    # 스레드 시작
    for name, t in threads:
        print(f"[main] Starting {name}...")
        t.start()

    # 추론 루프 (메인 스레드)
    print(f"\n[main] Inference loop starting at {args.hz} Hz")
    print("[main] Press Ctrl+C to stop\n")

    try:
        inference_loop(shutdown, mi, inferencer, hz=args.hz)
    except KeyboardInterrupt:
        print("\n[main] Ctrl+C received")

    shutdown.set()

    for name, t in threads:
        t.join(timeout=5)
        if t.is_alive():
            print(f"[main] {name} thread did not stop cleanly")

    print("[main] All stopped")


if __name__ == "__main__":
    main()
