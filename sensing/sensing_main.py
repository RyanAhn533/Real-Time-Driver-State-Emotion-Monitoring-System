import os, time, threading
import numpy as np

class Latest:
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
    # float sample ring
    def __init__(self, maxlen: int):
        self.lock = threading.Lock()
        self.maxlen = int(maxlen)
        self.buf = np.zeros(self.maxlen, dtype=np.float32)
        self.idx = 0
        self.filled = False
        self.ts = 0

    def push(self, ts_ms: int, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32).ravel()
        if x.size == 0:
            return
        with self.lock:
            self.ts = int(ts_ms)
            n = x.size
            if n >= self.maxlen:
                self.buf[:] = x[-self.maxlen:]
                self.idx = 0
                self.filled = True
                return
            end = self.idx + n
            if end <= self.maxlen:
                self.buf[self.idx:end] = x
            else:
                k = self.maxlen - self.idx
                self.buf[self.idx:] = x[:k]
                self.buf[:end - self.maxlen] = x[k:]
            self.idx = end % self.maxlen
            if self.idx == 0:
                self.filled = True
    
    def snapshot(self):
        with self.lock:
            ts = self.ts
            if not self.filled:
                return ts, self.buf[:self.idx].copy()
            return ts, np.concatenate([self.buf[self.idx:], self.buf[:self.idx]]).copy()
        
# 생체신호 큐
class BioQueues:
    # TODO: maxlen 512 적당한지 확인 필요
    def __init__(self, maxlen=512):
        self.lock = threading.Lock()
        # ppg: ts, d1, d2
        self.ppg = []
        # eda: ts,real
        self.eda = []
        # temp: ts,skin_c
        self.temp = []
        self.maxlen = int(maxlen)

    def _append(self, arr, item):
        arr.append(item)
        if len(arr) > self.maxlen:
            del arr[:len(arr)-self.maxlen]

    def push_ppg(self, ts, d1, d2):
        with self.lock: self._append(self.ppg, (int(ts), float(d1), float(d2)))

    def push_eda(self, ts, real):
        with self.lock: self._append(self.eda, (int(ts), float(real)))

    def push_temp(self, ts, skin_c):
        with self.lock: self._append(self.temp, (int(ts), float(skin_c)))

    def snapshot(self):
        with self.lock:
            return list(self.ppg), list(self.eda), list(self.temp)
        
class ModelInputs:
    # TODO: audio_sec 2초 적당한지 확인 필요
    def __init__(self, audio_sr=48000, audio_sec=2.0):
        self.frame_main = Latest()
        # self.frame_sub = Latest()
        self.audio = Ring1D(maxlen=int(audio_sr * audio_sec))
        self.bio = BioQueues(maxlen=512)

# Inference thread
def inference_loop(shutdown: threading.Event, mi: ModelInputs, inferencer, hz=10):
    dt = 1.0 / float(hz)
    last_print = 0.0
    while not shutdown.is_set():
        t0 = time.time()

        fts, frame = mi.frame_main.get()
        _, audio = mi.audio.snapshot()
        ppg, eda, temp = mi.bio.snapshot()

        if frame is not None:
            result = inferencer.forward(frame, audio, ppg, eda, temp)
            now = time.time()
            if now - last_print >= 1.0:
                arousal  = result.get("arousal")
                valence  = result.get("valence")
                compound = result.get("compound_label", "-")
                kfer_emo = result.get("kfer_emotion") or "-"
                kfer_cf  = result.get("kfer_confidence", 0.0)
                drowsy   = ["alert", "drowsy", "sleeping"][result.get("drowsy", 0)]
                av_str   = (f"A={arousal:.3f} V={valence:.3f}"
                            if valence is not None else f"A={arousal:.3f}")
                print(
                    f"[inference] fts={fts} | "
                    f"kfer={kfer_emo}({kfer_cf:.2f}) | "
                    f"{av_str} | drowsy={drowsy} | "
                    f"compound={compound} | "
                    f"bio={len(ppg)}ppg {len(eda)}eda {len(temp)}temp"
                )
                last_print = now

        time.sleep(max(0.0, dt - (time.time() - t0)))

# sensor threads + inference
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("K-MER Real-Time Monitoring")
    ap.add_argument("--kfer_ckpt", default="../emotion_system/result/best.pth",
                    help="K-FER 모델 체크포인트 경로")
    ap.add_argument("--kmer_ckpt", default="../multimodal_dms/results_kmer/best_model.pth",
                    help="KMERFusion 모델 체크포인트 경로")
    ap.add_argument("--device",    default="cuda",  help="cuda or cpu")
    ap.add_argument("--hz",        type=float, default=10.0, help="추론 루프 주기 (Hz)")
    ap.add_argument("--audio_sec", type=float, default=2.0,  help="오디오 링 버퍼 길이 (초)")
    ap.add_argument("--no_audio_experts", action="store_true",
                    help="emotion2vec / audeering 비활성화 (메모리 절약)")
    args = ap.parse_args()

    # ── KMERInferencer 초기화 ──────────────────────────────────────────
    from kmer_inferencer import KMERInferencer
    inferencer = KMERInferencer(
        kfer_ckpt            = args.kfer_ckpt,
        kmer_ckpt            = args.kmer_ckpt,
        device               = args.device,
        enable_audio_experts = not args.no_audio_experts,
    )

    shutdown = threading.Event()
    mi = ModelInputs(audio_sr=48000, audio_sec=args.audio_sec)

    # 센서 콜백
    def on_frame_main(ts_ms: int, frame_bgr: np.ndarray):
        mi.frame_main.set(ts_ms, frame_bgr)

    def on_audio_chunk(ts_ms: int, mono: np.ndarray, sr: int):
        mi.audio.push(ts_ms, mono)

    def on_ppg(ts_ms: int, d1: float, d2: float):
        mi.bio.push_ppg(ts_ms, d1, d2)

    def on_eda(ts_ms: int, real: float):
        mi.bio.push_eda(ts_ms, real)

    def on_temp(ts_ms: int, skin_c: float):
        mi.bio.push_temp(ts_ms, skin_c)

    # ---- sensor threads ----
    from realsense import run_realsense
    from rode import run_rode
    from watch import run_watch

    # 카메라 시리얼 하드코딩(카메라 변경 시 시리얼 번호 바꿔야 함)
    RS_MAIN_SERIAL = "021222070391"
    RS_SUB_SERIAL  = "405622073483"

    threads = [
        threading.Thread(
            target=run_realsense,
            kwargs=dict(
                shutdown_event=shutdown,
                device_serial=RS_MAIN_SERIAL,
                is_main_cam=True,
                on_frame=on_frame_main,
            ),
            daemon=True,
            name="RS_MAIN",
        ),
        threading.Thread(
            target=run_realsense,
            kwargs=dict(
                shutdown_event=shutdown,
                device_serial=RS_SUB_SERIAL,
                is_main_cam=False,
                on_frame=None,  # 서브캠은 모델에 안 넣으면 None
            ),
            daemon=True,
            name="RS_SUB",
        ),
        threading.Thread(
            target=run_rode,
            kwargs=dict(
                shutdown_event=shutdown,
                on_audio_chunk=on_audio_chunk,
            ),
            daemon=True,
            name="RODE",
        ),
        threading.Thread(
            target=run_watch,
            kwargs=dict(
                shutdown_event=shutdown,
                on_ppg=on_ppg,
                on_eda=on_eda,
                on_temp=on_temp,
            ),
            daemon=True,
            name="WATCH",
        ),
        threading.Thread(
            target=inference_loop,
            args=(shutdown, mi, inferencer),
            kwargs=dict(hz=args.hz),
            daemon=True,
            name="INFER",
        ),
    ]

    for t in threads: t.start()
    print("[MAIN] started")
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[MAIN] stopping")
    finally:
        shutdown.set()
        for t in threads: t.join(timeout=2.0)
        print("[MAIN] done.")