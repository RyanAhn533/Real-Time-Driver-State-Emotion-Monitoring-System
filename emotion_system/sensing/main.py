"""
Real-Time Multimodal Driver Emotion Monitoring
================================================
센서 스레드 (RealSense / RØDE / Watch) + 전체 멀티모달 추론.

Modality 1 (Video)  : RealSense RGB → FER (emotion 7-class + AU coords)
Modality 2 (Audio)  : RØDE Wireless GO II → mel spectrogram → arousal/valence
Modality 3 (Bio)    : ADI Study Watch → PPG/EDA/TEMP → bio features → arousal/valence

실행:
    cd emotion_system
    python sensing/main.py \
        --fer_ckpt result/best.pth \
        --av_ckpt  multimodal/checkpoints/ckpt_v3/fold_1/best.pth
"""

import os, sys, time, threading, argparse
from pathlib import Path

import numpy as np

# emotion_system 루트를 sys.path에 추가
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


# ─────────────────────────────────────────────
# 센서 데이터 버퍼
# ─────────────────────────────────────────────
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
    """float sample ring buffer"""
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


class BioQueues:
    def __init__(self, maxlen=512):
        self.lock = threading.Lock()
        self.ppg  = []   # (ts, d1, d2)
        self.eda  = []   # (ts, real)
        self.temp = []   # (ts, skin_c)
        self.maxlen = int(maxlen)

    def _append(self, arr, item):
        arr.append(item)
        if len(arr) > self.maxlen:
            del arr[:len(arr) - self.maxlen]

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
    def __init__(self, audio_sr=48000, audio_sec=2.0):
        self.frame_main = Latest()
        self.audio = Ring1D(maxlen=int(audio_sr * audio_sec))
        self.bio = BioQueues(maxlen=512)


# ─────────────────────────────────────────────
# 추론 스레드
# ─────────────────────────────────────────────
def inference_loop(
    shutdown: threading.Event,
    mi: ModelInputs,
    inferencer,
    hz: float = 10.0,
):
    """
    멀티모달 추론 루프.
    - FER:  frame_main  → emotion, confidence, AU coords
    - Audio: mi.audio   → mel → arousal/valence  (av_model 로드 시)
    - Bio:   mi.bio     → hand-crafted features → arousal/valence  (av_model 로드 시)
    """
    dt = 1.0 / hz
    last_print = 0.0

    while not shutdown.is_set():
        t0 = time.time()

        fts, frame  = mi.frame_main.get()
        ats, audio  = mi.audio.snapshot()
        ppg, eda, temp = mi.bio.snapshot()

        if frame is not None:
            result = inferencer.forward(frame, audio, ppg, eda, temp)

            now = time.time()
            if now - last_print >= 1.0:
                emotion    = result.get("emotion")   or "–"
                confidence = result.get("confidence") or 0.0
                arousal    = result.get("arousal")
                valence    = result.get("valence")
                av_str = (
                    f"A={arousal:.3f}  V={valence:.3f}"
                    if arousal is not None else "A/V: (av_model 미로드)"
                )
                print(
                    f"[inference] fts={fts} | "
                    f"emotion={emotion} ({confidence:.2f}) | "
                    f"{av_str} | "
                    f"bio={len(ppg)}ppg {len(eda)}eda {len(temp)}temp"
                )
                last_print = now

        time.sleep(max(0.0, dt - (time.time() - t0)))


# ─────────────────────────────────────────────
# 엔트리 포인트
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("Multimodal Driver Monitoring")
    ap.add_argument(
        "--fer_ckpt",
        default=str(_ROOT / "result" / "best.pth"),
        help="FER 모델 체크포인트 경로 (AUFERModel)",
    )
    ap.add_argument(
        "--av_ckpt",
        default=None,
        help="AV student 모델 체크포인트 경로 (ckpt_v3 fold best.pth). "
             "av_model_cls를 코드에서 주입해야 활성화됩니다.",
    )
    ap.add_argument("--hz",    type=float, default=10.0,  help="추론 루프 주기 (Hz)")
    ap.add_argument("--audio_sec", type=float, default=2.0, help="오디오 링 버퍼 길이 (초)")
    args = ap.parse_args()

    # ── MultimodalInferencer 생성 ──────────────────────────────────────
    from inference.multimodal_inferencer import MultimodalInferencer

    # ckpt_v3 모델 클래스가 준비되면 아래처럼 주입하세요:
    #   from models.audio_bio_student import AudioBioStudentModel
    #   inferencer = MultimodalInferencer(
    #       fer_ckpt=args.fer_ckpt,
    #       av_ckpt=args.av_ckpt,
    #       av_model_cls=AudioBioStudentModel,
    #   )
    inferencer = MultimodalInferencer(
        fer_ckpt=args.fer_ckpt,
        av_ckpt=args.av_ckpt,
        av_model_cls=None,          # TODO: ckpt_v3 model class 주입
    )

    # ── 공유 버퍼 ─────────────────────────────────────────────────────
    shutdown = threading.Event()
    mi = ModelInputs(audio_sr=48000, audio_sec=args.audio_sec)

    # 센서 → 버퍼 콜백
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

    # ── 센서 스레드 ───────────────────────────────────────────────────
    from sensing.realsense import run_realsense
    from sensing.rode import run_rode
    from sensing.watch import run_watch

    # 카메라 시리얼 (카메라 변경 시 수정)
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
            daemon=True, name="RS_MAIN",
        ),
        threading.Thread(
            target=run_realsense,
            kwargs=dict(
                shutdown_event=shutdown,
                device_serial=RS_SUB_SERIAL,
                is_main_cam=False,
                on_frame=None,
            ),
            daemon=True, name="RS_SUB",
        ),
        threading.Thread(
            target=run_rode,
            kwargs=dict(
                shutdown_event=shutdown,
                on_audio_chunk=on_audio_chunk,
            ),
            daemon=True, name="RODE",
        ),
        threading.Thread(
            target=run_watch,
            kwargs=dict(
                shutdown_event=shutdown,
                on_ppg=on_ppg,
                on_eda=on_eda,
                on_temp=on_temp,
            ),
            daemon=True, name="WATCH",
        ),
        threading.Thread(
            target=inference_loop,
            args=(shutdown, mi, inferencer),
            kwargs=dict(hz=args.hz),
            daemon=True, name="INFER",
        ),
    ]

    for t in threads:
        t.start()
    print("[MAIN] started")

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[MAIN] stopping")
    finally:
        shutdown.set()
        for t in threads:
            t.join(timeout=2.0)
        print("[MAIN] done.")


if __name__ == "__main__":
    main()
