"""
RODE Wireless GO II 마이크 센싱 모듈
====================================
pipewire 경유 48kHz stereo → mono float32 스트리밍.

Usage:
    python rode.py
"""

import time
import threading
import numpy as np
import sounddevice as sd

# ── 상수 ──
SAMPLE_RATE = 48000
CHUNK_SEC = 0.5        # 0.5초 블록 단위로 콜백
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)

KEYWORDS_IN = ["wireless", "rode", "røde", "go ii"]


def pick_input_device():
    """RODE 마이크 디바이스 인덱스 검색 (pipewire 경유 포함)."""
    rode_dev = None
    pipewire_dev = None

    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) <= 0:
            continue
        name_lower = d["name"].lower()
        if any(k in name_lower for k in KEYWORDS_IN):
            rode_dev = (i, d)
        if "pipewire" in name_lower:
            pipewire_dev = (i, d)

    # pipewire가 RODE를 잡고 있으면 pipewire를 통해 접근
    if pipewire_dev:
        return pipewire_dev
    if rode_dev:
        return rode_dev
    raise RuntimeError("RØDE 마이크를 찾지 못했습니다.")


def run_rode(
    shutdown_event: threading.Event,
    on_audio_chunk=None,   # (ts_ms: int, mono: np.ndarray, sr: int)
    sample_rate: int = SAMPLE_RATE,
    chunk_sec: float = CHUNK_SEC,
):
    """
    RODE 마이크 스트리밍 메인 루프 (polling 방식).

    Callback:
      on_audio_chunk(ts_ms, mono, sr)
        - ts_ms: 타임스탬프 (ms)
        - mono: float32 numpy array
        - sr: 샘플레이트
    """
    idx, info = pick_input_device()
    channels = min(int(info.get("max_input_channels", 2)), 2)
    chunk_samples = int(sample_rate * chunk_sec)

    print(
        f"[RODE] device: {idx} / {info['name']}"
        f" / channels={channels} / sr={sample_rate}"
        f" / chunk={chunk_sec}s ({chunk_samples} samples)"
    )

    while not shutdown_event.is_set():
        try:
            # blocking 녹음 (chunk 단위)
            audio = sd.rec(
                chunk_samples,
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
                device=idx,
            )
            sd.wait()

            if shutdown_event.is_set():
                break

            # stereo → mono
            if audio.shape[1] > 1:
                mono = audio.mean(axis=1)
            else:
                mono = audio[:, 0]

            ts_ms = int(time.time() * 1000)

            if on_audio_chunk is not None:
                on_audio_chunk(ts_ms, mono.astype(np.float32), sample_rate)

        except Exception as e:
            if shutdown_event.is_set():
                break
            print(f"[RODE] error: {e}, retrying in 2s...")
            time.sleep(2.0)

    print("[RODE] stopped")


# ── Self-test ──
if __name__ == "__main__":
    shutdown = threading.Event()
    chunk_count = [0]
    max_amp = [0.0]

    def _on_chunk(ts_ms, mono, sr):
        chunk_count[0] += 1
        amp = float(np.abs(mono).max())
        max_amp[0] = max(max_amp[0], amp)
        rms = float(np.sqrt(np.mean(mono ** 2)))
        print(f"  Audio #{chunk_count[0]}: ts={ts_ms}, samples={len(mono)}, "
              f"max={amp:.4f}, rms={rms:.6f}")

    print("=== RODE Mic Test (5초) ===")
    t = threading.Thread(
        target=run_rode,
        kwargs=dict(shutdown_event=shutdown, on_audio_chunk=_on_chunk),
        daemon=True,
    )
    t.start()

    try:
        time.sleep(5)
    except KeyboardInterrupt:
        pass

    shutdown.set()
    t.join(timeout=3)
    print(f"\nTotal chunks: {chunk_count[0]}, max amplitude: {max_amp[0]:.4f}")
    print(f"Signal present: {max_amp[0] > 0.001}")
