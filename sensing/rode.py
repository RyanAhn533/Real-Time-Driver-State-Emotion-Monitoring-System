import time
import threading
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000
BLOCKSIZE   = 8192
LATENCY     = 0.5

KEYWORDS_IN = ["Wireless GO II", "RØDE", "RODE", "Wireless GO", "GO II"]

def pick_input_device(keywords=None):
    devs = sd.query_devices()
    best = None
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) <= 0:
            continue
        name = (d.get("name") or "").lower()
        if keywords and any(k.lower() in name for k in keywords):
            best = i
            break
    if best is None:
        raise RuntimeError(f"RØDE 마이크를 찾지 못했습니다. 연결 상태를 확인하세요. (키워드: {keywords})")
    return best, sd.query_devices(best)

def run_rode(
    shutdown_event: threading.Event,
    on_audio_chunk=None,
    sample_rate: int = SAMPLE_RATE,
    blocksize: int = BLOCKSIZE,
    latency: float = LATENCY,
):
    in_idx, in_dev = pick_input_device(KEYWORDS_IN)
    in_ch = int(in_dev.get("max_input_channels", 1))
    use_ch = 2 if in_ch >= 2 else 1

    print(f"[RODE] input device: {in_idx} / {in_dev.get('name')} / channels={use_ch} / sr={sample_rate}")

    def callback(indata, frames, time_info, status):
        if status:
            # print("[RODE] status:", status)
            pass

        if use_ch == 1:
            mono = indata[:, 0]
        else:
            mono = indata.mean(axis=1)  # (frames,)

        ts_ms = int(time.time() * 1000)

        if on_audio_chunk is not None:
            on_audio_chunk(ts_ms, mono.astype(np.float32, copy=True), sr=sample_rate)

    def open_stream(idx, ch):
        s = sd.InputStream(
            device=idx,
            channels=ch,
            samplerate=sample_rate,
            blocksize=blocksize,
            dtype="float32",
            latency=latency,
            callback=callback,
        )
        s.start()
        return s

    stream = None
    try:
        stream = open_stream(in_idx, use_ch)
        while not shutdown_event.is_set():
            time.sleep(0.05)
            if stream.closed:
                print("[RODE] stream closed, reconnecting...")
                time.sleep(1.0)
                try:
                    new_idx, new_dev = pick_input_device(KEYWORDS_IN)
                    new_ch = 2 if int(new_dev.get("max_input_channels", 1)) >= 2 else 1
                    stream = open_stream(new_idx, new_ch)
                    print(f"[RODE] reconnected: {new_idx} / {new_dev.get('name')}")
                except Exception as e:
                    print(f"[RODE] reconnect failed: {e}, retrying...")
    except Exception as e:
        print(f"[RODE] stream error: {e}, retrying in 2s...")
        time.sleep(2.0)
        try:
            new_idx, new_dev = pick_input_device(KEYWORDS_IN)
            new_ch = 2 if int(new_dev.get("max_input_channels", 1)) >= 2 else 1
            print(f"[RODE] retry device: {new_idx} / {new_dev.get('name')}")
            stream = open_stream(new_idx, new_ch)
            while not shutdown_event.is_set():
                time.sleep(0.05)
        except Exception as e2:
            print(f"[RODE] retry failed: {e2}")
    finally:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        print("[RODE] stopped")
