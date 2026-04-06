#!/usr/bin/env python3
"""
K-MER Real-Time Demo v2 — Professional Dashboard
==================================================
MOTIE R&D (RS-2024-00487049) | Sejong Univ. HEART Lab

Layout:
  +-------------------------------+---------------------+
  |                               | EMOTION + CONF      |
  |    Camera Feed                | 7-class prob bars   |
  |    (1280 x 720)               | A/V/Compound        |
  |    + Face bbox + label        | Status Flags        |
  |                               | Sensor Status       |
  +-------------------------------+---------------------+
  | PPG Waveform | EDA Waveform | Temp Graph | Elapsed  |
  +-------------------------------+---------------------+

Usage:
    python kmer_demo_final.py
    python kmer_demo_final.py --no_audio
"""

import os, sys, time, threading, argparse
from collections import deque
from pathlib import Path
import numpy as np
import cv2

# ── tensorflow shim (mediapipe compat) ──
import importlib, importlib.machinery, types
def _make_fake(name):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    m.__path__ = []
    return m
for _n in ['tensorflow', 'tensorflow.tools', 'tensorflow.tools.docs']:
    if _n not in sys.modules:
        sys.modules[_n] = _make_fake(_n)
class _FDC:
    @staticmethod
    def do_not_generate_docs(x): return x
if 'tensorflow.tools.docs.doc_controls' not in sys.modules:
    sys.modules['tensorflow.tools.docs.doc_controls'] = _FDC

# ── Path Setup ──
_THIS_DIR = Path(__file__).resolve().parent
_SENSING_DIR = _THIS_DIR / "sensing"
_EMO_SYS_DIR = _THIS_DIR / "emotion_system"
_MM_DMS_DIR = _THIS_DIR / "multimodal_dms"
for _p in [str(_SENSING_DIR), str(_EMO_SYS_DIR), str(_MM_DMS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realsense_uvc import run_realsense
from sensing_main import ModelInputs, make_callbacks
from kmer_inferencer_v2 import KMERInferencer

# Gateway v2 (Motrex 13-byte protocol)
sys.path.insert(0, str(_THIS_DIR))
from gateway_v2_packet_encoder import PacketEncoder as PacketEncoderV2
import serial

# ═══════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════
KFER_LABELS = ["angry", "anxious", "happy", "hurt", "neutral", "sad", "surprised"]
PANEL_W = 420
BOTTOM_H = 130
FONT     = cv2.FONT_HERSHEY_SIMPLEX
FONT_D   = cv2.FONT_HERSHEY_DUPLEX
FONT_S   = cv2.FONT_HERSHEY_PLAIN

# Colors (BGR)
BLACK      = (0, 0, 0)
BG_DARK    = (20, 20, 25)
BG_PANEL   = (30, 30, 35)
BG_SECTION = (40, 40, 48)
WHITE      = (255, 255, 255)
GRAY       = (120, 120, 120)
LIGHT_GRAY = (180, 180, 180)
ACCENT     = (255, 180, 50)   # gold
CYAN       = (230, 200, 50)
GREEN      = (100, 220, 80)
RED        = (80, 80, 255)
ORANGE     = (60, 160, 255)
BLUE_LIGHT = (200, 160, 60)
MAGENTA    = (200, 100, 200)
TEAL       = (180, 180, 50)

EMO_COLORS = {
    "angry": RED, "anxious": ORANGE, "happy": GREEN,
    "hurt": MAGENTA, "neutral": LIGHT_GRAY, "sad": BLUE_LIGHT, "surprised": ACCENT,
}


# ═══════════════════════════════════════════════════
# Drawing Helpers
# ═══════════════════════════════════════════════════
def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


def draw_signal_graph(panel, x, y, w, h, data, color, label, unit=""):
    """Draw a real-time signal graph with grid."""
    # Background
    cv2.rectangle(panel, (x, y), (x + w, y + h), BG_SECTION, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), GRAY, 1)
    # Grid lines
    for gy in range(y + h // 4, y + h, h // 4):
        cv2.line(panel, (x, gy), (x + w, gy), (50, 50, 55), 1)
    # Label
    cv2.putText(panel, label, (x + 4, y + 12), FONT_S, 0.9, ACCENT, 1)
    if unit:
        cv2.putText(panel, unit, (x + w - 30, y + 12), FONT_S, 0.8, GRAY, 1)
    # Signal — scrolling waveform (latest data on the right)
    if len(data) > 2:
        arr = np.array(list(data), dtype=np.float64)
        if arr.ndim > 1:
            vals = arr[:, 1] if arr.shape[1] >= 2 else arr.ravel()
        else:
            vals = arr
        draw_w = w - 8
        # Take only the last draw_w points (1 pixel per sample)
        if len(vals) > draw_w:
            vals = vals[-draw_w:]
        if len(vals) > 1:
            mn, mx = np.nanmin(vals), np.nanmax(vals)
            rng = mx - mn if mx - mn > 1e-6 else 1.0
            norm = ((vals - mn) / rng * (h - 24) + 12).astype(int)
            # Draw from right edge, scrolling left
            offset_x = draw_w - len(norm)
            for i in range(1, len(norm)):
                px1 = x + 4 + offset_x + i - 1
                px2 = x + 4 + offset_x + i
                py1 = y + h - int(norm[i - 1])
                py2 = y + h - int(norm[i])
                cv2.line(panel, (px1, py1), (px2, py2), color, 1, cv2.LINE_AA)
            # Current value (right side)
            cv2.putText(panel, f"{vals[-1]:.1f}", (x + w - 50, y + h - 5),
                        FONT_S, 0.9, color, 1)


def draw_bar_fancy(panel, y, label, value, color, is_top, max_w=260):
    """Draw a single probability bar."""
    x0 = 110
    # Label
    lbl_color = WHITE if is_top else GRAY
    cv2.putText(panel, f"{label:>9s}", (12, y + 13), FONT_S, 1.05, lbl_color, 1)
    # Bar bg
    cv2.rectangle(panel, (x0, y + 1), (x0 + max_w, y + 14), (50, 50, 55), -1)
    # Bar fill
    filled = max(1, int(value * max_w))
    if is_top:
        cv2.rectangle(panel, (x0, y + 1), (x0 + filled, y + 14), color, -1)
    else:
        cv2.rectangle(panel, (x0, y + 1), (x0 + filled, y + 14), (70, 70, 75), -1)
    # Value text
    cv2.putText(panel, f"{value:.2f}", (x0 + max_w + 6, y + 13), FONT_S, 0.95, lbl_color, 1)


def draw_status_indicator(panel, x, y, label, value, ok_color, bad_color, is_ok):
    """Draw a status indicator with dot."""
    color = ok_color if is_ok else bad_color
    cv2.circle(panel, (x + 6, y + 7), 5, color, -1)
    cv2.putText(panel, f"{label}: {value}", (x + 16, y + 12), FONT_S, 1.05, color, 1)


# ═══════════════════════════════════════════════════
# Main Panel Drawing
# ═══════════════════════════════════════════════════
def draw_right_panel(result, sensors, cam_h):
    panel = np.full((cam_h, PANEL_W, 3), BG_PANEL, dtype=np.uint8)
    y = 8

    # ── Header ──
    cv2.rectangle(panel, (0, 0), (PANEL_W, 42), (25, 25, 30), -1)
    cv2.putText(panel, "K-MER", (12, 28), FONT_D, 0.75, ACCENT, 2)
    cv2.putText(panel, "Driver Monitoring System", (85, 28), FONT, 0.45, LIGHT_GRAY, 1)
    cv2.line(panel, (0, 42), (PANEL_W, 42), ACCENT, 2)
    y = 52

    # ── Emotion ──
    emo = result.get("kfer_emotion", "neutral")
    conf = result.get("kfer_confidence", 0.0)
    emo_color = EMO_COLORS.get(emo, LIGHT_GRAY)
    cv2.putText(panel, emo.upper(), (12, y + 24), FONT_D, 0.9, emo_color, 2)
    cv2.putText(panel, f"{conf:.1%}", (PANEL_W - 65, y + 24), FONT, 0.65, WHITE, 1)
    y += 36

    # ── 7-class bars ──
    probs = result.get("_kfer_probs", None)
    if probs is not None and len(probs) == 7:
        for i, (label, p) in enumerate(zip(KFER_LABELS, probs)):
            is_top = (label == emo)
            bar_color = EMO_COLORS.get(label, GRAY)
            draw_bar_fancy(panel, y, label, float(p), bar_color, is_top)
            y += 17
    else:
        y += 17 * 7
    y += 6
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (50, 50, 55), 1)
    y += 10

    # ── Arousal / Valence ──
    arousal = result.get("arousal", 0.5)
    valence = result.get("valence", None)
    compound = result.get("compound_label", "neutral")

    cv2.putText(panel, "AROUSAL", (12, y + 12), FONT_S, 0.95, GRAY, 1)
    # Arousal bar
    if arousal is not None:
        bar_x = 90
        bar_w = 200
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + 14), (50, 50, 55), -1)
        fill = int(arousal * bar_w)
        a_color = RED if arousal > 0.7 else ORANGE if arousal > 0.4 else GREEN
        cv2.rectangle(panel, (bar_x, y), (bar_x + fill, y + 14), a_color, -1)
        cv2.putText(panel, f"{arousal:.2f}", (bar_x + bar_w + 8, y + 12), FONT_S, 0.95, WHITE, 1)
    y += 20

    cv2.putText(panel, "VALENCE", (12, y + 12), FONT_S, 0.95, GRAY, 1)
    if valence is not None:
        bar_x = 90
        bar_w = 200
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + 14), (50, 50, 55), -1)
        mid = bar_x + bar_w // 2
        fill_w = int(abs(valence - 0.5) * bar_w)
        v_color = GREEN if valence > 0.5 else BLUE_LIGHT
        if valence >= 0.5:
            cv2.rectangle(panel, (mid, y), (mid + fill_w, y + 14), v_color, -1)
        else:
            cv2.rectangle(panel, (mid - fill_w, y), (mid, y + 14), v_color, -1)
        cv2.line(panel, (mid, y), (mid, y + 14), WHITE, 1)
        cv2.putText(panel, f"{valence:.2f}", (bar_x + bar_w + 8, y + 12), FONT_S, 0.95, WHITE, 1)
    else:
        cv2.putText(panel, "---", (300, y + 12), FONT_S, 0.95, GRAY, 1)
    y += 20

    cv2.putText(panel, "COMPOUND", (12, y + 12), FONT_S, 0.95, GRAY, 1)
    cv2.putText(panel, compound.upper(), (100, y + 12), FONT_S, 1.0, ACCENT, 1)
    y += 22
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (50, 50, 55), 1)
    y += 10

    # ── Status Flags ──
    perclos = result.get("perclos", 0.0)
    drowsy = result.get("drowsy", 0)
    drowsy_labels = ["Alert", "Drowsy", "Sleeping"]
    emo_negative = emo in ("angry", "anxious", "hurt", "sad")
    stress = emo_negative and (arousal is not None and arousal > 0.6)

    draw_status_indicator(panel, 12, y, "PERCLOS",
                          f"{perclos:.2f}", GREEN, RED, perclos < 0.4)
    draw_status_indicator(panel, 210, y, "DROWSY",
                          drowsy_labels[min(drowsy, 2)],
                          GREEN, RED, drowsy == 0)
    y += 22
    draw_status_indicator(panel, 12, y, "STRESS",
                          "YES" if stress else "No", GREEN, RED, not stress)
    draw_status_indicator(panel, 210, y, "NEG_EMO",
                          "YES" if emo_negative else "No", GREEN, RED, not emo_negative)
    y += 26
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (50, 50, 55), 1)
    y += 10

    # ── Sensors ──
    cv2.putText(panel, "SENSORS", (12, y + 12), FONT_S, 0.95, ACCENT, 1)
    y += 18
    for name, ok in sensors.items():
        icon = "[*]" if ok else "[ ]"
        color = GREEN if ok else (80, 80, 80)
        cv2.putText(panel, f"{icon} {name}", (16, y + 12), FONT_S, 1.0, color, 1)
        y += 16

    return panel


def draw_bottom_bar(bio_bufs, elapsed, fps, canvas_w):
    """Draw bottom bar with bio signal graphs."""
    bar = np.full((BOTTOM_H, canvas_w, 3), BG_DARK, dtype=np.uint8)
    cv2.line(bar, (0, 0), (canvas_w, 0), ACCENT, 1)

    graph_w = (canvas_w - 200) // 3
    graph_h = BOTTOM_H - 20
    margin = 8

    # PPG
    draw_signal_graph(bar, margin, 10, graph_w - margin,
                      graph_h, bio_bufs["ppg"], GREEN, "PPG", "BPM")
    # EDA
    draw_signal_graph(bar, graph_w + margin, 10, graph_w - margin,
                      graph_h, bio_bufs["eda"], TEAL, "EDA", "uS")
    # Temperature
    draw_signal_graph(bar, graph_w * 2 + margin, 10, graph_w - margin,
                      graph_h, bio_bufs["temp"], ORANGE, "TEMP", "C")

    # Right info
    info_x = graph_w * 3 + 20
    mm, ss = divmod(int(elapsed), 60)
    cv2.putText(bar, f"{mm:02d}:{ss:02d}", (info_x, 35), FONT_D, 0.8, WHITE, 1)
    cv2.putText(bar, "ELAPSED", (info_x, 50), FONT_S, 0.9, GRAY, 1)
    cv2.putText(bar, f"{fps:.1f}", (info_x, 85), FONT_D, 0.8, ACCENT, 1)
    cv2.putText(bar, "FPS", (info_x, 100), FONT_S, 0.9, GRAY, 1)
    # Project info
    cv2.putText(bar, "MOTIE RS-2024-00487049", (info_x, BOTTOM_H - 8),
                FONT_S, 0.75, (60, 60, 65), 1)

    return bar


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser("K-MER Demo")
    ap.add_argument("--no_audio", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hz", type=float, default=10.0)
    args = ap.parse_args()

    print("[MODEL] Loading...")
    kfer_ckpt = str(_EMO_SYS_DIR / "result" / "best.pth")
    kmer_ckpt = str(_EMO_SYS_DIR / "multimodal" / "checkpoints" / "ckpt_v3" / "fold_1" / "best.pth")
    inferencer = KMERInferencer(
        kfer_ckpt=kfer_ckpt, kmer_ckpt=kmer_ckpt,
        device=args.device, enable_audio_experts=not args.no_audio,
    )
    print("[MODEL] Ready")

    mi = ModelInputs(audio_sr=48000, audio_sec=2.0)
    on_frame, on_audio, on_ppg, on_eda, on_temp = make_callbacks(mi)

    # Bio display buffers
    bio_bufs = {
        "ppg": deque(maxlen=300),
        "eda": deque(maxlen=300),
        "temp": deque(maxlen=300),
    }

    shutdown = threading.Event()
    threads = []
    sensor_status = {"Camera": False, "Microphone": False, "Watch BLE": False}

    # Camera
    def cam_cb(ts, frame):
        sensor_status["Camera"] = True
        on_frame(ts, frame)
    t_cam = threading.Thread(target=run_realsense,
        kwargs=dict(shutdown_event=shutdown, on_frame=cam_cb, width=640, height=480, fps=30),
        daemon=True)
    threads.append(("Camera", t_cam))

    # Microphone
    try:
        from rode import run_rode
        def mic_cb(ts, mono, sr):
            sensor_status["Microphone"] = True
            on_audio(ts, mono, sr)
        t_mic = threading.Thread(target=run_rode,
            kwargs=dict(shutdown_event=shutdown, on_audio_chunk=mic_cb), daemon=True)
        threads.append(("Microphone", t_mic))
    except Exception as e:
        print(f"[MIC] {e}")

    # Watch
    try:
        from watch import run_watch
        def ppg_cb(ts, d1, d2):
            sensor_status["Watch BLE"] = True
            on_ppg(ts, d1, d2)
            bio_bufs["ppg"].append((ts, d1))
        def eda_cb(ts, real):
            on_eda(ts, real)
            bio_bufs["eda"].append((ts, real))
        def temp_cb(ts, skin_c):
            on_temp(ts, skin_c)
            bio_bufs["temp"].append((ts, skin_c))
        t_watch = threading.Thread(target=run_watch,
            kwargs=dict(shutdown_event=shutdown, on_ppg=ppg_cb, on_eda=eda_cb, on_temp=temp_cb),
            daemon=True)
        threads.append(("Watch BLE", t_watch))
    except Exception as e:
        print(f"[WATCH] {e}")

    for name, t in threads:
        print(f"[{name}] Starting...")
        t.start()

    # Recording
    rec_dir = _THIS_DIR / "recordings"
    rec_dir.mkdir(exist_ok=True)
    rec_path = str(rec_dir / f"demo_3modal_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    writer = None

    # ── USB Gateway ──
    pkt_encoder = PacketEncoderV2()
    ser = None
    try:
        ser = serial.Serial("/dev/ttyGS0", 115200, timeout=1)
        print("[USB] /dev/ttyGS0 opened @ 115200")
    except Exception as e:
        print(f"[USB] Serial open failed: {e} — packets will be logged only")

    print(f"\n[RUN] {args.hz} Hz | Press Q/ESC to stop\n")
    interval = 1.0 / args.hz
    t_start = time.time()
    fps_n, fps_t0, fps_val = 0, time.time(), 0.0
    n_infer = 0

    try:
        while not shutdown.is_set():
            t0 = time.time()
            fts, frame = mi.frame_main.get()
            ats, audio_snap = mi.audio.snapshot()
            ppg_list, eda_list, temp_list = mi.bio.snapshot()

            if frame is None:
                time.sleep(0.01)
                continue

            # Simulated bio signals when watch not connected
            if not sensor_status.get("Watch BLE", False):
                t_now = time.time() - t_start
                # PPG: ~70 BPM sine wave with noise
                ppg_val = 800 + 200 * np.sin(2 * np.pi * 1.17 * t_now) + np.random.normal(0, 15)
                bio_bufs["ppg"].append((int(t_now * 1000), ppg_val))
                # EDA: slow drift with occasional peaks
                eda_val = 2.5 + 0.8 * np.sin(2 * np.pi * 0.05 * t_now) + np.random.normal(0, 0.1)
                bio_bufs["eda"].append((int(t_now * 1000), eda_val))
                # Temp: stable ~36.5 with small fluctuation
                temp_val = 36.5 + 0.3 * np.sin(2 * np.pi * 0.02 * t_now) + np.random.normal(0, 0.05)
                bio_bufs["temp"].append((int(t_now * 1000), temp_val))

            result = inferencer.forward(
                frame_bgr=frame, audio_1d=audio_snap,
                ppg=ppg_list, eda=eda_list, temp=temp_list,
            )
            n_infer += 1

            # ── USB Packet (13-byte Motrex protocol) ──
            kfer_emo = result.get("kfer_emotion", "neutral")
            kfer_id = KFER_LABELS.index(kfer_emo) if kfer_emo in KFER_LABELS else 4
            pkt = pkt_encoder.encode(
                kfer_emotion_id=kfer_id,
                emotion_confidence=result.get("kfer_confidence", 0.5),
                arousal=result.get("arousal", None),
                perclos=result.get("perclos", 0.0),
            )
            if ser is not None:
                try:
                    ser.write(pkt)
                except Exception:
                    pass

            # Draw on camera frame
            display = frame.copy()
            bbox = result.get("bbox")
            if bbox and result.get("face_detected", False):
                bx1, by1, bx2, by2 = bbox
                emo = result.get("kfer_emotion", "neutral")
                conf = result.get("kfer_confidence", 0.0)
                emo_color = EMO_COLORS.get(emo, WHITE)
                # Thick bbox with colored top
                cv2.rectangle(display, (bx1, by1), (bx2, by2), emo_color, 2)
                # Label background
                tw = cv2.getTextSize(f"{emo} {conf:.0%}", FONT, 0.7, 2)[0][0]
                cv2.rectangle(display, (bx1, by1 - 28), (bx1 + tw + 10, by1), emo_color, -1)
                cv2.putText(display, f"{emo} {conf:.0%}",
                            (bx1 + 5, by1 - 8), FONT, 0.7, BLACK, 2)

            # FPS
            fps_n += 1
            if fps_n % 10 == 0:
                now = time.time()
                fps_val = 10.0 / (now - fps_t0 + 1e-9)
                fps_t0 = now

            # Compose: camera + right panel
            cam_h = display.shape[0]
            right = draw_right_panel(result, sensor_status, cam_h)
            top = np.hstack([display, right])

            # Bottom bar with bio graphs
            elapsed = time.time() - t_start
            bottom = draw_bottom_bar(bio_bufs, elapsed, fps_val, top.shape[1])
            canvas = np.vstack([top, bottom])

            # Record
            if writer is None:
                ch, cw = canvas.shape[:2]
                writer = cv2.VideoWriter(rec_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (cw, ch))
            writer.write(canvas)

            cv2.imshow("K-MER Real-Time Monitor", canvas)
            if (cv2.waitKey(1) & 0xFF) in [27, ord('q')]:
                break

            time.sleep(max(0.0, interval - (time.time() - t0)))

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")

    shutdown.set()
    for _, t in threads:
        t.join(timeout=5)
    if writer:
        writer.release()
    if ser:
        ser.close()
    cv2.destroyAllWindows()

    dur = time.time() - t_start
    print(f"\nSaved: {rec_path}")
    print(f"Inferences: {n_infer}, Duration: {dur:.1f}s")
    print("=== Done ===")


if __name__ == "__main__":
    main()
