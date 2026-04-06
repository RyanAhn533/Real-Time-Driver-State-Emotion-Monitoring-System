#!/usr/bin/env python3
"""
K-MER Real-Time Driver Emotion Monitoring System
==================================================
Ministry of Trade, Industry and Energy (MOTIE)
National R&D Project RS-2024-00487049

Sejong University HEART Lab | On-Device Multimodal DMS

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
_PIPELINE_DIR = _SENSING_DIR / "pipeline"
for _p in [str(_SENSING_DIR), str(_EMO_SYS_DIR), str(_MM_DMS_DIR), str(_PIPELINE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realsense_uvc import run_realsense
from sensing_main import ModelInputs, make_callbacks
from kmer_inferencer_v2 import KMERInferencer
from temporal_smoother import MultimodalTemporalSmoother

# Gateway v2 (Motrex 13-byte protocol)
sys.path.insert(0, str(_THIS_DIR))
from gateway_v2_packet_encoder import PacketEncoder as PacketEncoderV2
import serial

# ═══════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════
KFER_LABELS = ["angry", "anxious", "happy", "hurt", "neutral", "sad", "surprised"]
KFER_LABELS_KR = ["분노", "불안", "기쁨", "상처", "중립", "슬픔", "놀람"]
PANEL_W = 460
BOTTOM_H = 140
FONT     = cv2.FONT_HERSHEY_SIMPLEX
FONT_D   = cv2.FONT_HERSHEY_DUPLEX
FONT_S   = cv2.FONT_HERSHEY_PLAIN

# ── Official Color Palette (Government/MOTIE style) ──
BLACK       = (0, 0, 0)
WHITE       = (255, 255, 255)
BG_NAVY     = (45, 35, 25)       # deep navy background
BG_PANEL    = (50, 42, 32)       # panel background
BG_SECTION  = (60, 52, 42)       # section background
BG_HEADER   = (120, 70, 20)      # official blue header
BORDER_GOLD = (80, 180, 220)     # gold/amber accent
ACCENT_BLUE = (200, 160, 40)     # official blue
ACCENT_GOLD = (60, 175, 215)     # warm gold
GRAY        = (140, 140, 140)
LIGHT_GRAY  = (190, 190, 190)
TEXT_DIM    = (120, 120, 130)
DIVIDER     = (80, 70, 60)

# Status colors
GREEN       = (80, 200, 80)
RED         = (70, 70, 230)
ORANGE      = (50, 150, 240)
YELLOW      = (60, 210, 240)
CYAN        = (210, 190, 60)
TEAL        = (170, 180, 60)
MAGENTA     = (180, 90, 180)
BLUE_LIGHT  = (190, 155, 60)

EMO_COLORS = {
    "angry": (80, 80, 235),
    "anxious": (50, 140, 230),
    "happy": (80, 200, 80),
    "hurt": (180, 90, 180),
    "neutral": (180, 180, 180),
    "sad": (190, 155, 60),
    "surprised": (60, 200, 230),
}

DROWSY_LABELS = ["정상", "졸음주의", "졸음위험"]
DROWSY_COLORS = [GREEN, YELLOW, RED]

# FaceMesh visualization — key contour indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
NOSE_BRIDGE = [168, 6, 197, 195, 5]
LEFT_EYEBROW = [276, 283, 282, 295, 285]
RIGHT_EYEBROW = [46, 53, 52, 65, 55]

MESH_LINES = [
    (FACE_OVAL, (80, 70, 60)),
    (LEFT_EYE_CONTOUR, CYAN),
    (RIGHT_EYE_CONTOUR, CYAN),
    (LIPS_OUTER, (120, 100, 180)),
    (NOSE_BRIDGE, (80, 80, 90)),
    (LEFT_EYEBROW, (100, 90, 70)),
    (RIGHT_EYEBROW, (100, 90, 70)),
]


def draw_face_mesh(frame, landmarks):
    """Draw subtle FaceMesh contour lines on camera frame."""
    if landmarks is None or len(landmarks) < 468:
        return
    for indices, color in MESH_LINES:
        pts = []
        for idx in indices:
            if idx < len(landmarks):
                x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], color, 1, cv2.LINE_AA)
        if len(pts) > 2 and indices in (FACE_OVAL, LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR, LIPS_OUTER):
            cv2.line(frame, pts[-1], pts[0], color, 1, cv2.LINE_AA)


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
    cv2.rectangle(panel, (x, y), (x + w, y + h), BG_SECTION, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), DIVIDER, 1)
    for gy in range(y + h // 4, y + h, h // 4):
        cv2.line(panel, (x + 4, gy), (x + w - 4, gy), (55, 50, 45), 1)
    cv2.putText(panel, label, (x + 6, y + 14), FONT_S, 0.95, ACCENT_GOLD, 1)
    if unit:
        cv2.putText(panel, unit, (x + w - 32, y + 14), FONT_S, 0.8, TEXT_DIM, 1)
    if len(data) > 2:
        arr = np.array(list(data), dtype=np.float64)
        if arr.ndim > 1:
            vals = arr[:, 1] if arr.shape[1] >= 2 else arr.ravel()
        else:
            vals = arr
        draw_w = w - 10
        if len(vals) > draw_w:
            vals = vals[-draw_w:]
        if len(vals) > 1:
            mn, mx = np.nanmin(vals), np.nanmax(vals)
            rng = mx - mn if mx - mn > 1e-6 else 1.0
            norm = ((vals - mn) / rng * (h - 28) + 14).astype(int)
            offset_x = draw_w - len(norm)
            for i in range(1, len(norm)):
                px1 = x + 5 + offset_x + i - 1
                px2 = x + 5 + offset_x + i
                py1 = y + h - int(norm[i - 1])
                py2 = y + h - int(norm[i])
                cv2.line(panel, (px1, py1), (px2, py2), color, 1, cv2.LINE_AA)
            cv2.putText(panel, f"{vals[-1]:.1f}", (x + w - 55, y + h - 6),
                        FONT_S, 0.9, color, 1)


def draw_progress_bar(panel, x, y, w, h, value, color, bg_color=None):
    """Draw a clean progress bar."""
    if bg_color is None:
        bg_color = BG_SECTION
    cv2.rectangle(panel, (x, y), (x + w, y + h), bg_color, -1)
    filled = max(1, int(value * w))
    cv2.rectangle(panel, (x, y), (x + filled, y + h), color, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), DIVIDER, 1)


def draw_section_header(panel, y, title, width):
    """Draw a section divider with title."""
    cv2.line(panel, (10, y), (width - 10, y), DIVIDER, 1)
    cv2.putText(panel, title, (14, y + 16), FONT, 0.42, ACCENT_GOLD, 1)
    return y + 22


# ═══════════════════════════════════════════════════
# Header Bar (Top)
# ═══════════════════════════════════════════════════
HEADER_H = 48

def draw_header_bar(width, elapsed, fps):
    """Official project header bar."""
    bar = np.full((HEADER_H, width, 3), BG_HEADER, dtype=np.uint8)

    # Bottom gold line
    cv2.line(bar, (0, HEADER_H - 2), (width, HEADER_H - 2), ACCENT_GOLD, 2)

    # Left: Project title
    cv2.putText(bar, "K-MER", (16, 32), FONT_D, 0.85, WHITE, 2)
    cv2.putText(bar, "Real-Time Driver Emotion Monitoring System",
                (105, 22), FONT, 0.45, (220, 220, 225), 1)
    cv2.putText(bar, "On-Device Multimodal DMS  |  Jetson AGX Thor",
                (105, 38), FONT, 0.38, (170, 170, 180), 1)

    # Right: Time & FPS
    mm, ss = divmod(int(elapsed), 60)
    hh, mm = divmod(mm, 60)
    time_str = f"{hh:02d}:{mm:02d}:{ss:02d}"
    cv2.putText(bar, time_str, (width - 155, 22), FONT, 0.5, WHITE, 1)
    cv2.putText(bar, f"{fps:.1f} FPS", (width - 155, 40), FONT, 0.45, ACCENT_GOLD, 1)

    return bar


# ═══════════════════════════════════════════════════
# Right Panel
# ═══════════════════════════════════════════════════
def draw_right_panel(result, sensors, cam_h):
    panel = np.full((cam_h, PANEL_W, 3), BG_PANEL, dtype=np.uint8)

    # Left border accent line
    cv2.line(panel, (0, 0), (0, cam_h), ACCENT_GOLD, 2)

    y = 10

    # ── Primary Emotion Display ──
    emo = result.get("display_emotion", result.get("kfer_emotion", "neutral"))
    conf = result.get("display_confidence", result.get("kfer_confidence", 0.0))
    emo_idx = KFER_LABELS.index(emo) if emo in KFER_LABELS else 4
    emo_kr = KFER_LABELS_KR[emo_idx]
    emo_color = EMO_COLORS.get(emo, LIGHT_GRAY)
    stability = result.get("emotion_stability", 0.0)

    # Emotion box
    cv2.rectangle(panel, (10, y), (PANEL_W - 10, y + 62), BG_SECTION, -1)
    cv2.rectangle(panel, (10, y), (PANEL_W - 10, y + 62), emo_color, 2)

    cv2.putText(panel, emo.upper(), (20, y + 28), FONT_D, 0.85, emo_color, 2)
    cv2.putText(panel, emo_kr, (20, y + 52), FONT, 0.55, LIGHT_GRAY, 1)
    cv2.putText(panel, f"{conf:.1%}", (PANEL_W - 80, y + 28), FONT, 0.6, WHITE, 1)

    # Stability indicator
    if stability > 0:
        stab_text = f"Stability {stability:.0%}"
        stab_color = GREEN if stability > 0.6 else YELLOW if stability > 0.3 else RED
        cv2.putText(panel, stab_text, (PANEL_W - 130, y + 52), FONT, 0.38, stab_color, 1)
    y += 70

    # ── 7-Class Probability Bars ──
    y = draw_section_header(panel, y, "EMOTION DISTRIBUTION", PANEL_W)
    probs = result.get("_kfer_probs", None)
    if probs is not None and len(probs) == 7:
        bar_max_w = 240
        for i, (label, label_kr, p) in enumerate(zip(KFER_LABELS, KFER_LABELS_KR, probs)):
            is_top = (label == emo)
            bar_color = EMO_COLORS.get(label, GRAY) if is_top else (70, 65, 60)
            lbl_color = WHITE if is_top else GRAY

            cv2.putText(panel, f"{label_kr}", (16, y + 13), FONT, 0.4, lbl_color, 1)
            cv2.putText(panel, f"{label:>9s}", (55, y + 13), FONT_S, 0.9, lbl_color, 1)

            bx = 140
            cv2.rectangle(panel, (bx, y + 2), (bx + bar_max_w, y + 14), (50, 45, 40), -1)
            filled = max(1, int(float(p) * bar_max_w))
            cv2.rectangle(panel, (bx, y + 2), (bx + filled, y + 14), bar_color, -1)

            cv2.putText(panel, f"{float(p):.2f}", (bx + bar_max_w + 8, y + 13),
                        FONT_S, 0.9, lbl_color, 1)
            y += 18
    else:
        y += 18 * 7
    y += 4

    # ── Arousal / Valence ──
    y = draw_section_header(panel, y, "PHYSIOLOGICAL STATE", PANEL_W)

    arousal = result.get("display_arousal", result.get("arousal", 0.5))
    valence = result.get("display_valence", result.get("valence", None))
    compound = result.get("compound_label", "neutral")

    # Arousal bar
    cv2.putText(panel, "AROUSAL", (16, y + 12), FONT, 0.38, GRAY, 1)
    if arousal is not None:
        bar_x, bar_w = 100, 220
        a_color = RED if arousal > 0.7 else ORANGE if arousal > 0.4 else GREEN
        draw_progress_bar(panel, bar_x, y, bar_w, 14, arousal, a_color)
        cv2.putText(panel, f"{arousal:.2f}", (bar_x + bar_w + 8, y + 12), FONT_S, 0.9, WHITE, 1)
    y += 22

    # Valence bar (bipolar)
    cv2.putText(panel, "VALENCE", (16, y + 12), FONT, 0.38, GRAY, 1)
    if valence is not None:
        bar_x, bar_w = 100, 220
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + 14), BG_SECTION, -1)
        mid = bar_x + bar_w // 2
        fill_w = int(abs(valence - 0.5) * bar_w)
        v_color = GREEN if valence > 0.5 else BLUE_LIGHT
        if valence >= 0.5:
            cv2.rectangle(panel, (mid, y), (mid + fill_w, y + 14), v_color, -1)
        else:
            cv2.rectangle(panel, (mid - fill_w, y), (mid, y + 14), v_color, -1)
        cv2.line(panel, (mid, y), (mid, y + 14), WHITE, 1)
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + 14), DIVIDER, 1)
        cv2.putText(panel, f"{valence:.2f}", (bar_x + bar_w + 8, y + 12), FONT_S, 0.9, WHITE, 1)
    else:
        cv2.putText(panel, "---", (330, y + 12), FONT_S, 0.95, GRAY, 1)
    y += 22

    # Compound
    cv2.putText(panel, "COMPOUND", (16, y + 12), FONT, 0.38, GRAY, 1)
    cv2.putText(panel, compound.upper(), (110, y + 12), FONT, 0.42, ACCENT_GOLD, 1)
    y += 22

    # ── Driver Status ──
    y = draw_section_header(panel, y, "DRIVER STATUS", PANEL_W)

    perclos = result.get("perclos", 0.0)
    drowsy = result.get("display_drowsy", result.get("drowsy", 0))
    drowsy = min(int(drowsy), 2)
    emo_negative = emo in ("angry", "anxious", "hurt", "sad")
    stress = emo_negative and (arousal is not None and arousal > 0.6)

    # PERCLOS
    cv2.putText(panel, "PERCLOS", (16, y + 12), FONT, 0.38, GRAY, 1)
    perc_color = GREEN if perclos < 0.3 else YELLOW if perclos < 0.5 else RED
    draw_progress_bar(panel, 100, y, 140, 14, min(perclos, 1.0), perc_color)
    cv2.putText(panel, f"{perclos:.2f}", (248, y + 12), FONT_S, 0.9, perc_color, 1)
    y += 20

    # Drowsiness level
    cv2.putText(panel, "DROWSY", (16, y + 12), FONT, 0.38, GRAY, 1)
    d_color = DROWSY_COLORS[drowsy]
    cv2.circle(panel, (108, y + 7), 5, d_color, -1)
    cv2.putText(panel, DROWSY_LABELS[drowsy], (120, y + 12), FONT, 0.42, d_color, 1)
    y += 20

    # Stress
    cv2.putText(panel, "STRESS", (16, y + 12), FONT, 0.38, GRAY, 1)
    s_color = RED if stress else GREEN
    cv2.circle(panel, (108, y + 7), 5, s_color, -1)
    cv2.putText(panel, "DETECTED" if stress else "NORMAL", (120, y + 12), FONT, 0.42, s_color, 1)

    # Negative emotion
    cv2.putText(panel, "NEG EMO", (250, y + 12), FONT, 0.38, GRAY, 1)
    n_color = RED if emo_negative else GREEN
    cv2.circle(panel, (340, y + 7), 5, n_color, -1)
    cv2.putText(panel, "YES" if emo_negative else "NO", (352, y + 12), FONT, 0.42, n_color, 1)
    y += 26

    # ── Sensor Status ──
    y = draw_section_header(panel, y, "SENSOR STATUS", PANEL_W)
    for name, ok in sensors.items():
        dot_color = GREEN if ok else (60, 55, 50)
        text_color = WHITE if ok else (80, 80, 80)
        cv2.circle(panel, (22, y + 7), 4, dot_color, -1)
        if ok:
            cv2.circle(panel, (22, y + 7), 6, dot_color, 1)
        cv2.putText(panel, name, (34, y + 12), FONT, 0.4, text_color, 1)
        status_text = "CONNECTED" if ok else "DISCONNECTED"
        cv2.putText(panel, status_text, (160, y + 12), FONT, 0.35, dot_color, 1)
        y += 18

    return panel


# ═══════════════════════════════════════════════════
# Bottom Bar
# ═══════════════════════════════════════════════════
def draw_bottom_bar(bio_bufs, canvas_w, watch_connected):
    """Draw bottom bar with bio signal graphs."""
    bar = np.full((BOTTOM_H, canvas_w, 3), BG_NAVY, dtype=np.uint8)
    cv2.line(bar, (0, 0), (canvas_w, 0), ACCENT_GOLD, 1)

    graph_w = (canvas_w - 200) // 3
    graph_h = BOTTOM_H - 24
    margin = 10

    # Simulated data label
    if not watch_connected:
        cv2.putText(bar, "SIMULATED", (canvas_w - 190, 16), FONT, 0.35, TEXT_DIM, 1)

    # PPG
    draw_signal_graph(bar, margin, 14, graph_w - margin,
                      graph_h, bio_bufs["ppg"], GREEN, "PPG", "BPM")
    # EDA
    draw_signal_graph(bar, graph_w + margin, 14, graph_w - margin,
                      graph_h, bio_bufs["eda"], TEAL, "EDA", "uS")
    # Temperature
    draw_signal_graph(bar, graph_w * 2 + margin, 14, graph_w - margin,
                      graph_h, bio_bufs["temp"], ORANGE, "TEMP", "C")

    # Right info
    info_x = graph_w * 3 + 20
    cv2.putText(bar, "MOTIE R&D Project", (info_x, 30), FONT, 0.35, TEXT_DIM, 1)
    cv2.putText(bar, "RS-2024-00487049", (info_x, 50), FONT, 0.38, ACCENT_GOLD, 1)
    cv2.putText(bar, "Sejong Univ.", (info_x, 75), FONT, 0.35, TEXT_DIM, 1)
    cv2.putText(bar, "HEART Lab", (info_x, 95), FONT, 0.38, LIGHT_GRAY, 1)
    cv2.putText(bar, "Jetson AGX Thor", (info_x, 120), FONT, 0.32, TEXT_DIM, 1)

    return bar


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser("K-MER Demo")
    ap.add_argument("--no_audio", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--smooth_window", type=int, default=9,
                    help="Emotion majority vote window size (default: 9)")
    ap.add_argument("--ema_alpha", type=float, default=0.2,
                    help="EMA alpha for arousal/valence smoothing (default: 0.2)")
    args = ap.parse_args()

    print("[MODEL] Loading...")
    kfer_ckpt = str(_EMO_SYS_DIR / "result" / "best.pth")
    kmer_ckpt = str(_EMO_SYS_DIR / "multimodal" / "checkpoints" / "ckpt_v3" / "fold_1" / "best.pth")
    inferencer = KMERInferencer(
        kfer_ckpt=kfer_ckpt, kmer_ckpt=kmer_ckpt,
        device=args.device, enable_audio_experts=not args.no_audio,
    )
    print("[MODEL] Ready")

    # Temporal Smoother
    smoother = MultimodalTemporalSmoother(
        emotion_window=args.smooth_window,
        drowsy_window=5,
        compound_window=args.smooth_window,
        ema_alpha=args.ema_alpha,
    )
    print(f"[SMOOTH] emotion_window={args.smooth_window}, ema_alpha={args.ema_alpha}")

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
        print(f"[USB] Serial open failed: {e} -- packets will be logged only")

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
                ppg_val = 800 + 200 * np.sin(2 * np.pi * 1.17 * t_now) + np.random.normal(0, 15)
                bio_bufs["ppg"].append((int(t_now * 1000), ppg_val))
                eda_val = 2.5 + 0.8 * np.sin(2 * np.pi * 0.05 * t_now) + np.random.normal(0, 0.1)
                bio_bufs["eda"].append((int(t_now * 1000), eda_val))
                temp_val = 36.5 + 0.3 * np.sin(2 * np.pi * 0.02 * t_now) + np.random.normal(0, 0.05)
                bio_bufs["temp"].append((int(t_now * 1000), temp_val))

            result = inferencer.forward(
                frame_bgr=frame, audio_1d=audio_snap,
                ppg=ppg_list, eda=eda_list, temp=temp_list,
            )
            n_infer += 1

            # ── Temporal Smoothing ──
            smoothed = smoother.smooth(result)
            display_emo = smoothed.get("smoothed_kfer_emotion", result.get("kfer_emotion", "neutral"))
            display_arousal = smoothed.get("smoothed_arousal", result.get("arousal", 0.5))
            display_valence = smoothed.get("smoothed_valence", result.get("valence", None))
            display_drowsy = smoothed.get("smoothed_drowsy", result.get("drowsy", 0))
            emotion_stability = smoothed.get("smoothed_emotion_confidence", 0.0)

            # Inject smoothed values for panel display
            result["display_emotion"] = display_emo
            result["display_confidence"] = result.get("kfer_confidence", 0.0)
            result["display_arousal"] = display_arousal
            result["display_valence"] = display_valence
            result["display_drowsy"] = display_drowsy
            result["emotion_stability"] = emotion_stability

            # ── USB Packet (13-byte Motrex protocol) ──
            # Send smoothed emotion for stable output
            kfer_id = KFER_LABELS.index(display_emo) if display_emo in KFER_LABELS else 4
            pkt = pkt_encoder.encode(
                kfer_emotion_id=kfer_id,
                emotion_confidence=result.get("kfer_confidence", 0.5),
                arousal=display_arousal,
                perclos=result.get("perclos", 0.0),
            )
            if ser is not None:
                try:
                    ser.write(pkt)
                except Exception:
                    pass

            # Draw on camera frame
            display = frame.copy()

            # FaceMesh contours
            draw_face_mesh(display, result.get("_landmarks"))

            bbox = result.get("bbox")
            if bbox and result.get("face_detected", False):
                bx1, by1, bx2, by2 = bbox
                emo_color = EMO_COLORS.get(display_emo, WHITE)

                # Clean bbox
                cv2.rectangle(display, (bx1, by1), (bx2, by2), emo_color, 2)

                # Label with background
                label_text = f"{display_emo.upper()} {result.get('kfer_confidence', 0.0):.0%}"
                tw = cv2.getTextSize(label_text, FONT, 0.6, 1)[0][0]
                cv2.rectangle(display, (bx1, by1 - 24), (bx1 + tw + 10, by1), emo_color, -1)
                cv2.putText(display, label_text,
                            (bx1 + 5, by1 - 7), FONT, 0.6, BLACK, 1)

            # FPS
            fps_n += 1
            if fps_n % 10 == 0:
                now = time.time()
                fps_val = 10.0 / (now - fps_t0 + 1e-9)
                fps_t0 = now

            elapsed = time.time() - t_start

            # Compose: header + camera + right panel + bottom
            cam_h = display.shape[0]
            cam_w = display.shape[1]

            right = draw_right_panel(result, sensor_status, cam_h)
            main_row = np.hstack([display, right])
            total_w = main_row.shape[1]

            header = draw_header_bar(total_w, elapsed, fps_val)
            bottom = draw_bottom_bar(bio_bufs, total_w,
                                     sensor_status.get("Watch BLE", False))
            canvas = np.vstack([header, main_row, bottom])

            # Record
            if writer is None:
                ch, cw = canvas.shape[:2]
                writer = cv2.VideoWriter(rec_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (cw, ch))
            writer.write(canvas)

            cv2.imshow("K-MER Driver Monitoring System", canvas)
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
