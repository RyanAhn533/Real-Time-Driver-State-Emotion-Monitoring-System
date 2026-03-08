#!/usr/bin/env python3
"""
K-MER 시스템 아키텍처 다이어그램 생성
산업부 국책과제 실증 보고서용
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
from matplotlib.font_manager import FontProperties
import numpy as np

# ── Font Setup ──
FONT_KR = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_KR_BOLD = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
FONT_KR_MED = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
FONT_EN = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

# ── Color Palette (산업부/정부과제 느낌 — 깔끔한 블루톤) ──
C_BG = "#FFFFFF"
C_HEADER_BG = "#1B3A5C"       # 진한 남색 (산업부 색감)
C_HEADER_TEXT = "#FFFFFF"
C_SENSOR_BG = "#E8F4FD"       # 연한 하늘
C_SENSOR_BORDER = "#4A90D9"
C_EXPERT_BG = "#FFF3E0"       # 연한 오렌지
C_EXPERT_BORDER = "#E67E22"
C_FUSION_BG = "#E8F5E9"       # 연한 녹색
C_FUSION_BORDER = "#27AE60"
C_POST_BG = "#F3E5F5"         # 연한 보라
C_POST_BORDER = "#8E24AA"
C_GATEWAY_BG = "#FFEBEE"      # 연한 빨강
C_GATEWAY_BORDER = "#C62828"
C_ARROW = "#2C3E50"
C_SECTION_LABEL = "#1B3A5C"
C_SUBTEXT = "#555555"
C_ACCENT_BLUE = "#2980B9"
C_ACCENT_GREEN = "#27AE60"
C_PERF_BG = "#F0F7FF"
C_PERF_BORDER = "#3498DB"
C_TOKEN_FACE = "#BBDEFB"
C_TOKEN_AUDIO = "#FFE0B2"
C_TOKEN_BIO = "#C8E6C9"
C_TOKEN_AUX = "#E1BEE7"
C_TOKEN_META = "#CFD8DC"


def draw_rounded_box(ax, x, y, w, h, color, border_color, text="",
                     fontsize=10, text_color="#333333", alpha=1.0,
                     radius=0.015, linewidth=1.5, bold=False,
                     text_y_offset=0, sub_text="", sub_fontsize=7.5):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor=border_color,
        linewidth=linewidth, alpha=alpha,
        transform=ax.transData, zorder=2
    )
    ax.add_patch(box)

    fp = FONT_KR_BOLD if bold else FONT_KR
    if text:
        ty = y + h / 2 + text_y_offset
        if sub_text:
            ty = y + h / 2 + h * 0.13 + text_y_offset
        ax.text(x + w / 2, ty, text, ha="center", va="center",
                fontsize=fontsize, color=text_color,
                fontproperties=fp, zorder=3)
    if sub_text:
        ax.text(x + w / 2, y + h / 2 - h * 0.18 + text_y_offset, sub_text,
                ha="center", va="center", fontsize=sub_fontsize,
                color=C_SUBTEXT, fontproperties=FONT_KR, zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, linewidth=1.8,
               head_width=0.006, style="-|>", zorder=1):
    """Draw an arrow between two points."""
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=style,
                    color=color,
                    linewidth=linewidth,
                    mutation_scale=12,
                ),
                zorder=zorder)


def draw_section_band(ax, y, h, color, alpha=0.08):
    """Draw a horizontal section background band."""
    ax.add_patch(plt.Rectangle(
        (0.02, y), 0.96, h,
        facecolor=color, edgecolor="none", alpha=alpha, zorder=0
    ))


def draw_section_label(ax, x, y, text, color=C_SECTION_LABEL, fontsize=9):
    """Draw a section label on the left side."""
    ax.text(x, y, text, ha="left", va="center", fontsize=fontsize,
            fontproperties=FONT_KR_BOLD, color=color, zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, alpha=0.9, linewidth=0.8))


def draw_dashed_line(ax, y, color="#CCCCCC", linewidth=0.5):
    ax.plot([0.04, 0.96], [y, y], color=color, linewidth=linewidth,
            linestyle="--", zorder=0)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(20, 26))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)

    # ══════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════
    header = FancyBboxPatch(
        (0.02, 0.955), 0.96, 0.038,
        boxstyle="round,pad=0,rounding_size=0.008",
        facecolor=C_HEADER_BG, edgecolor=C_HEADER_BG,
        linewidth=0, zorder=2
    )
    ax.add_patch(header)
    ax.text(0.5, 0.974, "K-MER: Korean Multimodal Emotion Recognition System Architecture",
            ha="center", va="center", fontsize=16, color=C_HEADER_TEXT,
            fontproperties=FONT_KR_BOLD, zorder=3)

    # Sub header
    ax.text(0.5, 0.948, "산업부 전자부품산업기술개발 국책과제  |  실시간 운전자 감정·상태 모니터링  |  NVIDIA Jetson Orin",
            ha="center", va="center", fontsize=8.5, color="#888888",
            fontproperties=FONT_KR, zorder=3)

    # ══════════════════════════════════════════════
    # LAYER 1: SENSOR INPUT
    # ══════════════════════════════════════════════
    sy = 0.895
    sh = 0.042
    draw_section_band(ax, sy - 0.005, sh + 0.01, C_SENSOR_BORDER)
    draw_section_label(ax, 0.035, sy + sh / 2, "① Input\n   Sensors", C_SENSOR_BORDER, 8)

    # Camera
    draw_rounded_box(ax, 0.15, sy, 0.18, sh, C_SENSOR_BG, C_SENSOR_BORDER,
                     "RGB Camera", 10, bold=True,
                     sub_text="Intel RealSense D435  |  1280×720 @30fps")
    # E4
    draw_rounded_box(ax, 0.40, sy, 0.18, sh, C_SENSOR_BG, C_SENSOR_BORDER,
                     "Physiological Sensor", 10, bold=True,
                     sub_text="E4 Wristband  |  BVP/EDA/HR/TEMP")
    # Mic
    draw_rounded_box(ax, 0.65, sy, 0.18, sh, C_SENSOR_BG, C_SENSOR_BORDER,
                     "Audio Sensor", 10, bold=True,
                     sub_text="Microphone  |  16kHz Mono")

    # ══════════════════════════════════════════════
    # ARROWS: Sensor → Expert
    # ══════════════════════════════════════════════
    arr_y1 = sy
    arr_y2 = 0.845

    # ══════════════════════════════════════════════
    # LAYER 2: EXPERT FEATURE EXTRACTION
    # ══════════════════════════════════════════════
    ey = 0.695
    eh = 0.155
    draw_section_band(ax, ey - 0.01, eh + 0.02, C_EXPERT_BORDER)
    draw_section_label(ax, 0.035, ey + eh / 2 + 0.01, "② Expert\n   Layer\n   (Frozen)", C_EXPERT_BORDER, 8)

    # ── K-FER Expert (Large Box) ──
    kfer_x, kfer_w = 0.12, 0.24
    draw_rounded_box(ax, kfer_x, ey, kfer_w, eh, C_EXPERT_BG, C_EXPERT_BORDER,
                     "", 0)

    # K-FER title
    ax.text(kfer_x + kfer_w / 2, ey + eh - 0.012, "K-FER Expert",
            ha="center", va="center", fontsize=11, color=C_EXPERT_BORDER,
            fontproperties=FONT_KR_BOLD, zorder=3)

    # Backbone sub-box
    bb_y = ey + eh - 0.048
    draw_rounded_box(ax, kfer_x + 0.01, bb_y, kfer_w - 0.02, 0.028,
                     "#FFF8E1", C_EXPERT_BORDER, "MobileViTv2-100", 8.5,
                     bold=True, sub_text="Backbone  |  5M params  |  d=384", sub_fontsize=7,
                     linewidth=1.0, radius=0.008)

    # AU RoI sub-box
    au_y = bb_y - 0.035
    draw_rounded_box(ax, kfer_x + 0.01, au_y, kfer_w - 0.02, 0.028,
                     "#FFF8E1", C_EXPERT_BORDER, "AU RoI Cross-Attention", 8.5,
                     bold=True, sub_text="8 AU Regions  |  Bilinear Sampling  |  Gated Fusion", sub_fontsize=7,
                     linewidth=1.0, radius=0.008)

    # FER Head sub-box
    fh_y = au_y - 0.035
    draw_rounded_box(ax, kfer_x + 0.01, fh_y, kfer_w - 0.02, 0.028,
                     "#FFF8E1", C_EXPERT_BORDER, "FER Classification Head", 8.5,
                     bold=True, sub_text="CLS token → MLP → 7-class Softmax", sub_fontsize=7,
                     linewidth=1.0, radius=0.008)

    # Output label
    ax.text(kfer_x + kfer_w / 2, ey + 0.012,
            "Output: 7-class probs  |  FACS 6d  |  EAR/PERCLOS",
            ha="center", va="center", fontsize=7, color=C_SUBTEXT,
            fontproperties=FONT_KR, zorder=3,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="#DDD", linewidth=0.5))

    # Internal arrows
    draw_arrow(ax, kfer_x + kfer_w / 2, bb_y, kfer_x + kfer_w / 2, au_y + 0.028,
               color=C_EXPERT_BORDER, linewidth=1.2)
    draw_arrow(ax, kfer_x + kfer_w / 2, au_y, kfer_x + kfer_w / 2, fh_y + 0.028,
               color=C_EXPERT_BORDER, linewidth=1.2)

    # ── Bio Expert ──
    bio_x, bio_w = 0.40, 0.18
    draw_rounded_box(ax, bio_x, ey, bio_w, eh, C_EXPERT_BG, C_EXPERT_BORDER, "", 0)

    ax.text(bio_x + bio_w / 2, ey + eh - 0.012, "Bio Expert",
            ha="center", va="center", fontsize=11, color=C_EXPERT_BORDER,
            fontproperties=FONT_KR_BOLD, zorder=3)

    bio_items = [
        ("HRV Features", "BVP → mean_hr, sdnn,\nrmssd, lf/hf  (4d)"),
        ("EDA Features", "SCR → mean_scl, peaks,\namp, auc  (5d)"),
        ("Temp Features", "HR/Temp → mean, std,\nrange, slope  (6d)"),
    ]
    for i, (title, desc) in enumerate(bio_items):
        iy = ey + eh - 0.048 - i * 0.038
        draw_rounded_box(ax, bio_x + 0.008, iy, bio_w - 0.016, 0.030,
                         "#FFF8E1", C_EXPERT_BORDER, title, 8, bold=True,
                         sub_text=desc, sub_fontsize=6.5,
                         linewidth=0.8, radius=0.006)

    ax.text(bio_x + bio_w / 2, ey + 0.012,
            "Output: 18d  |  neurokit2  |  0 learnable params",
            ha="center", va="center", fontsize=7, color=C_SUBTEXT,
            fontproperties=FONT_KR, zorder=3,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="#DDD", linewidth=0.5))

    # ── Audio Expert ──
    aud_x, aud_w = 0.62, 0.22
    draw_rounded_box(ax, aud_x, ey, aud_w, eh, C_EXPERT_BG, C_EXPERT_BORDER, "", 0)

    ax.text(aud_x + aud_w / 2, ey + eh - 0.012, "Audio Expert",
            ha="center", va="center", fontsize=11, color=C_EXPERT_BORDER,
            fontproperties=FONT_KR_BOLD, zorder=3)

    aud_items = [
        ("emotion2vec", "ACL 2024  |  768d frozen\n→ 9-class Softmax"),
        ("audeering wav2vec2", "Arousal / Valence /\nDominance (AVD 3d)"),
        ("Audio Quality", "RMS, Voicing Ratio,\nSNR estimation (3d)"),
    ]
    for i, (title, desc) in enumerate(aud_items):
        iy = ey + eh - 0.048 - i * 0.038
        draw_rounded_box(ax, aud_x + 0.008, iy, aud_w - 0.016, 0.030,
                         "#FFF8E1", C_EXPERT_BORDER, title, 8, bold=True,
                         sub_text=desc, sub_fontsize=6.5,
                         linewidth=0.8, radius=0.006)

    ax.text(aud_x + aud_w / 2, ey + 0.012,
            "Output: 15d  |  emotion2vec frozen  |  ~10K MLP only",
            ha="center", va="center", fontsize=7, color=C_SUBTEXT,
            fontproperties=FONT_KR, zorder=3,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="#DDD", linewidth=0.5))

    # Sensor → Expert arrows
    draw_arrow(ax, 0.24, sy, 0.24, ey + eh, color=C_ARROW, linewidth=2.0)
    draw_arrow(ax, 0.49, sy, 0.49, ey + eh, color=C_ARROW, linewidth=2.0)
    draw_arrow(ax, 0.74, sy, 0.73, ey + eh, color=C_ARROW, linewidth=2.0)

    # ══════════════════════════════════════════════
    # LAYER 3: K-MER FUSION
    # ══════════════════════════════════════════════
    fy = 0.49
    fh = 0.175
    draw_section_band(ax, fy - 0.01, fh + 0.02, C_FUSION_BORDER)
    draw_section_label(ax, 0.035, fy + fh / 2, "③ K-MER\n   Fusion", C_FUSION_BORDER, 8)

    # Main fusion box
    draw_rounded_box(ax, 0.12, fy, 0.72, fh, C_FUSION_BG, C_FUSION_BORDER, "", 0,
                     linewidth=2.0)

    ax.text(0.48, fy + fh - 0.012,
            "K-MER Multimodal Fusion  (~144K params)",
            ha="center", va="center", fontsize=12, color=C_FUSION_BORDER,
            fontproperties=FONT_KR_BOLD, zorder=3)

    # Token strip
    tok_y = fy + fh - 0.048
    tok_h = 0.025
    tokens = [
        ("T1", "kfer\n7d", C_TOKEN_FACE),
        ("T2", "meta\n2d", C_TOKEN_FACE),
        ("T3", "face\n3d", C_TOKEN_FACE),
        ("T4", "e2v\n9d", C_TOKEN_AUDIO),
        ("T5", "avd\n3d", C_TOKEN_AUDIO),
        ("T6", "aq\n3d", C_TOKEN_AUDIO),
        ("T7", "bvp\n4d", C_TOKEN_BIO),
        ("T8", "eda\n5d", C_TOKEN_BIO),
        ("T9", "hrt\n6d", C_TOKEN_BIO),
        ("T10", "bq\n3d", C_TOKEN_BIO),
        ("T11", "per\n2d", C_TOKEN_AUX),
        ("T12", "facs\n6d", C_TOKEN_AUX),
        ("T13", "xmd\n3d", C_TOKEN_AUX),
        ("T14", "flg\n3d", C_TOKEN_META),
        ("CLS", "\n64d", C_TOKEN_META),
    ]
    tok_start = 0.135
    tok_w = 0.046
    tok_gap = 0.001
    for i, (name, desc, color) in enumerate(tokens):
        tx = tok_start + i * (tok_w + tok_gap)
        box = FancyBboxPatch(
            (tx, tok_y), tok_w, tok_h,
            boxstyle="round,pad=0,rounding_size=0.004",
            facecolor=color, edgecolor="#999",
            linewidth=0.6, zorder=3
        )
        ax.add_patch(box)
        ax.text(tx + tok_w / 2, tok_y + tok_h * 0.7, name,
                ha="center", va="center", fontsize=6.5,
                fontproperties=FONT_KR_BOLD, color="#333", zorder=4)
        ax.text(tx + tok_w / 2, tok_y + tok_h * 0.25, desc,
                ha="center", va="center", fontsize=5,
                fontproperties=FONT_KR, color="#666", zorder=4)

    # Token labels
    ax.text(0.135, tok_y + tok_h + 0.006, "15 Tokens × 64d   (Linear Projection + Modality Type Embedding)",
            ha="left", va="bottom", fontsize=7.5, color=C_SUBTEXT,
            fontproperties=FONT_KR, zorder=3)

    # Token legend
    legend_items = [
        (C_TOKEN_FACE, "Face"), (C_TOKEN_AUDIO, "Audio"),
        (C_TOKEN_BIO, "Bio"), (C_TOKEN_AUX, "Aux"), (C_TOKEN_META, "Meta")
    ]
    for i, (c, lbl) in enumerate(legend_items):
        lx = 0.58 + i * 0.055
        box = FancyBboxPatch(
            (lx, tok_y + tok_h + 0.005), 0.012, 0.008,
            boxstyle="round,pad=0,rounding_size=0.002",
            facecolor=c, edgecolor="#999", linewidth=0.4, zorder=3
        )
        ax.add_patch(box)
        ax.text(lx + 0.015, tok_y + tok_h + 0.009, lbl,
                ha="left", va="center", fontsize=6, color="#555",
                fontproperties=FONT_KR, zorder=3)

    # Pool-FFN boxes
    pf_y = tok_y - 0.037
    pf_h = 0.028
    pf_items = [
        (0.14, 0.14, "Pool-FFN (Face)", "Intra-modal  T1-T3", C_TOKEN_FACE),
        (0.34, 0.14, "Pool-FFN (Audio)", "Intra-modal  T4-T6", C_TOKEN_AUDIO),
        (0.54, 0.18, "Pool-FFN (Bio)", "Intra-modal  T7-T10", C_TOKEN_BIO),
    ]
    for px, pw, title, sub, c in pf_items:
        draw_rounded_box(ax, px, pf_y, pw, pf_h, c, "#999",
                         title, 8, bold=True, sub_text=sub, sub_fontsize=6.5,
                         linewidth=0.8, radius=0.006)

    # MHSA box
    mhsa_y = pf_y - 0.038
    draw_rounded_box(ax, 0.14, mhsa_y, 0.58, 0.028,
                     "#E0F2F1", C_FUSION_BORDER,
                     "Global Multi-Head Self-Attention  (1 layer, 4 heads, d=64)",
                     9, bold=True,
                     sub_text="Pre-Norm → Attention → Residual → FFN(64→128→64) → Residual  |  Validity Masking",
                     sub_fontsize=6.5, linewidth=1.2, radius=0.008)

    # Output heads
    head_y = mhsa_y - 0.04
    head_h = 0.028
    heads = [
        (0.17, "Arousal Head", "64→32→1, Sigmoid\narousal ∈ [0,1]", "#C8E6C9"),
        (0.37, "Valence Head", "64→32→1, Sigmoid\nvalence ∈ [0,1]", "#C8E6C9"),
        (0.57, "Drowsy Head", "66→32→3, Softmax\nalert/drowsy/sleep", "#C8E6C9"),
    ]
    for hx, title, sub, c in heads:
        draw_rounded_box(ax, hx, head_y, 0.14, head_h, c, C_FUSION_BORDER,
                         title, 8, bold=True, sub_text=sub, sub_fontsize=6,
                         linewidth=0.8, radius=0.006)

    # CLS Pooling label
    ax.text(0.43, mhsa_y - 0.007, "CLS Pooling → fused_repr (64d)",
            ha="center", va="center", fontsize=7.5, color=C_FUSION_BORDER,
            fontproperties=FONT_KR_BOLD, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=C_FUSION_BORDER, linewidth=0.6))

    # Internal arrows for fusion
    draw_arrow(ax, 0.43, tok_y, 0.43, pf_y + pf_h, color=C_FUSION_BORDER, linewidth=1.2)
    draw_arrow(ax, 0.43, pf_y, 0.43, mhsa_y + 0.028, color=C_FUSION_BORDER, linewidth=1.2)
    draw_arrow(ax, 0.24, mhsa_y, 0.24, head_y + head_h, color=C_FUSION_BORDER, linewidth=1.0)
    draw_arrow(ax, 0.44, mhsa_y, 0.44, head_y + head_h, color=C_FUSION_BORDER, linewidth=1.0)
    draw_arrow(ax, 0.64, mhsa_y, 0.64, head_y + head_h, color=C_FUSION_BORDER, linewidth=1.0)

    # Expert → Fusion arrows
    draw_arrow(ax, 0.24, ey, 0.24, fy + fh, color=C_ARROW, linewidth=2.0)
    draw_arrow(ax, 0.49, ey, 0.49, fy + fh, color=C_ARROW, linewidth=2.0)
    draw_arrow(ax, 0.73, ey, 0.73, fy + fh, color=C_ARROW, linewidth=2.0)

    # ══════════════════════════════════════════════
    # LAYER 4: POST-PROCESSING
    # ══════════════════════════════════════════════
    py = 0.33
    ph = 0.12
    draw_section_band(ax, py - 0.01, ph + 0.02, C_POST_BORDER)
    draw_section_label(ax, 0.035, py + ph / 2, "④ Post-\n   Processing", C_POST_BORDER, 8)

    # Temporal Smoothing
    draw_rounded_box(ax, 0.12, py + 0.01, 0.20, ph - 0.02, C_POST_BG, C_POST_BORDER,
                     "Temporal Smoothing", 9.5, bold=True, linewidth=1.2,
                     sub_text="7-frame Sliding Window\nMajority Vote\n단일프레임 79.7% → 운용시 94.0%",
                     sub_fontsize=7)

    # Compound Emotion
    draw_rounded_box(ax, 0.36, py + 0.01, 0.20, ph - 0.02, C_POST_BG, C_POST_BORDER,
                     "Compound Emotion", 9.5, bold=True, linewidth=1.2,
                     sub_text="K-FER 7-class × Arousal\n→ 13 Refined Labels\n(calm, excited, stressed...)",
                     sub_fontsize=7)

    # Drowsiness / Status
    draw_rounded_box(ax, 0.60, py + 0.01, 0.24, ph - 0.02, C_POST_BG, C_POST_BORDER,
                     "Status Flag Detector", 9.5, bold=True, linewidth=1.2,
                     sub_text="Stress: arousal>0.6 + 부정감정\nLow Attention: PERCLOS∈[0.2,0.4)\nDrowsy: PERCLOS≥0.4  (NHTSA)",
                     sub_fontsize=7)

    # Fusion → Post-Processing arrows
    draw_arrow(ax, 0.24, fy, 0.22, py + ph, color=C_ARROW, linewidth=2.0)
    draw_arrow(ax, 0.44, fy, 0.46, py + ph, color=C_ARROW, linewidth=2.0)
    draw_arrow(ax, 0.64, fy, 0.72, py + ph, color=C_ARROW, linewidth=2.0)

    # ══════════════════════════════════════════════
    # LAYER 5: GATEWAY PACKET
    # ══════════════════════════════════════════════
    gy = 0.175
    gh = 0.115
    draw_section_band(ax, gy - 0.01, gh + 0.02, C_GATEWAY_BORDER)
    draw_section_label(ax, 0.035, gy + gh / 2, "⑤ Gateway\n   Output", C_GATEWAY_BORDER, 8)

    # Main gateway box
    draw_rounded_box(ax, 0.12, gy, 0.72, gh, C_GATEWAY_BG, C_GATEWAY_BORDER, "", 0,
                     linewidth=2.0)

    ax.text(0.48, gy + gh - 0.012,
            "8-Byte USB Packet Encoder → Vehicle Gateway",
            ha="center", va="center", fontsize=11, color=C_GATEWAY_BORDER,
            fontproperties=FONT_KR_BOLD, zorder=3)

    # Packet byte layout
    pkt_y = gy + gh - 0.055
    pkt_h = 0.028
    bytes_layout = [
        (0.135, 0.06, "0xAA\nSOF", "#FFCDD2"),
        (0.20, 0.06, "TYPE\n0x01", "#FFCDD2"),
        (0.265, 0.06, "SEQ\n0~255", "#FFCDD2"),
        (0.33, 0.06, "LEN\n0x02", "#FFCDD2"),
        (0.395, 0.14, "Byte 4\nEmoCode(4b)+Flags(4b)", "#EF9A9A"),
        (0.54, 0.12, "Byte 5\nIntensity(6b)+NegEmo(1b)", "#EF9A9A"),
        (0.665, 0.06, "CRC8\nB1~B5", "#FFCDD2"),
        (0.73, 0.06, "0xFE\nEOF", "#FFCDD2"),
    ]
    for bx, bw, txt, bc in bytes_layout:
        draw_rounded_box(ax, bx, pkt_y, bw, pkt_h, bc, C_GATEWAY_BORDER,
                         txt, 7, linewidth=0.8, radius=0.005)

    # Emotion code mapping (compact)
    map_y = pkt_y - 0.042
    map_text = "Emotion Code (6종):  0=공포  1=놀람  2=분노  3=슬픔/혐오(sad+hurt 병합)  4=행복  5=중립"
    ax.text(0.48, map_y + 0.016, map_text,
            ha="center", va="center", fontsize=7.5, color="#333",
            fontproperties=FONT_KR, zorder=3)

    flag_text = "Status Flags (4종):  [Byte4] Stress | Low Attention | Drowsy  [Byte5] Negative Emotion"
    ax.text(0.48, map_y, flag_text,
            ha="center", va="center", fontsize=7.5, color="#333",
            fontproperties=FONT_KR, zorder=3)

    # Post-Processing → Gateway arrows
    draw_arrow(ax, 0.22, py, 0.35, gy + gh, color=C_ARROW, linewidth=2.0)
    draw_arrow(ax, 0.46, py, 0.48, gy + gh, color=C_ARROW, linewidth=2.0)
    draw_arrow(ax, 0.72, py, 0.60, gy + gh, color=C_ARROW, linewidth=2.0)

    # ══════════════════════════════════════════════
    # VEHICLE GATEWAY (output)
    # ══════════════════════════════════════════════
    vg_y = 0.12
    draw_rounded_box(ax, 0.30, vg_y, 0.36, 0.035, C_HEADER_BG, C_HEADER_BG,
                     "차량 게이트웨이  (USB Communication)", 11,
                     text_color=C_HEADER_TEXT, bold=True, radius=0.01)
    draw_arrow(ax, 0.48, gy, 0.48, vg_y + 0.035, color=C_GATEWAY_BORDER, linewidth=2.5)

    # ══════════════════════════════════════════════
    # PERFORMANCE TABLE (bottom)
    # ══════════════════════════════════════════════
    perf_y = 0.01
    perf_h = 0.095
    draw_rounded_box(ax, 0.04, perf_y, 0.92, perf_h, C_PERF_BG, C_PERF_BORDER,
                     "", 0, linewidth=1.5, radius=0.01)

    ax.text(0.5, perf_y + perf_h - 0.012,
            "실증 검증 성능  —  10개 인식 항목 종합 정확도",
            ha="center", va="center", fontsize=10, color=C_PERF_BORDER,
            fontproperties=FONT_KR_BOLD, zorder=3)

    # Performance items in two rows
    perf_items_r1 = [
        ("공포", "97.9%"), ("놀람", "99.4%"), ("분노", "100%"),
        ("슬픔/혐오", "95.7%"), ("행복", "100%"), ("중립", "100%"),
    ]
    perf_items_r2 = [
        ("스트레스", "95.0%"), ("주의분산", "95.0%"), ("졸음", "96.0%"), ("부정감정", "99.8%"),
    ]

    # Row 1 - emotions (6종)
    for i, (name, acc) in enumerate(perf_items_r1):
        px = 0.09 + i * 0.12
        ax.text(px, perf_y + perf_h - 0.038, f"Code {i}",
                ha="center", va="center", fontsize=6, color="#888",
                fontproperties=FONT_KR, zorder=3)
        ax.text(px, perf_y + perf_h - 0.052, name,
                ha="center", va="center", fontsize=8.5, color="#333",
                fontproperties=FONT_KR_BOLD, zorder=3)
        acc_color = "#C62828" if float(acc.strip('%')) < 80 else C_ACCENT_GREEN
        ax.text(px, perf_y + perf_h - 0.067, acc,
                ha="center", va="center", fontsize=9, color=acc_color,
                fontproperties=FONT_KR_BOLD, zorder=3)

    # Row 2 - status flags (4종)
    for i, (name, acc) in enumerate(perf_items_r2):
        px = 0.15 + i * 0.16
        ax.text(px, perf_y + 0.022, f"Flag {i+1}",
                ha="center", va="center", fontsize=6, color="#888",
                fontproperties=FONT_KR, zorder=3)
        ax.text(px, perf_y + 0.012, name,
                ha="center", va="center", fontsize=8.5, color="#333",
                fontproperties=FONT_KR_BOLD, zorder=3)

    # Macro average highlight
    avg_box = FancyBboxPatch(
        (0.72, perf_y + 0.005), 0.20, 0.035,
        boxstyle="round,pad=0,rounding_size=0.008",
        facecolor=C_ACCENT_GREEN, edgecolor=C_ACCENT_GREEN,
        linewidth=0, alpha=0.15, zorder=2
    )
    ax.add_patch(avg_box)
    ax.text(0.82, perf_y + 0.028, "종합 Macro Average",
            ha="center", va="center", fontsize=8, color=C_ACCENT_GREEN,
            fontproperties=FONT_KR_BOLD, zorder=3)
    ax.text(0.82, perf_y + 0.012, "98.0%",
            ha="center", va="center", fontsize=16, color=C_ACCENT_GREEN,
            fontproperties=FONT_KR_BOLD, zorder=3)

    # ══════════════════════════════════════════════
    # KD annotation (right side)
    # ══════════════════════════════════════════════
    kd_x, kd_y = 0.86, 0.52
    kd_w, kd_h = 0.11, 0.13
    draw_rounded_box(ax, kd_x, kd_y, kd_w, kd_h, "#E3F2FD", "#1565C0",
                     "", 0, linewidth=1.2, radius=0.008)

    ax.text(kd_x + kd_w / 2, kd_y + kd_h - 0.012, "KD Student",
            ha="center", va="center", fontsize=8.5, color="#1565C0",
            fontproperties=FONT_KR_BOLD, zorder=3)
    kd_info = [
        "Teacher: 144K",
        "Student: 12K",
        "",
        "Face: 12d→64d",
        "Bio: 18d→64d",
        "Audio: 15d→64d",
        "",
        "UAR: 59.2%",
        "(교사 98.6%)",
    ]
    for i, line in enumerate(kd_info):
        ax.text(kd_x + kd_w / 2, kd_y + kd_h - 0.028 - i * 0.012, line,
                ha="center", va="center", fontsize=6, color="#333",
                fontproperties=FONT_KR, zorder=3)

    # Dashed arrow from fusion to KD
    ax.annotate("",
                xy=(kd_x, kd_y + kd_h / 2),
                xytext=(0.84, fy + fh / 2),
                arrowprops=dict(arrowstyle="-|>", color="#1565C0",
                                linewidth=1.2, linestyle="dashed",
                                mutation_scale=10),
                zorder=1)
    ax.text(0.855, fy + fh / 2 + 0.015, "Knowledge\nDistillation",
            ha="center", va="center", fontsize=6.5, color="#1565C0",
            fontproperties=FONT_KR, zorder=3, rotation=0)

    # ══════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════
    out_path = "/home/ajy/Jetson_thor/kmer_system_architecture.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor=C_BG, edgecolor="none", pad_inches=0.3)
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")
    print(f"     Size: {fig.get_size_inches()[0]*200:.0f} x {fig.get_size_inches()[1]*200:.0f} px")


if __name__ == "__main__":
    main()
