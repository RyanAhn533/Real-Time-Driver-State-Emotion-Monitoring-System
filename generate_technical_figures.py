#!/usr/bin/env python3
"""
K-MER 시스템 기술 아키텍처 다이어그램 생성
교수님 발표용 — 내부 구조 상세 시각화
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties
import numpy as np

# ── Fonts ──
FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_B = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
FONT_M = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")

# ── Colors ──
BG = "#FFFFFF"
NAVY = "#1B3A5C"
BLUE = "#2980B9"
DBLUE = "#1565C0"
LBLUE = "#E3F2FD"
GREEN = "#27AE60"
LGREEN = "#E8F5E9"
ORANGE = "#E67E22"
LORANGE = "#FFF3E0"
RED = "#C62828"
LRED = "#FFEBEE"
PURPLE = "#8E24AA"
LPURPLE = "#F3E5F5"
GRAY = "#888888"
DARK = "#2C3E50"
LGRAY = "#F5F5F5"

def box(ax, x, y, w, h, fc, ec, text="", fs=9, tc=DARK, bold=False,
        sub="", sfs=7, radius=0.012, lw=1.5, alpha=1.0, zorder=2):
    b = FancyBboxPatch((x,y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(b)
    fp = FONT_B if bold else FONT
    if text:
        ty = y + h/2 + (h*0.12 if sub else 0)
        ax.text(x+w/2, ty, text, ha="center", va="center", fontsize=fs,
                color=tc, fontproperties=fp, zorder=zorder+1)
    if sub:
        ax.text(x+w/2, y+h/2 - h*0.15, sub, ha="center", va="center",
                fontsize=sfs, color=GRAY, fontproperties=FONT, zorder=zorder+1)

def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.5, style="-|>"):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle=style, color=color, linewidth=lw, mutation_scale=12),
                zorder=5)

def darrow(ax, x1, y1, x2, y2, color=DBLUE, lw=1.2):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lw,
                                linestyle="dashed", mutation_scale=10), zorder=5)

def label(ax, x, y, text, fs=8, color=DARK, bold=False, ha="center"):
    fp = FONT_B if bold else FONT
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color,
            fontproperties=fp, zorder=6)

def badge(ax, x, y, text, fs=7, fc=LBLUE, ec=BLUE, tc=DBLUE):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
            fontproperties=FONT_B, zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=fc, edgecolor=ec, linewidth=0.8))


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: AU RoI Cross-Attention (K-FER 내부 구조)
# ═══════════════════════════════════════════════════════════════
def fig1_au_cross_attention():
    fig, ax = plt.subplots(figsize=(18, 22))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("auto"); ax.axis("off")
    fig.patch.set_facecolor(BG)

    # Title
    box(ax, 0.02, 0.96, 0.96, 0.032, NAVY, NAVY,
        "K-FER: AU RoI Cross-Attention Architecture", 15, "#FFF", True, radius=0.008)
    label(ax, 0.5, 0.952, "MobileViTv2 Backbone + Action Unit Region-of-Interest Cross-Attention Fusion",
          8, GRAY)

    # ── INPUT ──
    box(ax, 0.35, 0.90, 0.30, 0.035, LBLUE, BLUE,
        "Input Image  [B, 3, 224, 224]", 10, DBLUE, True, radius=0.008)

    arrow(ax, 0.50, 0.90, 0.50, 0.875)

    # ── BACKBONE ──
    box(ax, 0.15, 0.80, 0.70, 0.07, LORANGE, ORANGE, "", radius=0.01, lw=2)
    label(ax, 0.50, 0.855, "MobileViTv2-100 Backbone (Frozen)", 12, ORANGE, True)
    label(ax, 0.50, 0.84, "ImageNet Pretrained  |  5M params  |  Single Forward Pass", 8, GRAY)

    box(ax, 0.17, 0.81, 0.28, 0.025, "#FFF8E1", ORANGE,
        "Feature Map F : [B, 384, 7, 7]", 9, DARK, True, radius=0.006, lw=1)
    box(ax, 0.55, 0.81, 0.28, 0.025, "#FFF8E1", ORANGE,
        "Global Feature g ∈ ℝ^(B×384)", 9, DARK, True, radius=0.006, lw=1)

    arrow(ax, 0.50, 0.80, 0.50, 0.785)

    # ── AU ROI EXTRACTION ──
    box(ax, 0.08, 0.66, 0.84, 0.12, LBLUE, BLUE, "", radius=0.01, lw=2)
    label(ax, 0.50, 0.77, "AU RoI Extraction Module", 12, DBLUE, True)

    # MediaPipe landmarks
    box(ax, 0.10, 0.72, 0.22, 0.035, "#E1F5FE", BLUE,
        "MediaPipe FaceMesh", 9, DBLUE, True,
        sub="468 Landmarks → 8 AU Coords", sfs=7, radius=0.006, lw=1)

    # Normalize
    box(ax, 0.34, 0.72, 0.18, 0.035, "#E1F5FE", BLUE,
        "Normalize to [-1,1]", 9, DBLUE, True,
        sub="Grid Format for grid_sample", sfs=7, radius=0.006, lw=1)

    # grid_sample
    box(ax, 0.54, 0.72, 0.18, 0.035, "#BBDEFB", BLUE,
        "F.grid_sample", 9, DBLUE, True,
        sub="mode='bilinear'\npadding='border'", sfs=6.5, radius=0.006, lw=1)

    # AU Embedding
    box(ax, 0.74, 0.72, 0.16, 0.035, "#BBDEFB", BLUE,
        "AU Embedding", 9, DBLUE, True,
        sub="Embedding(8, 384)", sfs=7, radius=0.006, lw=1)

    arrow(ax, 0.32, 0.737, 0.34, 0.737, BLUE, 1.2)
    arrow(ax, 0.52, 0.737, 0.54, 0.737, BLUE, 1.2)
    arrow(ax, 0.72, 0.737, 0.74, 0.737, BLUE, 1.2)

    # Output
    box(ax, 0.25, 0.67, 0.50, 0.035, "#E8EAF6", DBLUE,
        "AU Tokens  [B, 8, 384]   +   AU Position Embedding", 9, DBLUE, True, radius=0.006, lw=1)

    # 8 AU regions
    au_names = ["L-Brow", "R-Brow", "L-Eye", "R-Eye", "Nose", "L-Mouth", "R-Mouth", "Chin"]
    for i, name in enumerate(au_names):
        bx = 0.10 + i * 0.10
        box(ax, bx, 0.675, 0.075, 0.015, "#C5CAE9", "#5C6BC0",
            name, 5.5, "#333", False, radius=0.003, lw=0.6)

    arrow(ax, 0.50, 0.66, 0.50, 0.645)

    # ── TOKEN ASSEMBLY ──
    box(ax, 0.15, 0.60, 0.70, 0.04, LGRAY, GRAY,
        "[CLS] (384d)  +  [Global] (384d)  +  [AU1, AU2, ... AU8] (384d each)  =  [B, 10, 384]",
        9, DARK, True, radius=0.006, lw=1)

    arrow(ax, 0.50, 0.60, 0.50, 0.585)

    # ══════════════════════════════════════
    # CROSS-ATTENTION FUSION LAYER
    # ══════════════════════════════════════
    box(ax, 0.05, 0.295, 0.90, 0.285, LGREEN, GREEN, "", radius=0.012, lw=2.5)
    label(ax, 0.50, 0.572, "Cross-Attention Fusion Layer  (×1)", 13, GREEN, True)
    label(ax, 0.50, 0.555, "Pre-Norm Architecture  |  d=384  |  8 Heads  |  48 dims/head", 8, GRAY)

    # ── CROSS-ATTENTION BLOCK ──
    box(ax, 0.07, 0.44, 0.42, 0.10, "#C8E6C9", GREEN, "", radius=0.008, lw=1.5)
    label(ax, 0.28, 0.535, "Cross-Attention Block", 10, GREEN, True)

    # Q branch
    box(ax, 0.09, 0.49, 0.12, 0.03, "#E8F5E9", GREEN,
        "Q: [CLS, Global]", 7.5, DARK, True, radius=0.005, lw=0.8)
    box(ax, 0.09, 0.455, 0.12, 0.025, "#E8F5E9", GREEN,
        "LayerNorm(384)", 7, DARK, False, radius=0.005, lw=0.8)

    # KV branch
    box(ax, 0.23, 0.49, 0.12, 0.03, "#E8F5E9", GREEN,
        "K,V: [AU1..AU8]", 7.5, DARK, True, radius=0.005, lw=0.8)
    box(ax, 0.23, 0.455, 0.12, 0.025, "#E8F5E9", GREEN,
        "LayerNorm(384)", 7, DARK, False, radius=0.005, lw=0.8)

    # MHSA
    box(ax, 0.37, 0.47, 0.10, 0.04, "#A5D6A7", GREEN,
        "MHSA\n8 heads", 8, DARK, True, radius=0.006, lw=1)
    arrow(ax, 0.15, 0.455, 0.37, 0.492, GREEN, 1)
    arrow(ax, 0.29, 0.455, 0.37, 0.485, GREEN, 1)

    # ── GATED RESIDUAL (핵심!) ──
    box(ax, 0.07, 0.36, 0.42, 0.07, "#FFF9C4", "#F9A825", "", radius=0.008, lw=2)
    label(ax, 0.28, 0.425, "[Key] Gated Residual Connection", 10, "#F57F17", True)

    # Gate mechanism
    box(ax, 0.09, 0.375, 0.18, 0.035, "#FFF8E1", "#F9A825",
        "gate = σ(W·[global; au])", 8, DARK, True,
        sub="Per-dimension learnable  |  init=0.0", sfs=6.5, radius=0.005, lw=1)
    box(ax, 0.29, 0.375, 0.18, 0.035, "#FFF8E1", "#F9A825",
        "out = Q + gate ⊙ attn_out", 8, DARK, True,
        sub="Element-wise gating on 384 dims", sfs=6.5, radius=0.005, lw=1)
    arrow(ax, 0.27, 0.392, 0.29, 0.392, "#F9A825", 1.2)

    # ── SELF-ATTENTION BLOCK ──
    box(ax, 0.52, 0.44, 0.40, 0.10, "#C8E6C9", GREEN, "", radius=0.008, lw=1.5)
    label(ax, 0.72, 0.535, "Self-Attention Block", 10, GREEN, True)

    box(ax, 0.54, 0.49, 0.16, 0.03, "#E8F5E9", GREEN,
        "All 10 tokens", 8, DARK, True, radius=0.005, lw=0.8)
    box(ax, 0.54, 0.455, 0.16, 0.025, "#E8F5E9", GREEN,
        "LayerNorm(384)", 7, DARK, False, radius=0.005, lw=0.8)
    box(ax, 0.72, 0.455, 0.18, 0.04, "#A5D6A7", GREEN,
        "Full Self-MHSA\n8 heads, d=384", 8, DARK, True, radius=0.006, lw=1)
    arrow(ax, 0.62, 0.455, 0.72, 0.477, GREEN, 1)
    box(ax, 0.72, 0.505, 0.18, 0.025, "#E8F5E9", GREEN,
        "Residual: x += self_out", 7, DARK, False, radius=0.005, lw=0.8)

    # ── FFN BLOCK ──
    box(ax, 0.52, 0.305, 0.40, 0.12, "#C8E6C9", GREEN, "", radius=0.008, lw=1.5)
    label(ax, 0.72, 0.42, "Feed-Forward Network", 10, GREEN, True)

    ffn_items = [
        (0.54, 0.39, "LayerNorm(384)"),
        (0.54, 0.365, "Linear(384 → 1536)"),
        (0.54, 0.34, "GELU + Dropout(0.1)"),
        (0.74, 0.39, "Linear(1536 → 384)"),
        (0.74, 0.365, "Dropout(0.1)"),
        (0.74, 0.34, "Residual: x += ffn_out"),
    ]
    for fx, fy, ft in ffn_items:
        box(ax, fx, fy, 0.16, 0.02, "#E8F5E9", GREEN,
            ft, 6.5, DARK, False, radius=0.004, lw=0.6)

    # Flow arrow between cross and self
    arrow(ax, 0.49, 0.49, 0.52, 0.49, DARK, 1.5)

    # Output of fusion layer
    arrow(ax, 0.50, 0.295, 0.50, 0.275)

    # ── CLS TOKEN EXTRACTION ──
    box(ax, 0.25, 0.235, 0.50, 0.035, LPURPLE, PURPLE,
        "Extract CLS Token  →  [B, 384]", 10, PURPLE, True, radius=0.008, lw=1.5)

    arrow(ax, 0.50, 0.235, 0.50, 0.215)

    # ── CLASSIFICATION HEAD ──
    box(ax, 0.15, 0.14, 0.70, 0.07, LPURPLE, PURPLE, "", radius=0.01, lw=2)
    label(ax, 0.50, 0.20, "FER Classification Head", 11, PURPLE, True)

    head_items = [
        (0.18, "LayerNorm(384)"),
        (0.35, "Linear(384→384)\n+ GELU"),
        (0.55, "Dropout(0.2)"),
        (0.72, "Linear(384→7)"),
    ]
    for hx, ht in head_items:
        box(ax, hx, 0.155, 0.14, 0.03, "#E1BEE7", PURPLE,
            ht, 7.5, DARK, True, radius=0.005, lw=0.8)
        if hx < 0.72:
            arrow(ax, hx+0.14, 0.17, hx+0.17, 0.17, PURPLE, 1)

    # ── OUTPUT ──
    arrow(ax, 0.50, 0.14, 0.50, 0.12)

    out_items = [
        (0.12, "7-class Softmax\n감정 확률"),
        (0.35, "FACS 6d\nAU Activation"),
        (0.58, "EAR\nEye Aspect Ratio"),
        (0.78, "PERCLOS\n졸음 지표"),
    ]
    for ox, ot in out_items:
        box(ax, ox, 0.06, 0.17, 0.05, LRED, RED,
            ot, 8, RED, True, radius=0.006, lw=1.2)

    # ── Key insight callout ──
    box(ax, 0.03, 0.015, 0.94, 0.035, "#FFF9C4", "#F9A825", "", radius=0.008, lw=1.5)
    label(ax, 0.50, 0.035, "핵심 혁신:  기존 321회 Backbone Forward → 단 1회 Forward + grid_sample AU 추출  |  "
          "Per-dim Gated Residual로 Global-Local 정보 균형 제어", 8.5, "#F57F17", True)

    fig.savefig("/home/ajy/Jetson_thor/fig_kfer_architecture.png",
                dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.2)
    plt.close(fig)
    print("[OK] fig_kfer_architecture.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: K-MER Fusion 내부 파이프라인
# ═══════════════════════════════════════════════════════════════
def fig2_kmer_fusion():
    fig, ax = plt.subplots(figsize=(20, 24))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("auto"); ax.axis("off")
    fig.patch.set_facecolor(BG)

    # Title
    box(ax, 0.02, 0.965, 0.96, 0.028, NAVY, NAVY,
        "K-MER Fusion: Cross-Modal Multimodal Fusion Architecture (~144K params)", 14, "#FFF", True, radius=0.008)

    # ═══ STAGE 1: TOKEN FORMATION ═══
    box(ax, 0.03, 0.855, 0.94, 0.10, LBLUE, BLUE, "", radius=0.01, lw=2)
    label(ax, 0.50, 0.95, "Stage 1:  Token Formation  (14 Expert Tokens + 1 CLS Token)", 11, DBLUE, True)

    # Token groups
    colors_t = {"face": "#BBDEFB", "audio": "#FFE0B2", "bio": "#C8E6C9", "aux": "#E1BEE7", "meta": "#CFD8DC"}
    tokens = [
        ("T1", "kfer\n7d", "face"), ("T2", "meta\n2d", "face"), ("T3", "face\n3d", "face"),
        ("T4", "e2v\n9d", "audio"), ("T5", "avd\n3d", "audio"), ("T6", "aq\n3d", "audio"),
        ("T7", "bvp\n4d", "bio"), ("T8", "eda\n5d", "bio"), ("T9", "hrt\n6d", "bio"), ("T10", "bq\n3d", "bio"),
        ("T11", "per\n2d", "aux"), ("T12", "facs\n6d", "aux"), ("T13", "xmd\n3d", "aux"),
        ("T14", "flg\n3d", "meta"), ("CLS", "learn\n64d", "meta"),
    ]
    tw = 0.055
    for i, (name, desc, mod) in enumerate(tokens):
        tx = 0.05 + i * 0.06
        box(ax, tx, 0.90, tw, 0.035, colors_t[mod], GRAY,
            name, 7, DARK, True, sub=desc, sfs=5.5, radius=0.005, lw=0.8)

    # Projection annotation
    label(ax, 0.50, 0.893, "Linear(d_in → 64) + LayerNorm(64)  per token  |  + Modality Type Embedding(5 types → 64d)", 7.5, GRAY)

    # Token groups bracket
    brackets = [
        (0.05, 0.23, "Face (3)", BLUE), (0.23, 0.41, "Audio (3)", ORANGE),
        (0.41, 0.65, "Bio (4)", GREEN), (0.65, 0.83, "Aux (3)", PURPLE),
        (0.83, 0.95, "Meta (2)", GRAY),
    ]
    for x1, x2, lbl, c in brackets:
        ax.plot([x1, x2], [0.865, 0.865], color=c, linewidth=2, zorder=5)
        label(ax, (x1+x2)/2, 0.860, lbl, 6.5, c, True)

    # Arrow down
    arrow(ax, 0.50, 0.855, 0.50, 0.845)

    # ═══ STAGE 2: POOL-FFN ═══
    box(ax, 0.03, 0.72, 0.94, 0.12, LORANGE, ORANGE, "", radius=0.01, lw=2)
    label(ax, 0.50, 0.835, "Stage 2:  Intra-Modal Pool-FFN  (EfficientFormer-style)", 11, ORANGE, True)

    pfn_items = [
        (0.05, 0.19, "Face Pool-FFN", "T1,T2,T3 → [B,3,64]", "AvgPool1d(k=3)\n→ FFN(64→128→64)\n→ LayerNorm + GELU\n→ Residual", BLUE),
        (0.26, 0.19, "Audio Pool-FFN", "T4,T5,T6 → [B,3,64]", "AvgPool1d(k=3)\n→ FFN(64→128→64)\n→ LayerNorm + GELU\n→ Residual", ORANGE),
        (0.47, 0.22, "Bio Pool-FFN", "T7~T10 → [B,4,64]", "AvgPool1d(k=4)\n→ FFN(64→128→64)\n→ LayerNorm + GELU\n→ Residual", GREEN),
        (0.71, 0.22, "Passthrough", "T11~T14,CLS → [B,5,64]", "No processing\n(Aux + Meta tokens\npass unchanged)", GRAY),
    ]
    for px, pw, title, inp, desc, c in pfn_items:
        box(ax, px, 0.735, pw, 0.085, "#FFF8E1", c,
            title, 9, c, True, sub=f"{inp}\n\n{desc}", sfs=6.5, radius=0.006, lw=1.2)

    arrow(ax, 0.50, 0.72, 0.50, 0.71)

    # ═══ STAGE 3: GLOBAL MHSA ═══
    box(ax, 0.03, 0.54, 0.94, 0.165, LGREEN, GREEN, "", radius=0.01, lw=2.5)
    label(ax, 0.50, 0.70, "Stage 3:  Global Cross-Modal Multi-Head Self-Attention", 11, GREEN, True)

    # Pre-Norm
    box(ax, 0.06, 0.66, 0.24, 0.028, "#C8E6C9", GREEN,
        "Pre-Norm: LayerNorm(64)", 8, DARK, True, radius=0.005, lw=1)

    # MHSA
    box(ax, 0.32, 0.64, 0.36, 0.05, "#A5D6A7", GREEN, "", radius=0.008, lw=1.5)
    label(ax, 0.50, 0.673, "Multi-Head Self-Attention", 10, GREEN, True)
    label(ax, 0.50, 0.655, "4 heads  |  d_k = 16  |  All 15 tokens attend to each other", 7.5, DARK)

    # Validity Masking
    box(ax, 0.70, 0.66, 0.24, 0.028, "#FFCDD2", RED,
        "Validity Masking (key_padding_mask)", 7.5, RED, True, radius=0.005, lw=1)
    label(ax, 0.82, 0.645, "센서 미연결 시 해당 모달리티 토큰 masking", 6, GRAY)

    arrow(ax, 0.30, 0.674, 0.32, 0.674, GREEN, 1.2)
    arrow(ax, 0.68, 0.665, 0.70, 0.665, RED, 1.0)

    # Residual
    box(ax, 0.15, 0.61, 0.30, 0.022, "#E8F5E9", GREEN,
        "Residual:  tokens = tokens + attn_output", 7.5, DARK, True, radius=0.005, lw=0.8)

    # FFN
    box(ax, 0.06, 0.565, 0.86, 0.035, "#E8F5E9", GREEN, "", radius=0.006, lw=1.2)
    label(ax, 0.49, 0.588, "Feed-Forward Network", 9, GREEN, True)
    label(ax, 0.49, 0.573, "LayerNorm(64) → Linear(64→128) → GELU → Dropout(0.1) → Linear(128→64) → Dropout(0.1) → Residual", 7, DARK)

    # CLS Pooling
    box(ax, 0.30, 0.545, 0.40, 0.018, "#C8E6C9", GREEN,
        "CLS Pooling:  fused_repr = tokens[:, -1, :]  →  [B, 64]", 7.5, GREEN, True, radius=0.005, lw=1)

    arrow(ax, 0.50, 0.54, 0.50, 0.525)

    # ═══ STAGE 4: OUTPUT HEADS ═══
    box(ax, 0.03, 0.39, 0.94, 0.13, LPURPLE, PURPLE, "", radius=0.01, lw=2)
    label(ax, 0.50, 0.515, "Stage 4:  Task-Specific Output Heads", 11, PURPLE, True)

    heads = [
        (0.06, "Arousal Head", "fused_repr [B,64]\n→ Linear(64→32) + ReLU\n→ Dropout(0.1)\n→ Linear(32→1) + Sigmoid\n→ arousal ∈ [0,1]",
         "#E1BEE7", PURPLE),
        (0.33, "Valence Head", "fused_repr [B,64]\n→ Linear(64→32) + ReLU\n→ Dropout(0.1)\n→ Linear(32→1) + Sigmoid\n→ valence ∈ [0,1]",
         "#E1BEE7", PURPLE),
        (0.60, "Drowsy Head", "concat([fused.detach(), perclos_ear])\n→ [B, 66]\n→ Linear(66→32) + ReLU\n→ Dropout(0.1)\n→ Linear(32→3)\n→ alert / drowsy / sleep",
         "#FFCDD2", RED),
    ]
    for hx, title, desc, fc, ec in heads:
        box(ax, hx, 0.405, 0.24, 0.095, fc, ec,
            title, 9, ec, True, sub=desc, sfs=6.5, radius=0.006, lw=1.2)

    # Detach callout
    badge(ax, 0.80, 0.445, "[Key] .detach()로 gradient 차단\n→ 감정 loss가 졸음 판별에 간섭 방지",
          6, "#FFF9C4", "#F9A825", "#F57F17")

    arrow(ax, 0.50, 0.39, 0.50, 0.375)

    # ═══ STAGE 5: LOSS & TRAINING ═══
    box(ax, 0.03, 0.275, 0.94, 0.09, "#F0F7FF", DBLUE, "", radius=0.01, lw=2)
    label(ax, 0.50, 0.36, "Stage 5:  Training Strategy", 11, DBLUE, True)

    box(ax, 0.05, 0.29, 0.28, 0.055, LBLUE, BLUE,
        "Uncertainty-Weighted MTL", 8.5, DBLUE, True,
        sub="L = Σ (1/2s^2_t)·L_t + log(s_t)\nKendall et al., CVPR 2018\nAuto-balancing task weights", sfs=6.5, radius=0.006, lw=1)

    box(ax, 0.35, 0.29, 0.28, 0.055, LBLUE, BLUE,
        "Loss Functions", 8.5, DBLUE, True,
        sub="Arousal/Valence: MSE Loss\nDrowsy: CrossEntropy Loss\n+ Focal Loss (γ=2) option", sfs=6.5, radius=0.006, lw=1)

    box(ax, 0.65, 0.29, 0.28, 0.055, LBLUE, BLUE,
        "K-EMocon Dataset", 8.5, DBLUE, True,
        sub="6-fold GroupKFold CV\nSubject-wise split\nPair-aware grouping", sfs=6.5, radius=0.006, lw=1)

    # ═══ STAGE 6: KD ═══
    arrow(ax, 0.50, 0.275, 0.50, 0.26)
    box(ax, 0.03, 0.155, 0.94, 0.10, "#E3F2FD", DBLUE, "", radius=0.01, lw=2)
    label(ax, 0.50, 0.25, "Stage 6:  Knowledge Distillation (Teacher → Student)", 11, DBLUE, True)

    box(ax, 0.05, 0.17, 0.25, 0.065, LBLUE, BLUE,
        "Teacher (~144K)", 9, DBLUE, True,
        sub="15-Token Pool-FFN + MHSA\nArousal UAR: 60.02%", sfs=7, radius=0.006, lw=1)

    # KD arrow
    ax.annotate("", xy=(0.38, 0.20), xytext=(0.30, 0.20),
                arrowprops=dict(arrowstyle="-|>", color=DBLUE, linewidth=2.5, mutation_scale=15), zorder=5)
    label(ax, 0.34, 0.215, "KD", 8, DBLUE, True)

    box(ax, 0.38, 0.17, 0.25, 0.065, LGREEN, GREEN,
        "Student (~12K)", 9, GREEN, True,
        sub="3 MLPs (Face/Bio/Audio → 64d)\n→ Concat → FFN → Heads", sfs=7, radius=0.006, lw=1)

    box(ax, 0.68, 0.17, 0.27, 0.065, LORANGE, ORANGE,
        "KD Loss (3-component)", 9, ORANGE, True,
        sub="α·L_task (MSE+CE)\n+ β·L_repr (CosineEmbedding)\n+ γ·L_logit (SmoothL1)", sfs=6.5, radius=0.006, lw=1)

    # ═══ PERFORMANCE ═══
    box(ax, 0.03, 0.06, 0.94, 0.08, "#E8F5E9", GREEN, "", radius=0.01, lw=2)
    label(ax, 0.50, 0.135, "Ablation Results (K-EMocon 6-fold CV)", 10, GREEN, True)

    perf = [
        (0.08, "LGBM\nBaseline", "56.47%", GRAY),
        (0.23, "3-Expert\nPool-FFN", "57.83%", BLUE),
        (0.38, "5-Expert\n15-Token", "59.14%", BLUE),
        (0.53, "Full\nKMERFusion", "60.02%", GREEN),
        (0.68, "KD Student\n(heavy)", "59.18%", ORANGE),
        (0.83, "Teacher\nFidelity", "98.6%", RED),
    ]
    for px, title, acc, c in perf:
        box(ax, px, 0.07, 0.12, 0.05, "#FFF", c,
            title, 7, c, True, sub=acc, sfs=10, radius=0.005, lw=1.2)

    # Key annotation
    box(ax, 0.10, 0.015, 0.80, 0.035, "#FFF9C4", "#F9A825", "", radius=0.008, lw=1.5)
    label(ax, 0.50, 0.035, "핵심:  15개 이종 토큰을 통합하는 Pool-FFN(intra) + MHSA(inter) 2단계 구조  |  "
          "Validity Masking으로 센서 결측 대응  |  Detached Drowsy Head", 7.5, "#F57F17", True)

    fig.savefig("/home/ajy/Jetson_thor/fig_kmer_fusion.png",
                dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.2)
    plt.close(fig)
    print("[OK] fig_kmer_fusion.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: End-to-End Pipeline (전체 시스템 흐름)
# ═══════════════════════════════════════════════════════════════
def fig3_e2e_pipeline():
    fig, ax = plt.subplots(figsize=(22, 10))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("auto"); ax.axis("off")
    fig.patch.set_facecolor(BG)

    # Title
    box(ax, 0.01, 0.92, 0.98, 0.065, NAVY, NAVY,
        "K-MER System:  End-to-End Real-Time Driver Monitoring Pipeline", 16, "#FFF", True, radius=0.01)
    label(ax, 0.50, 0.93, "Sensor Input → Expert Feature Extraction → Cross-Modal Fusion → Post-Processing → Vehicle Gateway", 9, "#BBCCDD")

    # ═══ SENSORS ═══
    box(ax, 0.01, 0.50, 0.12, 0.38, LBLUE, BLUE, "", radius=0.01, lw=2)
    label(ax, 0.07, 0.87, "Sensors", 12, DBLUE, True)

    sensors = [
        ("RGB Camera", "RealSense D435\n1280×720 @30fps"),
        ("E4 Wristband", "BVP/EDA/HR/Temp\n4Hz~64Hz"),
        ("Microphone", "16kHz Mono\n4-sec window"),
    ]
    for i, (name, desc) in enumerate(sensors):
        sy = 0.76 - i * 0.10
        box(ax, 0.02, sy, 0.10, 0.07, "#E1F5FE", BLUE,
            name, 8, DBLUE, True, sub=desc, sfs=6, radius=0.005, lw=1)

    # ═══ EXPERTS ═══
    arrow(ax, 0.13, 0.69, 0.15, 0.69, DARK, 2)

    box(ax, 0.15, 0.50, 0.22, 0.38, LORANGE, ORANGE, "", radius=0.01, lw=2)
    label(ax, 0.26, 0.87, "Expert Layer (Frozen)", 11, ORANGE, True)

    experts = [
        ("K-FER Expert", "MobileViTv2 + AU Cross-Attn\n→ 7-class probs + FACS + EAR\n5M params, d=384", 0.13),
        ("Audio Expert", "emotion2vec (9-class, 768d)\n+ audeering wav2vec2 (AVD 3d)\n+ Audio Quality (3d)", 0.10),
        ("Bio Expert", "neurokit2 features\nHRV(4d) + EDA(5d) + HR/Temp(6d)\n0 learnable params", 0.10),
    ]
    ey = 0.78
    for name, desc, eh in experts:
        box(ax, 0.16, ey, 0.20, eh, "#FFF8E1", ORANGE,
            name, 9, ORANGE, True, sub=desc, sfs=6, radius=0.006, lw=1)
        ey -= eh + 0.02

    # ═══ FUSION ═══
    arrow(ax, 0.37, 0.69, 0.39, 0.69, DARK, 2)

    box(ax, 0.39, 0.50, 0.22, 0.38, LGREEN, GREEN, "", radius=0.01, lw=2)
    label(ax, 0.50, 0.87, "K-MER Fusion", 11, GREEN, True)
    label(ax, 0.50, 0.85, "~144K params", 8, GRAY)

    fusion_steps = [
        ("15 Token Formation", "14 expert + 1 CLS → 64d"),
        ("Pool-FFN (Intra-modal)", "Face/Audio/Bio 내부 집약"),
        ("Global MHSA (Cross-modal)", "4 heads, d=64, validity mask"),
        ("CLS Pooling → 64d repr", "Cross-modal fused representation"),
        ("Output Heads", "Arousal / Valence / Drowsy"),
    ]
    fy = 0.79
    for name, desc in fusion_steps:
        box(ax, 0.40, fy, 0.20, 0.045, "#C8E6C9", GREEN,
            name, 7.5, GREEN, True, sub=desc, sfs=6, radius=0.005, lw=0.8)
        if fy > 0.56:
            arrow(ax, 0.50, fy, 0.50, fy - 0.005, GREEN, 0.8)
        fy -= 0.055

    # ═══ POST-PROCESSING ═══
    arrow(ax, 0.61, 0.69, 0.63, 0.69, DARK, 2)

    box(ax, 0.63, 0.50, 0.16, 0.38, LPURPLE, PURPLE, "", radius=0.01, lw=2)
    label(ax, 0.71, 0.87, "Post-Processing", 11, PURPLE, True)

    post_items = [
        ("Temporal Smoothing", "7-frame majority vote\n+ 4-strategy stabilizer"),
        ("Compound Emotion", "K-FER × Arousal\n→ 13 refined labels"),
        ("Status Flags", "Stress / Low Attn\nDrowsy / NegEmo"),
        ("Protocol Mapping", "7-class → 6 codes\n(sad+hurt 병합)"),
    ]
    py = 0.78
    for name, desc in post_items:
        box(ax, 0.64, py, 0.14, 0.06, "#E1BEE7", PURPLE,
            name, 8, PURPLE, True, sub=desc, sfs=6, radius=0.005, lw=0.8)
        py -= 0.075

    # ═══ GATEWAY ═══
    arrow(ax, 0.79, 0.69, 0.81, 0.69, DARK, 2)

    box(ax, 0.81, 0.50, 0.18, 0.38, LRED, RED, "", radius=0.01, lw=2)
    label(ax, 0.90, 0.87, "Gateway Output", 11, RED, True)

    box(ax, 0.82, 0.76, 0.16, 0.08, "#FFCDD2", RED,
        "8-Byte USB Packet", 9, RED, True,
        sub="[SOF][TYPE][SEQ][LEN]\n[EmoCode+Flags]\n[Intensity+NegEmo]\n[CRC8][EOF]", sfs=6, radius=0.006, lw=1)

    box(ax, 0.82, 0.64, 0.16, 0.08, "#FFCDD2", RED,
        "Emotion Code (6종)", 8, RED, True,
        sub="0=공포 1=놀람 2=분노\n3=슬픔/혐오 4=행복\n5=중립", sfs=6.5, radius=0.006, lw=1)

    box(ax, 0.82, 0.52, 0.16, 0.08, "#FFCDD2", RED,
        "Status Flags (4종)", 8, RED, True,
        sub="Stress | Low Attention\nDrowsy | NegEmo\n→ 차량 게이트웨이 전송", sfs=6.5, radius=0.006, lw=1)

    # ═══ BOTTOM: Technical specs ═══
    box(ax, 0.01, 0.02, 0.98, 0.45, "#FAFAFA", "#DDD", "", radius=0.01, lw=1)

    # KD section
    box(ax, 0.03, 0.30, 0.30, 0.15, "#E3F2FD", DBLUE, "", radius=0.008, lw=1.5)
    label(ax, 0.18, 0.44, "Knowledge Distillation", 10, DBLUE, True)
    label(ax, 0.18, 0.42, "Teacher (144K) → Student (12K)", 8, GRAY)
    label(ax, 0.18, 0.39, "L_KD = α·L_task + β·L_repr + γ·L_logit", 8, DARK)
    label(ax, 0.18, 0.37, "CosineEmbedding + SmoothL1", 7, GRAY)
    label(ax, 0.18, 0.34, "파라미터 12× 축소, 성능 98.6% 유지", 7.5, GREEN, True)

    # Gated Residual
    box(ax, 0.35, 0.30, 0.30, 0.15, "#FFF9C4", "#F9A825", "", radius=0.008, lw=1.5)
    label(ax, 0.50, 0.44, "Gated Residual (K-FER)", 10, "#F57F17", True)
    label(ax, 0.50, 0.42, "gate = σ(W · [global; au_attn])", 8, DARK)
    label(ax, 0.50, 0.39, "output = Q + gate ⊙ cross_attn_out", 8, DARK)
    label(ax, 0.50, 0.37, "Per-dim learnable, init=0.0", 7, GRAY)
    label(ax, 0.50, 0.34, "→ Global-Local 정보 균형 자동 학습", 7.5, "#F57F17", True)

    # Performance
    box(ax, 0.67, 0.30, 0.30, 0.15, "#E8F5E9", GREEN, "", radius=0.008, lw=1.5)
    label(ax, 0.82, 0.44, "Performance Summary", 10, GREEN, True)
    label(ax, 0.82, 0.41, "K-FER: 79.7% (single) → 98.8% (temporal)", 8, DARK)
    label(ax, 0.82, 0.39, "K-MER Fusion: 60.02% Arousal UAR", 8, DARK)
    label(ax, 0.82, 0.37, "KD Student: 59.18% (12K params)", 8, DARK)
    label(ax, 0.82, 0.34, "종합 10항목: 98.0% Macro Average", 9, GREEN, True)

    # Technical specs table
    specs = [
        ("Backbone", "MobileViTv2-100, 5M params, d=384"),
        ("AU Regions", "8 facial AUs, bilinear grid_sample"),
        ("Cross-Attention", "8 heads, d=384, gated residual"),
        ("K-MER Tokens", "15 tokens × 64d, 5 modality types"),
        ("Pool-FFN", "AvgPool1d + FFN(64→128→64), GELU"),
        ("Global MHSA", "4 heads, d=64, validity masking"),
        ("Temporal", "7-frame majority vote + 4-strategy"),
        ("Gateway", "8-byte USB, CRC8, 6 codes + 4 flags"),
    ]
    for i, (k, v) in enumerate(specs):
        col = i // 4
        row = i % 4
        sx = 0.04 + col * 0.48
        sy = 0.25 - row * 0.05
        label(ax, sx, sy, f"> {k}:", 7.5, DBLUE, True, ha="left")
        label(ax, sx + 0.12, sy, v, 7, DARK, ha="left")

    fig.savefig("/home/ajy/Jetson_thor/fig_e2e_pipeline.png",
                dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.2)
    plt.close(fig)
    print("[OK] fig_e2e_pipeline.png")


if __name__ == "__main__":
    fig1_au_cross_attention()
    fig2_kmer_fusion()
    fig3_e2e_pipeline()
    print("\n[DONE] All 3 technical figures generated.")
