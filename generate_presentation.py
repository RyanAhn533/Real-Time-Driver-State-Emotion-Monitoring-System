#!/usr/bin/env python3
"""
K-MER 시스템 교수님 발표용 PPT — 기술 파이프라인 중심
Top-down: 전체 파이프라인 → 각 모듈 내부 구조 → 기술적 혁신 포인트
기술 다이어그램 (fig_*.png) 포함
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# ── Colors ──
C_NAVY = RGBColor(0x1B, 0x3A, 0x5C)
C_BLUE = RGBColor(0x29, 0x80, 0xB9)
C_DBLUE = RGBColor(0x15, 0x65, 0xC0)
C_DARK = RGBColor(0x2C, 0x3E, 0x50)
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_LGRAY = RGBColor(0xF5, 0xF5, 0xF5)
C_GRAY = RGBColor(0x88, 0x88, 0x88)
C_GREEN = RGBColor(0x27, 0xAE, 0x60)
C_ORANGE = RGBColor(0xE6, 0x7E, 0x22)
C_RED = RGBColor(0xC6, 0x28, 0x28)
C_PURPLE = RGBColor(0x8E, 0x24, 0xAA)
C_ACCENT = RGBColor(0x34, 0x98, 0xDB)
C_GOLD = RGBColor(0xF5, 0x7F, 0x17)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
BASE = "/home/ajy/Jetson_thor"


def add_bg(slide, color=C_WHITE):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def shape(slide, left, top, width, height, fill_color, border_color=None, border_width=Pt(0)):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    s.fill.solid()
    s.fill.fore_color.rgb = fill_color
    if border_color:
        s.line.color.rgb = border_color
        s.line.width = border_width
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def text(slide, left, top, width, height, content, font_size=18,
         color=C_DARK, bold=False, alignment=PP_ALIGN.LEFT, font_name="맑은 고딕"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = content
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def multitext(slide, left, top, width, height, lines, font_name="맑은 고딕"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line_data in enumerate(lines):
        if isinstance(line_data, str):
            t, s, c, b, a = line_data, 14, C_DARK, False, PP_ALIGN.LEFT
        else:
            t = line_data[0]
            s = line_data[1] if len(line_data) > 1 else 14
            c = line_data[2] if len(line_data) > 2 else C_DARK
            b = line_data[3] if len(line_data) > 3 else False
            a = line_data[4] if len(line_data) > 4 else PP_ALIGN.LEFT
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = t
        p.font.size = Pt(s)
        p.font.color.rgb = c
        p.font.bold = b
        p.font.name = font_name
        p.alignment = a
        p.space_after = Pt(3)
    return txBox


def topbar(slide, title, subtitle=""):
    shape(slide, Inches(0), Inches(0), SLIDE_W, Inches(1.0), C_NAVY)
    text(slide, Inches(0.6), Inches(0.12), Inches(12), Inches(0.5),
         title, 26, C_WHITE, True)
    if subtitle:
        text(slide, Inches(0.6), Inches(0.58), Inches(12), Inches(0.3),
             subtitle, 13, RGBColor(0xBB, 0xCC, 0xDD))


def pagenum(slide, num, total):
    text(slide, Inches(12.3), Inches(7.05), Inches(0.8), Inches(0.3),
         f"{num}/{total}", 10, C_GRAY, False, PP_ALIGN.RIGHT)


def content_box(slide, left, top, width, height, title, lines,
                box_color=RGBColor(0xE8, 0xF4, 0xFD), border_color=C_BLUE, title_color=C_NAVY):
    shape(slide, left, top, width, height, box_color, border_color, Pt(1.5))
    text(slide, left + Inches(0.12), top + Inches(0.06), width - Inches(0.24), Inches(0.3),
         title, 15, title_color, True)
    multitext(slide, left + Inches(0.15), top + Inches(0.35),
              width - Inches(0.3), height - Inches(0.4), lines)


def img(slide, path, left, top, width=None, height=None):
    if os.path.exists(path):
        slide.shapes.add_picture(path, left, top, width, height)


TOTAL = 11


def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    n = 0

    # ═══════════════════════════════════════
    # 1. TITLE
    # ═══════════════════════════════════════
    n += 1
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(sl, C_NAVY)

    text(sl, Inches(1), Inches(1.2), Inches(11), Inches(1.0),
         "K-MER", 64, C_WHITE, True, PP_ALIGN.CENTER)
    text(sl, Inches(1), Inches(2.3), Inches(11), Inches(0.8),
         "Korean Multimodal Emotion Recognition\nfor Real-Time Driver Monitoring",
         28, RGBColor(0xBB, 0xDD, 0xFF), False, PP_ALIGN.CENTER)

    line = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.5), Inches(3.4), Inches(4.3), Pt(2))
    line.fill.solid(); line.fill.fore_color.rgb = C_ACCENT; line.line.fill.background()

    text(sl, Inches(1), Inches(3.7), Inches(11), Inches(0.5),
         "Technical Architecture & Pipeline", 22, RGBColor(0xAA, 0xBB, 0xCC), False, PP_ALIGN.CENTER)
    text(sl, Inches(1), Inches(4.4), Inches(11), Inches(0.4),
         "안준영  |  지도교수: 문연국  |  산업부 국책과제", 16, RGBColor(0x88, 0x99, 0xAA), False, PP_ALIGN.CENTER)

    # Bottom tech specs
    specs = [
        ("3 Modalities", "Vision + Audio + Bio"),
        ("~144K Params", "Lightweight Fusion"),
        ("98.0% Accuracy", "10 Recognition Items"),
        ("8-Byte Protocol", "Vehicle Gateway"),
    ]
    for i, (t, d) in enumerate(specs):
        bx = Inches(1.5 + i * 2.7)
        shape(sl, bx, Inches(5.3), Inches(2.3), Inches(0.9),
              RGBColor(0x23, 0x4E, 0x74), C_ACCENT, Pt(1))
        text(sl, bx, Inches(5.35), Inches(2.3), Inches(0.35),
             t, 14, C_WHITE, True, PP_ALIGN.CENTER)
        text(sl, bx, Inches(5.7), Inches(2.3), Inches(0.35),
             d, 11, RGBColor(0xBB, 0xCC, 0xDD), False, PP_ALIGN.CENTER)

    pagenum(sl, n, TOTAL)

    # ═══════════════════════════════════════
    # 2. E2E PIPELINE OVERVIEW
    # ═══════════════════════════════════════
    n += 1
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(sl)
    topbar(sl, "1. System Pipeline Overview", "Top-down: 전체 시스템을 한 장으로")
    pagenum(sl, n, TOTAL)

    img(sl, f"{BASE}/fig_e2e_pipeline.png", Inches(0.3), Inches(1.1), Inches(12.7), Inches(6.2))

    # ═══════════════════════════════════════
    # 3. K-FER ARCHITECTURE (기술 다이어그램)
    # ═══════════════════════════════════════
    n += 1
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(sl)
    topbar(sl, "2. K-FER Expert — Internal Architecture",
           "MobileViTv2 Backbone + AU RoI Cross-Attention + Gated Residual")
    pagenum(sl, n, TOTAL)

    img(sl, f"{BASE}/fig_kfer_architecture.png", Inches(0.3), Inches(1.1), Inches(5.8), Inches(6.2))

    # Right side: key technical points
    content_box(sl, Inches(6.4), Inches(1.2), Inches(6.5), Inches(2.5),
                "Core Innovation: Single-Forward AU Extraction",
                [("기존 방식 (POSTER++, DAtt-Net 등):", 13, C_RED, True),
                 ("  각 AU 영역마다 별도 backbone forward", 12, C_DARK, False),
                 ("  8 AU regions x backbone = 큰 연산량", 12, C_DARK, False),
                 ("", 6, C_DARK, False),
                 ("K-FER 방식:", 13, C_GREEN, True),
                 ("  1회 backbone forward -> feature map [B, 384, 7, 7]", 12, C_DARK, False),
                 ("  F.grid_sample(bilinear) 로 AU 영역 직접 추출", 12, C_DARK, False),
                 ("  연산량 8배 절감, 동일 성능", 12, C_GREEN, True)],
                RGBColor(0xE8, 0xF4, 0xFD), C_BLUE, C_DBLUE)

    content_box(sl, Inches(6.4), Inches(3.9), Inches(6.5), Inches(1.7),
                "Gated Residual Connection",
                [("gate = sigmoid(W . [global; au_attn])", 13, C_GOLD, True),
                 ("output = Q + gate * cross_attn_out", 13, C_DARK, False),
                 ("", 6, C_DARK, False),
                 ("Per-dimension learnable gate (384d), init=0.0", 12, C_DARK, False),
                 ("-> 학습 초기: gate~0 (global 유지)", 12, C_GRAY, False),
                 ("-> 학습 후: AU 정보 점진적 반영", 12, C_GRAY, False)],
                RGBColor(0xFF, 0xF9, 0xC4), C_GOLD, C_GOLD)

    content_box(sl, Inches(6.4), Inches(5.8), Inches(6.5), Inches(1.3),
                "Performance",
                [("Dataset: AI Hub 한국인 감정인식 (413K/52K)", 13, C_DARK, False),
                 ("Single-frame: 79.7% (Macro F1: 0.7953)", 13, C_DARK, False),
                 ("+ Temporal (W=7): 98.8% (6-class)", 14, C_GREEN, True)],
                RGBColor(0xE8, 0xF5, 0xE9), C_GREEN, C_GREEN)

    # ═══════════════════════════════════════
    # 4. CROSS-ATTENTION DETAIL
    # ═══════════════════════════════════════
    n += 1
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(sl)
    topbar(sl, "3. Cross-Attention Fusion Layer — Detail",
           "Pre-Norm Architecture: Cross-Attention -> Self-Attention -> FFN")
    pagenum(sl, n, TOTAL)

    # Cross-Attention flow diagram
    stages = [
        ("Input Tokens\n[B, 10, 384]", "CLS + Global + 8 AU", RGBColor(0xE8, 0xF4, 0xFD), C_BLUE),
        ("Cross-Attention\n(Q: CLS,Global | KV: AUs)", "8 heads, d_k=48\n+ Gated Residual", RGBColor(0xC8, 0xE6, 0xC9), C_GREEN),
        ("Self-Attention\n(All 10 tokens)", "8 heads, d=384\n+ Standard Residual", RGBColor(0xE8, 0xF5, 0xE9), C_GREEN),
        ("FFN Block", "LN -> Linear(384->1536)\n-> GELU -> Linear(1536->384)\n+ Residual", RGBColor(0xFF, 0xF3, 0xE0), C_ORANGE),
        ("CLS Token\n[B, 384]", "-> FER Head\n-> 7-class logits", RGBColor(0xE1, 0xBE, 0xE7), C_PURPLE),
    ]

    for i, (title, desc, bg, border) in enumerate(stages):
        bx = Inches(0.4 + i * 2.55)
        shape(sl, bx, Inches(1.3), Inches(2.2), Inches(2.0), bg, border, Pt(2))
        text(sl, bx, Inches(1.4), Inches(2.2), Inches(0.7), title, 14, border, True, PP_ALIGN.CENTER)
        text(sl, bx, Inches(2.1), Inches(2.2), Inches(1.0), desc, 11, C_DARK, False, PP_ALIGN.CENTER)
        if i < len(stages) - 1:
            arr = sl.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                      Inches(2.55 + i * 2.55), Inches(2.15), Inches(0.4), Inches(0.2))
            arr.fill.solid(); arr.fill.fore_color.rgb = C_GRAY; arr.line.fill.background()

    # Gated Residual Detail
    content_box(sl, Inches(0.4), Inches(3.6), Inches(6.0), Inches(3.5),
                "Gated Residual Mechanism (Key Innovation)",
                [("Cross-first, Self-second 설계:", 14, C_DBLUE, True),
                 ("", 4, C_DARK, False),
                 ("1) Cross-Attention:", 13, C_DARK, True),
                 ("   Q = [CLS, Global] (2 tokens)", 12, C_DARK, False),
                 ("   K, V = [AU1, AU2, ..., AU8] (8 tokens)", 12, C_DARK, False),
                 ("   -> CLS/Global이 AU 지역 정보를 선택적 수집", 12, C_GREEN, False),
                 ("", 4, C_DARK, False),
                 ("2) Gated Residual:", 13, C_DARK, True),
                 ("   gate = sigmoid(W_gate)  (384d, init=0.0)", 12, C_DARK, False),
                 ("   -> init=0.0 이므로 sigmoid(0)=0.5", 12, C_DARK, False),
                 ("   -> 학습 진행하면서 차원별 최적 gate 학습", 12, C_DARK, False),
                 ("   -> global과 local AU 정보 자동 균형", 12, C_GREEN, False),
                 ("", 4, C_DARK, False),
                 ("3) Self-Attention:", 13, C_DARK, True),
                 ("   모든 10개 토큰 간 전역 정보 교환", 12, C_DARK, False),
                 ("   Standard residual (gate 없음)", 12, C_DARK, False)],
                RGBColor(0xFF, 0xF9, 0xC4), C_GOLD, C_GOLD)

    # FER Head Detail
    content_box(sl, Inches(6.8), Inches(3.6), Inches(6.0), Inches(3.5),
                "Classification Head & Output",
                [("FER Head:", 14, C_PURPLE, True),
                 ("  CLS token [B, 384]", 12, C_DARK, False),
                 ("  -> LayerNorm(384)", 12, C_DARK, False),
                 ("  -> Linear(384, 384) + GELU", 12, C_DARK, False),
                 ("  -> Dropout(0.2)", 12, C_DARK, False),
                 ("  -> Linear(384, 7)", 12, C_DARK, False),
                 ("  -> 7-class logits", 12, C_DARK, False),
                 ("", 6, C_DARK, False),
                 ("Output (per frame):", 14, C_DBLUE, True),
                 ("  7-class emotion probs (softmax)", 12, C_DARK, False),
                 ("  FACS 6d (AU activation scores)", 12, C_DARK, False),
                 ("  EAR (Eye Aspect Ratio)", 12, C_DARK, False),
                 ("  PERCLOS (눈감은 시간 비율)", 12, C_DARK, False),
                 ("", 6, C_DARK, False),
                 ("Training:", 14, C_DBLUE, True),
                 ("  Focal Loss (gamma=2) + Label Smoothing (0.1)", 12, C_DARK, False),
                 ("  AdamW, lr=3e-4, CosineAnnealing", 12, C_DARK, False)],
                RGBColor(0xF3, 0xE5, 0xF5), C_PURPLE, C_PURPLE)

    # ═══════════════════════════════════════
    # 5. AUDIO & BIO EXPERTS
    # ═══════════════════════════════════════
    n += 1
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(sl)
    topbar(sl, "4. Audio Expert & Bio Expert", "Frozen Feature Extractors: emotion2vec + neurokit2")
    pagenum(sl, n, TOTAL)

    content_box(sl, Inches(0.3), Inches(1.2), Inches(6.2), Inches(3.0),
                "Audio Expert Pipeline",
                [("emotion2vec (ACL 2024, FunASR):", 13, C_BLUE, True),
                 ("  Waveform 16kHz -> 1024d embedding -> 9-class probs", 12, C_DARK, False),
                 ("  Angry, Disgusted, Fearful, Happy, Neutral,", 11, C_GRAY, False),
                 ("  Other, Sad, Surprised, Unknown", 11, C_GRAY, False),
                 ("", 4, C_DARK, False),
                 ("audeering wav2vec2:", 13, C_BLUE, True),
                 ("  wav2vec2 (1024d) -> Dense(1024,1024) + Tanh", 12, C_DARK, False),
                 ("  -> Linear(1024, 3): Arousal / Valence / Dominance", 12, C_DARK, False),
                 ("", 4, C_DARK, False),
                 ("Audio Quality Gate (3d):", 13, C_BLUE, True),
                 ("  RMS energy, Voicing ratio, SNR estimation", 12, C_DARK, False),
                 ("  품질 미달 시 validity mask = 0", 12, C_RED, False)],
                RGBColor(0xE8, 0xF4, 0xFD), C_BLUE, C_DBLUE)

    content_box(sl, Inches(6.8), Inches(1.2), Inches(6.2), Inches(3.0),
                "Bio Expert Pipeline (neurokit2)",
                [("BVP -> HRV Features (4d):", 13, C_GREEN, True),
                 ("  ppg_clean -> ppg_findpeaks -> IBI analysis", 12, C_DARK, False),
                 ("  mean_hr, sdnn, rmssd, lf_hf_ratio", 12, C_DARK, False),
                 ("", 4, C_DARK, False),
                 ("EDA -> SCR Features (5d):", 13, C_GREEN, True),
                 ("  mean_scl, std_scl, n_peaks, mean_amp, auc", 12, C_DARK, False),
                 ("  교감신경 활성화 -> 스트레스 지표", 12, C_GRAY, False),
                 ("", 4, C_DARK, False),
                 ("HR/Temperature Features (6d):", 13, C_GREEN, True),
                 ("  mean, std, range, slope 등", 12, C_DARK, False),
                 ("", 4, C_DARK, False),
                 ("0 learnable params (hand-crafted features)", 13, C_ORANGE, True)],
                RGBColor(0xE8, 0xF5, 0xE9), C_GREEN, C_GREEN)

    # Bottom: Why these experts
    content_box(sl, Inches(0.3), Inches(4.4), Inches(12.7), Inches(2.8),
                "Expert 설계 원리 (Design Rationale)",
                [("Frozen Expert 전략:  사전학습 모델 고정 + 경량 projection만 학습", 14, C_DBLUE, True),
                 ("", 4, C_DARK, False),
                 ("  1. 데이터 효율성: K-EMocon (소규모) 데이터로도 안정적 학습 가능", 13, C_DARK, False),
                 ("  2. 도메인 지식 보존: emotion2vec(ACL2024), neurokit2(생리학) 등 전문 지식 유지", 13, C_DARK, False),
                 ("  3. 연산 효율: Expert 출력 캐싱 가능 -> Fusion만 반복 학습", 13, C_DARK, False),
                 ("  4. 모듈성: 새 센서/모델 추가 시 token만 추가하면 됨 (plug-and-play)", 13, C_DARK, False),
                 ("", 4, C_DARK, False),
                 ("총 Expert 출력: Face 12d + Audio 15d + Bio 18d + Aux 11d + Meta 3d = 59d -> 15 tokens x 64d", 13, C_ORANGE, True)],
                RGBColor(0xF0, 0xF7, 0xFF), C_ACCENT, C_ACCENT)

    # ═══════════════════════════════════════
    # 6. K-MER FUSION (기술 다이어그램)
    # ═══════════════════════════════════════
    n += 1
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(sl)
    topbar(sl, "5. K-MER Fusion — Internal Architecture",
           "15 Tokens x 64d -> Pool-FFN (Intra-modal) -> MHSA (Cross-modal) -> Output Heads")
    pagenum(sl, n, TOTAL)

    img(sl, f"{BASE}/fig_kmer_fusion.png", Inches(0.2), Inches(1.1), Inches(13.0), Inches(6.2))

    # ═══════════════════════════════════════
    # 7. FUSION DETAIL
    # ═══════════════════════════════════════
    n += 1
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(sl)
    topbar(sl, "6. K-MER Fusion — Technical Details",
           "Pool-FFN, MHSA, Validity Masking, Detached Drowsy Head")
    pagenum(sl, n, TOTAL)

    # Pool-FFN
    content_box(sl, Inches(0.3), Inches(1.2), Inches(4.1), Inches(3.0),
                "Pool-FFN (EfficientFormer-style)",
                [("같은 모달리티 토큰끼리 정보 집약:", 13, C_ORANGE, True),
                 ("", 4, C_DARK, False),
                 ("  AvgPool1d(kernel=k, stride=1, pad=k//2)", 12, C_DARK, False),
                 ("  -> 인접 토큰 평균으로 local 패턴 포착", 12, C_DARK, False),
                 ("", 4, C_DARK, False),
                 ("  Residual:", 12, C_DARK, True),
                 ("    LayerNorm(64)", 11, C_DARK, False),
                 ("    Linear(64 -> 128) + GELU", 11, C_DARK, False),
                 ("    Linear(128 -> 64)", 11, C_DARK, False),
                 ("    + Skip connection", 11, C_DARK, False),
                 ("", 4, C_DARK, False),
                 ("Face k=3, Audio k=3, Bio k=4", 12, C_ORANGE, True)],
                RGBColor(0xFF, 0xF3, 0xE0), C_ORANGE, C_ORANGE)

    # MHSA
    content_box(sl, Inches(4.6), Inches(1.2), Inches(4.1), Inches(3.0),
                "Global MHSA (Cross-modal)",
                [("모든 15개 토큰 간 cross-modal attention:", 13, C_GREEN, True),
                 ("", 4, C_DARK, False),
                 ("  Multi-Head Self-Attention:", 12, C_DARK, True),
                 ("    4 heads, d_k = 64/4 = 16", 12, C_DARK, False),
                 ("    Pre-Norm (LayerNorm before attention)", 12, C_DARK, False),
                 ("    + Standard Residual", 12, C_DARK, False),
                 ("", 4, C_DARK, False),
                 ("  FFN:", 12, C_DARK, True),
                 ("    LayerNorm -> Linear(64->128)", 12, C_DARK, False),
                 ("    -> GELU -> Dropout(0.1)", 12, C_DARK, False),
                 ("    -> Linear(128->64) -> Residual", 12, C_DARK, False)],
                RGBColor(0xE8, 0xF5, 0xE9), C_GREEN, C_GREEN)

    # Validity Masking + Detached Head
    content_box(sl, Inches(8.9), Inches(1.2), Inches(4.1), Inches(3.0),
                "Key Technical Points",
                [("Validity Masking:", 13, C_RED, True),
                 ("  key_padding_mask로 MHSA에 적용", 12, C_DARK, False),
                 ("  센서 미연결/품질미달 -> 해당 토큰 masking", 12, C_DARK, False),
                 ("  -> 결측 데이터에도 robust한 추론", 12, C_DARK, False),
                 ("", 6, C_DARK, False),
                 ("Detached Drowsy Head:", 13, C_RED, True),
                 ("  input = concat([fused.detach(), perclos])", 12, C_DARK, False),
                 ("  .detach() -> gradient 차단", 12, C_DARK, False),
                 ("  -> 감정 loss가 졸음 판별에 간섭 방지", 12, C_DARK, False),
                 ("  -> Multi-task 학습 안정화", 12, C_GREEN, False)],
                RGBColor(0xFF, 0xEB, 0xEE), C_RED, C_RED)

    # Output Heads
    content_box(sl, Inches(0.3), Inches(4.4), Inches(6.2), Inches(2.8),
                "Output Heads (3 Tasks)",
                [("Arousal Head: Linear(64->32) + ReLU + Dropout(0.1) + Linear(32->1) + Sigmoid", 12, C_DARK, False),
                 ("  -> arousal in [0, 1]", 11, C_GRAY, False),
                 ("", 4, C_DARK, False),
                 ("Valence Head: Linear(64->32) + ReLU + Dropout(0.1) + Linear(32->1) + Sigmoid", 12, C_DARK, False),
                 ("  -> valence in [0, 1]", 11, C_GRAY, False),
                 ("", 4, C_DARK, False),
                 ("Drowsy Head: Linear(66->32) + ReLU + Dropout(0.1) + Linear(32->3)", 12, C_DARK, False),
                 ("  Input: [fused_repr(64d, detached) + perclos_ear(2d)] = 66d", 11, C_GRAY, False),
                 ("  -> 3-class logits: alert / drowsy / sleeping", 11, C_GRAY, False)],
                RGBColor(0xF3, 0xE5, 0xF5), C_PURPLE, C_PURPLE)

    # Training
    content_box(sl, Inches(6.8), Inches(4.4), Inches(6.0), Inches(2.8),
                "Training Strategy",
                [("Uncertainty-Weighted MTL (Kendall, CVPR 2018):", 13, C_DBLUE, True),
                 ("  L = SUM (1/2*s_t^2) * L_t + log(s_t)", 12, C_DARK, False),
                 ("  s_t: learnable log-variance per task", 12, C_GRAY, False),
                 ("  -> task weight 자동 학습", 12, C_DARK, False),
                 ("", 4, C_DARK, False),
                 ("Dataset: K-EMocon", 13, C_DBLUE, True),
                 ("  32명 참가자 다자간 대화", 12, C_DARK, False),
                 ("  6-fold GroupKFold CV (subject-wise)", 12, C_DARK, False),
                 ("  Pair-aware grouping (대화 파트너 분리)", 12, C_DARK, False),
                 ("", 4, C_DARK, False),
                 ("Total params: ~144K (teacher) / ~12K (student)", 13, C_GREEN, True)],
                RGBColor(0xE3, 0xF2, 0xFD), C_DBLUE, C_DBLUE)

    # ═══════════════════════════════════════
    # 8. POST-PROCESSING
    # ═══════════════════════════════════════
    n += 1
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(sl)
    topbar(sl, "7. Post-Processing Pipeline",
           "Temporal Smoothing + Compound Emotion + Status Flags + Protocol Mapping")
    pagenum(sl, n, TOTAL)

    content_box(sl, Inches(0.3), Inches(1.2), Inches(4.1), Inches(5.8),
                "StableEmotionDetector (4-Strategy)",
                [("1) 7-frame Majority Vote:", 13, C_PURPLE, True),
                 ("   최근 7프레임 투표 -> 최빈 감정", 12, C_DARK, False),
                 ("   감정 그룹별 차등 threshold:", 12, C_DARK, False),
                 ("     Sustained (neutral, happy, sad): 높은 기준", 11, C_GRAY, False),
                 ("     Transient (angry, anxious...): 낮은 기준", 11, C_GRAY, False),
                 ("", 4, C_DARK, False),
                 ("2) Confidence Threshold (0.5):", 13, C_PURPLE, True),
                 ("   softmax prob < 0.5 -> 이전 감정 유지", 12, C_DARK, False),
                 ("   3프레임 연속 미달 -> alert OFF", 12, C_DARK, False),
                 ("", 4, C_DARK, False),
                 ("3) EMA Smoothing (alpha=0.3):", 13, C_PURPLE, True),
                 ("   new = 0.3*raw + 0.7*prev", 12, C_DARK, False),
                 ("   EMA 최대 감정 = vote 최대 감정 검증", 12, C_DARK, False),
                 ("", 4, C_DARK, False),
                 ("4) Min-Hold Duration (0.5s):", 13, C_PURPLE, True),
                 ("   감정 전환 후 0.5초 유지 보장", 12, C_DARK, False),
                 ("   급격한 깜빡임 방지", 12, C_DARK, False),
                 ("", 6, C_DARK, False),
                 ("효과: 79.7% -> 98.8% (운용 정확도)", 14, C_GREEN, True)],
                RGBColor(0xF3, 0xE5, 0xF5), C_PURPLE, C_PURPLE)

    content_box(sl, Inches(4.6), Inches(1.2), Inches(4.1), Inches(2.8),
                "Compound Emotion (13 labels)",
                [("K-FER 7-class x Arousal level -> 13 labels:", 13, C_BLUE, True),
                 ("", 4, C_DARK, False),
                 ("  happy + high arousal -> excited", 12, C_DARK, False),
                 ("  happy + low arousal -> calm_happy", 12, C_DARK, False),
                 ("  angry + high arousal -> enraged", 12, C_DARK, False),
                 ("  sad + high arousal -> distressed", 12, C_DARK, False),
                 ("  neutral + any -> neutral", 12, C_DARK, False),
                 ("  ...", 12, C_GRAY, False),
                 ("", 4, C_DARK, False),
                 ("Russell Circumplex Model 기반", 12, C_GRAY, False),
                 ("이산 감정 + 연속 각성도 결합", 12, C_GRAY, False)],
                RGBColor(0xE8, 0xF4, 0xFD), C_BLUE, C_DBLUE)

    content_box(sl, Inches(4.6), Inches(4.2), Inches(4.1), Inches(2.8),
                "Protocol Mapping (7 -> 6 codes)",
                [("K-FER 7-class -> Gateway 6 codes:", 13, C_ORANGE, True),
                 ("", 4, C_DARK, False),
                 ("  Code 0: 공포 <- anxious (class 4)", 12, C_DARK, False),
                 ("  Code 1: 놀람 <- surprised (class 6)", 12, C_DARK, False),
                 ("  Code 2: 분노 <- angry (class 2)", 12, C_DARK, False),
                 ("  Code 3: 슬픔/혐오 <- sad+hurt (class 3,5)", 12, C_RED, True),
                 ("  Code 4: 행복 <- happy (class 1)", 12, C_DARK, False),
                 ("  Code 5: 중립 <- neutral (class 0)", 12, C_DARK, False),
                 ("", 4, C_DARK, False),
                 ("sad+hurt 병합 근거: 혼동행렬 32.7% confusion", 12, C_GRAY, False)],
                RGBColor(0xFF, 0xF3, 0xE0), C_ORANGE, C_ORANGE)

    content_box(sl, Inches(8.9), Inches(1.2), Inches(4.1), Inches(5.8),
                "Status Flag Detector (4 Flags)",
                [("Flag 1 - Stress:", 14, C_RED, True),
                 ("  arousal > 0.6 AND 부정감정", 12, C_DARK, False),
                 ("  (angry, anxious, sad, hurt)", 11, C_GRAY, False),
                 ("  정확도: 95.0%", 12, C_GREEN, False),
                 ("", 6, C_DARK, False),
                 ("Flag 2 - Low Attention:", 14, C_RED, True),
                 ("  PERCLOS in [0.2, 0.4)", 12, C_DARK, False),
                 ("  눈 감은 비율 중간 -> 주의분산", 11, C_GRAY, False),
                 ("  정확도: 95.0%", 12, C_GREEN, False),
                 ("", 6, C_DARK, False),
                 ("Flag 3 - Drowsy:", 14, C_RED, True),
                 ("  PERCLOS >= 0.4 (NHTSA 기준)", 12, C_DARK, False),
                 ("  졸음 경고", 11, C_GRAY, False),
                 ("  정확도: 96.0%", 12, C_GREEN, False),
                 ("", 6, C_DARK, False),
                 ("Flag 4 - Negative Emotion:", 14, C_RED, True),
                 ("  K-FER in {angry, anxious, sad, hurt}", 12, C_DARK, False),
                 ("  이진 부정감정 판별", 11, C_GRAY, False),
                 ("  정확도: 99.8%", 12, C_GREEN, False)],
                RGBColor(0xFF, 0xEB, 0xEE), C_RED, C_RED)

    # ═══════════════════════════════════════
    # 9. GATEWAY PACKET
    # ═══════════════════════════════════════
    n += 1
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(sl)
    topbar(sl, "8. Gateway Packet Protocol", "8-Byte USB Packet -> Vehicle Gateway")
    pagenum(sl, n, TOTAL)

    # Packet bytes
    byte_info = [
        ("Byte 0", "0xAA", "SOF", RGBColor(0xFF, 0xCD, 0xD2)),
        ("Byte 1", "0x01", "TYPE", RGBColor(0xFF, 0xCD, 0xD2)),
        ("Byte 2", "0~255", "SEQ", RGBColor(0xFF, 0xCD, 0xD2)),
        ("Byte 3", "0x02", "LEN", RGBColor(0xFF, 0xCD, 0xD2)),
        ("Byte 4", "Payload", "EmoCode(4b)\n+Flags(4b)", RGBColor(0xEF, 0x9A, 0x9A)),
        ("Byte 5", "Payload", "Intensity(6b)\n+NegEmo(1b)", RGBColor(0xEF, 0x9A, 0x9A)),
        ("Byte 6", "CRC8", "B1~B5", RGBColor(0xFF, 0xCD, 0xD2)),
        ("Byte 7", "0xFE", "EOF", RGBColor(0xFF, 0xCD, 0xD2)),
    ]

    for i, (label_t, value, desc, bc) in enumerate(byte_info):
        bx = Inches(0.4 + i * 1.55)
        shape(sl, bx, Inches(1.3), Inches(1.3), Inches(1.6), bc, C_RED, Pt(1.5))
        text(sl, bx, Inches(1.35), Inches(1.3), Inches(0.25), label_t, 11, C_RED, True, PP_ALIGN.CENTER)
        text(sl, bx, Inches(1.65), Inches(1.3), Inches(0.3), value, 14, C_DARK, True, PP_ALIGN.CENTER)
        text(sl, bx, Inches(2.05), Inches(1.3), Inches(0.7), desc, 10, C_DARK, False, PP_ALIGN.CENTER)

    # Byte 4 detail
    content_box(sl, Inches(0.3), Inches(3.2), Inches(6.2), Inches(3.8),
                "Byte 4: Emotion Code (4 bits) + Status Flags (4 bits)",
                [("Upper 4 bits - Emotion Code:", 13, C_DBLUE, True),
                 ("  Bit[7:4] = 0~5 (6 emotion codes)", 12, C_DARK, False),
                 ("  0=Fear, 1=Surprise, 2=Anger", 12, C_DARK, False),
                 ("  3=Sad/Disgust, 4=Happy, 5=Neutral", 12, C_DARK, False),
                 ("", 4, C_DARK, False),
                 ("Lower 4 bits - Status Flags:", 13, C_DBLUE, True),
                 ("  Bit[3] = Stress flag", 12, C_DARK, False),
                 ("  Bit[2] = Low Attention flag", 12, C_DARK, False),
                 ("  Bit[1] = Drowsy flag", 12, C_DARK, False),
                 ("  Bit[0] = END marker (reserved)", 12, C_GRAY, False),
                 ("", 6, C_DARK, False),
                 ("Example: happy + drowsy", 13, C_ORANGE, True),
                 ("  Emotion=4 (0100), Flags=0010 (drowsy)", 12, C_DARK, False),
                 ("  -> Byte 4 = 0100_0010 = 0x42", 12, C_GREEN, True)],
                RGBColor(0xE8, 0xF4, 0xFD), C_BLUE, C_DBLUE)

    # Byte 5 detail
    content_box(sl, Inches(6.8), Inches(3.2), Inches(6.0), Inches(3.8),
                "Byte 5: Intensity (6 bits) + NegEmo (1 bit)",
                [("Bits [7:5] - Emotion Intensity (3 bits):", 13, C_DBLUE, True),
                 ("  K-FER softmax max prob -> 0~7 quantize", 12, C_DARK, False),
                 ("  intensity = min(7, int(prob * 8))", 12, C_DARK, False),
                 ("", 4, C_DARK, False),
                 ("Bits [4:2] - State Intensity (3 bits):", 13, C_DBLUE, True),
                 ("  Arousal level -> 0~7 quantize", 12, C_DARK, False),
                 ("  state_intensity = min(7, int(arousal * 8))", 12, C_DARK, False),
                 ("", 4, C_DARK, False),
                 ("Bit [1] - Negative Emotion Flag:", 13, C_RED, True),
                 ("  K-FER in {angry, anxious, sad, hurt} -> 1", 12, C_DARK, False),
                 ("  NegativeEmotionDetector class", 12, C_DARK, False),
                 ("", 4, C_DARK, False),
                 ("Bit [0] - Reserved:", 13, C_GRAY, False),
                 ("  Future use", 12, C_GRAY, False),
                 ("", 4, C_DARK, False),
                 ("CRC8: XOR-shift over Byte 1~5", 13, C_DBLUE, True)],
                RGBColor(0xFF, 0xF3, 0xE0), C_ORANGE, C_ORANGE)

    # ═══════════════════════════════════════
    # 10. KNOWLEDGE DISTILLATION
    # ═══════════════════════════════════════
    n += 1
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(sl)
    topbar(sl, "9. Knowledge Distillation & Results",
           "Teacher (144K) -> Student (12K): 파라미터 12x 축소")
    pagenum(sl, n, TOTAL)

    content_box(sl, Inches(0.3), Inches(1.2), Inches(4.0), Inches(3.0),
                "Teacher Model (~144K)",
                [("15 Tokens x 64d", 14, C_DBLUE, True),
                 ("Pool-FFN (3 modalities)", 13, C_DARK, False),
                 ("Global MHSA (4 heads)", 13, C_DARK, False),
                 ("3 Output Heads", 13, C_DARK, False),
                 ("", 6, C_DARK, False),
                 ("Arousal UAR: 60.02%", 16, C_GREEN, True)],
                RGBColor(0xE8, 0xF4, 0xFD), C_BLUE, C_DBLUE)

    # Arrow
    arr = sl.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                               Inches(4.5), Inches(2.4), Inches(0.8), Inches(0.4))
    arr.fill.solid(); arr.fill.fore_color.rgb = C_ACCENT; arr.line.fill.background()
    text(sl, Inches(4.35), Inches(1.9), Inches(1.1), Inches(0.4), "KD", 16, C_ACCENT, True, PP_ALIGN.CENTER)

    content_box(sl, Inches(5.5), Inches(1.2), Inches(4.0), Inches(3.0),
                "Student Model (~12K)",
                [("Face MLP: 12d -> 64d", 13, C_DARK, False),
                 ("Bio MLP: 18d -> 64d", 13, C_DARK, False),
                 ("Audio MLP: 15d -> 64d", 13, C_DARK, False),
                 ("Concat(192d) -> FFN -> Heads", 13, C_DARK, False),
                 ("", 6, C_DARK, False),
                 ("Arousal UAR: 59.18%", 16, C_GREEN, True)],
                RGBColor(0xE8, 0xF5, 0xE9), C_GREEN, C_GREEN)

    content_box(sl, Inches(9.7), Inches(1.2), Inches(3.3), Inches(3.0),
                "KD Loss",
                [("L_KD = a*L_task + b*L_repr + c*L_logit", 12, C_DARK, True),
                 ("", 4, C_DARK, False),
                 ("L_task: MSE + CrossEntropy", 11, C_DARK, False),
                 ("  Student 자체 task loss", 10, C_GRAY, False),
                 ("", 4, C_DARK, False),
                 ("L_repr: CosineEmbeddingLoss", 11, C_DARK, False),
                 ("  Teacher-Student repr 거리", 10, C_GRAY, False),
                 ("", 4, C_DARK, False),
                 ("L_logit: SmoothL1Loss", 11, C_DARK, False),
                 ("  Teacher-Student output 매칭", 10, C_GRAY, False)],
                RGBColor(0xFF, 0xF3, 0xE0), C_ORANGE, C_ORANGE)

    # Results table
    content_box(sl, Inches(0.3), Inches(4.4), Inches(12.7), Inches(2.8),
                "10 Recognition Items - Combined Accuracy: 98.0%",
                [("Emotion (6):  공포 97.9% | 놀람 99.4% | 분노 100% | 슬픔/혐오 95.7% | 행복 100% | 중립 100%", 14, C_DBLUE, True),
                 ("Status (4):   스트레스 95.0% | 주의분산 95.0% | 졸음 96.0% | 부정감정 99.8%", 14, C_PURPLE, True),
                 ("", 6, C_DARK, False),
                 ("Macro Average: 98.0%  (6 Emotion avg: 98.8% + 4 Status avg: 96.4%)", 16, C_GREEN, True),
                 ("", 4, C_DARK, False),
                 ("Ablation: LGBM 56.47% -> 3-Expert 57.83% -> 5-Expert 59.14% -> Full KMERFusion 60.02%", 13, C_DARK, False),
                 ("KD Student: 59.18% Arousal UAR (Teacher fidelity 98.6%)", 13, C_DARK, False)],
                RGBColor(0xE8, 0xF5, 0xE9), C_GREEN, C_GREEN)

    # ═══════════════════════════════════════
    # 11. SUMMARY
    # ═══════════════════════════════════════
    n += 1
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(sl, C_NAVY)
    pagenum(sl, n, TOTAL)

    text(sl, Inches(1), Inches(0.4), Inches(11), Inches(0.7),
         "Technical Summary", 32, C_WHITE, True, PP_ALIGN.CENTER)

    innovations = [
        ("Single-Forward\nAU Extraction", "Feature map level\ngrid_sample\n8x compute savings", C_BLUE),
        ("Gated Residual\nCross-Attention", "Per-dim learnable gate\ninit=0.0\nGlobal-Local balance", C_GOLD),
        ("15-Token\nPool-FFN + MHSA", "Intra-modal pooling\nCross-modal attention\nValidity masking", C_GREEN),
        ("Detached\nDrowsy Head", "Gradient isolation\nMulti-task stability\n.detach() design", C_PURPLE),
        ("KD Pipeline\n144K -> 12K", "3-component loss\n12x compression\n98.6% fidelity", C_ACCENT),
    ]

    for i, (title, desc, color) in enumerate(innovations):
        bx = Inches(0.5 + i * 2.5)
        shape(sl, bx, Inches(1.3), Inches(2.2), Inches(2.2),
              RGBColor(0x23, 0x4E, 0x74), color, Pt(2))
        text(sl, bx, Inches(1.5), Inches(2.2), Inches(0.7),
             title, 15, C_WHITE, True, PP_ALIGN.CENTER)
        text(sl, bx, Inches(2.3), Inches(2.2), Inches(1.0),
             desc, 11, RGBColor(0xBB, 0xCC, 0xDD), False, PP_ALIGN.CENTER)

    # Pipeline
    text(sl, Inches(0.5), Inches(3.8), Inches(12), Inches(0.4),
         "System Performance", 20, C_WHITE, True, PP_ALIGN.CENTER)

    perf_items = [
        ("K-FER", "79.7% -> 98.8%\n(temporal)", C_BLUE),
        ("K-MER Fusion", "60.02%\nArousal UAR", C_GREEN),
        ("KD Student", "59.18%\n12K params", C_ORANGE),
        ("Combined", "98.0%\n10 items", C_GREEN),
    ]
    for i, (t, d, c) in enumerate(perf_items):
        bx = Inches(1.0 + i * 2.9)
        shape(sl, bx, Inches(4.3), Inches(2.5), Inches(1.3),
              RGBColor(0x23, 0x4E, 0x74), c, Pt(1.5))
        text(sl, bx, Inches(4.4), Inches(2.5), Inches(0.3),
             t, 14, c, True, PP_ALIGN.CENTER)
        text(sl, bx, Inches(4.8), Inches(2.5), Inches(0.6),
             d, 13, C_WHITE, False, PP_ALIGN.CENTER)

    # Key message
    shape(sl, Inches(1.5), Inches(5.9), Inches(10.3), Inches(1.0),
          RGBColor(0x23, 0x4E, 0x74), C_GREEN, Pt(2))
    text(sl, Inches(1.5), Inches(5.95), Inches(10.3), Inches(0.45),
         "Sensor -> Expert (Frozen) -> K-MER Fusion (144K) -> Post-Processing -> 8-Byte Packet -> Vehicle",
         15, RGBColor(0xBB, 0xDD, 0xFF), False, PP_ALIGN.CENTER)
    text(sl, Inches(1.5), Inches(6.4), Inches(10.3), Inches(0.4),
         "경량 모델 (144K) + 높은 정확도 (98.0%) + 실시간 Edge 배포 + 차량 프로토콜 직접 연동",
         14, C_GREEN, True, PP_ALIGN.CENTER)

    # ═══ SAVE ═══
    out = f"{BASE}/K-MER_Presentation.pptx"
    prs.save(out)
    print(f"[OK] Saved: {out}")
    print(f"     Slides: {n}")


if __name__ == "__main__":
    main()
