# K-MER: 멀티모달 운전자 감정인식 시스템 파이프라인 이론서

> **Korean Multimodal Emotion Recognition — Pipeline & Theoretical Foundations**
> 산업부 전자부품산업기술개발 국책과제 | NVIDIA Jetson Orin 실증 시스템

---

## 목차

1. [시스템 전체 파이프라인](#1-시스템-전체-파이프라인)
2. [Stage 1: 센서 입력 및 전처리](#2-stage-1-센서-입력-및-전처리)
3. [Stage 2: K-FER — 얼굴 감정 인식 Expert](#3-stage-2-k-fer--얼굴-감정-인식-expert)
4. [Stage 3: Audio Expert — 음성 감정 인식](#4-stage-3-audio-expert--음성-감정-인식)
5. [Stage 4: Bio Expert — 생체 신호 기반 각성도](#5-stage-4-bio-expert--생체-신호-기반-각성도)
6. [Stage 5: FACS Auxiliary — 안면 기하 특징 + 졸음](#6-stage-5-facs-auxiliary--안면-기하-특징--졸음)
7. [Stage 6: K-MER Fusion — 멀티모달 토큰 퓨전](#7-stage-6-k-mer-fusion--멀티모달-토큰-퓨전)
8. [Stage 7: Post-Processing — 시간적 안정화 및 복합감정 해석](#8-stage-7-post-processing--시간적-안정화-및-복합감정-해석)
9. [Stage 8: Gateway Packet Encoding — 차량 게이트웨이 출력](#9-stage-8-gateway-packet-encoding--차량-게이트웨이-출력)
10. [Knowledge Distillation — 경량 학생 모델](#10-knowledge-distillation--경량-학생-모델)
11. [손실 함수 및 학습 전략](#11-손실-함수-및-학습-전략)
12. [설계 근거 및 이론적 배경](#12-설계-근거-및-이론적-배경)

---

## 1. 시스템 전체 파이프라인

K-MER 시스템은 **4개의 독립 Expert 모듈**이 각 모달리티에서 특징을 추출하고,
**토큰 기반 멀티모달 퓨전 모듈**이 이를 통합하여 운전자의 감정·각성도·졸음 상태를
판별하는 구조이다. 최종 출력은 8-byte USB 패킷으로 차량 게이트웨이에 전송된다.

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         K-MER END-TO-END PIPELINE                               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   [Stage 1] 센서 입력                                                            ║
║   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                         ║
║   │ RGB Camera   │   │ E4 Wristband │   │ Microphone   │                         ║
║   │ (RealSense)  │   │ (BVP/EDA/HR) │   │ (16kHz)      │                         ║
║   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                         ║
║          │                  │                   │                                 ║
║   [Stage 2]          [Stage 4]          [Stage 3]                                ║
║   ┌──────▼───────┐   ┌──────▼───────┐   ┌──────▼───────┐   ┌──────────────┐     ║
║   │ K-FER Expert │   │ Bio Expert   │   │ Audio Expert │   │ FACS Aux     │     ║
║   │ MobileViTv2  │   │ neurokit2    │   │ emotion2vec  │   │ [Stage 5]    │     ║
║   │ +AU Cross-   │   │ hand-craft   │   │ +audeering   │   │ EAR/PERCLOS  │     ║
║   │  Attention   │   │ 0 params     │   │ frozen enc.  │   │ +FACS 6d     │     ║
║   │ ~5M params   │   │              │   │              │   │              │     ║
║   ├──────────────┤   ├──────────────┤   ├──────────────┤   ├──────────────┤     ║
║   │ T1: probs 7d │   │ T7: BVP  4d │   │ T4: e2v  9d │   │ T11: PER 2d │     ║
║   │ T2: meta  2d │   │ T8: EDA  5d │   │ T5: AVD  3d │   │ T12: FAC 6d │     ║
║   │ T3: face  3d │   │ T9: HRT  6d │   │ T6: AQ   3d │   │ T13: XMD 3d │     ║
║   │              │   │ T10: BQ  3d │   │              │   │ T14: FLG 3d │     ║
║   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘     ║
║          │                  │                   │                  │              ║
║          └──────────────────┴───────────────────┴──────────────────┘              ║
║                                      │                                           ║
║   [Stage 6]                          ▼                                           ║
║   ┌──────────────────────────────────────────────────────────────────┐            ║
║   │                    K-MER Multimodal Fusion                       │            ║
║   │  15 Tokens × 64d → Pool-FFN → MHSA → CLS Pooling → Heads      │            ║
║   │  Arousal [0,1] | Valence [0,1] | Drowsy {0,1,2}                │            ║
║   └──────────────────────────┬───────────────────────────────────────┘            ║
║                              │                                                   ║
║   [Stage 7]                  ▼                                                   ║
║   ┌──────────────────────────────────────────────────────────────────┐            ║
║   │                       Post-Processing                            │            ║
║   │  Temporal Smoothing (7-frame MV) | Compound Emotion (7×A→13)    │            ║
║   │  StableEmotionDetector (4-strategy) | DrowsinessJudge           │            ║
║   └──────────────────────────┬───────────────────────────────────────┘            ║
║                              │                                                   ║
║   [Stage 8]                  ▼                                                   ║
║   ┌──────────────────────────────────────────────────────────────────┐            ║
║   │                  Gateway Packet Encoder                          │            ║
║   │  8-byte USB: [SOF][TYPE][SEQ][LEN][Emo+Flags][Int+NE][CRC][EOF] │            ║
║   │  감정 6종 + 상태 4종 = 10개 인식 항목 → 종합 98.0%              │            ║
║   └──────────────────────────┬───────────────────────────────────────┘            ║
║                              │                                                   ║
║                              ▼                                                   ║
║                    차량 게이트웨이 (USB)                                          ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Stage 1: 센서 입력 및 전처리

### 2.1 RGB Camera — Intel RealSense D435

| 항목 | 값 |
|------|-----|
| 해상도 | 1280 × 720 (color stream) |
| 프레임 | 30 fps |
| 인터페이스 | USB 3.0 |

카메라 스레드는 별도 daemon thread에서 실행되며, **latest-frame 패턴**을 사용한다.
메인 스레드는 항상 최신 1프레임만 가져와 처리하므로, 프레임 드롭 없이 추론 지연에
유연하게 대응한다.

**전처리 파이프라인:**

```
frame_bgr (1280×720)
    │
    ▼ short_side → 800px 리사이즈
    │
    ▼ MediaPipe FaceMesh (468 + 10 landmark)
    │
    ├─ 전체 landmark bbox → 20% padding → 얼굴 crop
    │   └─ crop → 224×224 리사이즈 (K-FER 입력)
    │
    └─ 8개 AU 영역 좌표 추출
        └─ pixel → [0, 224] 범위 변환 → au_coords [8, 2]
```

### 2.2 E4 Wristband (생체 센서)

| 채널 | 샘플링 | 생리학적 의미 |
|------|--------|-------------|
| BVP (Blood Volume Pulse) | 64 Hz | 광용적맥파 → 심박변이도(HRV) → 교감/부교감 신경 활성 |
| EDA (Electrodermal Activity) | 4 Hz | 피부전도도 → 피부전도반응(SCR) → 정서적 각성(arousal) |
| Temperature | 4 Hz | 피부 온도 → 자율신경 활성 변화 추세 |
| HR (Heart Rate) | 1 Hz | 심박수 → 전반적 각성 수준 |

BVP에서 HRV를 추출하면 **교감신경(LF) vs 부교감신경(HF)** 활성도 비율을 계산할 수
있으며, 이는 정서적 각성도의 핵심 생리 지표이다 (Kreibig, 2010).

### 2.3 Microphone (음성)

| 항목 | 값 |
|------|-----|
| 샘플링 | 16 kHz, mono |
| 세그먼트 | 5-second windows |

음성 신호는 두 개의 frozen encoder를 통해 각각 **이산적 감정 범주**(emotion2vec)와
**연속적 정서 차원**(audeering wav2vec2)을 추출한다.

---

## 3. Stage 2: K-FER — 얼굴 감정 인식 Expert

K-FER(Korean Facial Emotion Recognition)은 본 시스템의 **핵심 감정 분류기**로,
단일 얼굴 이미지에서 Action Unit(AU) 기반 Cross-Attention을 통해 7-class 감정을
분류한다. AI Hub 한국인 감정인식 데이터셋(413K 학습, 52K 검증)으로 학습되었다.

### 3.1 전체 구조

```
Input Face Image [B, 3, 224, 224]
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                   MobileViTv2-100 Backbone                     │
│                   (timm, ImageNet pretrained, ~5M params)      │
│                   1× forward only (핵심 최적화)                │
│                                                                │
│   forward_features() → feat_map [B, 384, 7, 7]               │
└───────────────────────────┬───────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
      Global Avg Pool              AU RoI Extractor
      feat_map.mean([-2,-1])       8 AU regions
      → global_feat [B, 384]      grid_sample(bilinear)
                                   + AU Positional Embed
              │                    → au_tokens [B, 8, 384]
              │                           │
              ▼                           ▼
       ┌──────────────────────────────────────────┐
       │     Token Sequence (10 tokens × 384d)     │
       │                                            │
       │   [CLS]  [Global]  [AU₁] [AU₂] ... [AU₈] │
       │   384d     384d     ────── 8×384d ──────  │
       │                                            │
       │   CLS: nn.Parameter, trunc_normal(σ=0.02) │
       └──────────────────┬───────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────────┐
       │     CrossAttentionFusion (1 layer)        │
       │                                            │
       │  Step 1: Cross-Attention (Pre-Norm)       │
       │    Q = [CLS, Global]                      │
       │    K, V = [AU₁, AU₂, ..., AU₈]           │
       │    Attn = softmax(QKᵀ/√d_k) · V          │
       │    → Gated Residual: x + σ(g) ⊙ Attn(x) │
       │    (g: per-dim learnable, init=0)          │
       │                                            │
       │  Step 2: Self-Attention (Pre-Norm)        │
       │    Q = K = V = all 10 tokens              │
       │    전체 토큰 간 정보 교환                  │
       │                                            │
       │  Step 3: Feed-Forward Network             │
       │    384 → 1536 (GELU) → 384                │
       │    + Residual + Dropout(0.1)              │
       └──────────────────┬───────────────────────┘
                          │
                     CLS token [B, 384]
                          │
                          ▼
       ┌──────────────────────────────────────────┐
       │              FER Head                      │
       │  LayerNorm(384)                            │
       │  → Linear(384 → 384) → GELU               │
       │  → Dropout(0.2)                            │
       │  → Linear(384 → 7)                         │
       └──────────────────┬───────────────────────┘
                          │
                          ▼
                  logits [B, 7]
                  softmax → probs [B, 7]
                  ┌─────────────────────────────────────────────┐
                  │ angry │ anxious │ happy │ hurt │ neutral │ sad │ surprised │
                  │   0   │    1    │   2   │  3   │    4    │  5  │     6     │
                  └─────────────────────────────────────────────┘
```

### 3.2 AU RoI Extraction — 이론적 근거

전통적 얼굴 감정 인식 연구에서는 **Facial Action Coding System (FACS)**의
Action Unit(AU)이 표정의 근육 단위 분해를 제공한다 (Ekman & Friesen, 1978).
그러나 개별 AU Detector를 학습하는 것은 annotation cost가 높고, 각 AU별 별도
backbone forward가 필요하여 추론 비용이 급증한다.

K-FER은 이를 해결하기 위해 **단일 backbone의 feature map에서 AU 영역을 직접
추출**하는 방식을 채택하였다:

1. MediaPipe FaceMesh의 468개 landmark에서 8개 AU 관심 영역의 중심 좌표를 계산
2. `F.grid_sample(bilinear)`로 feature map에서 해당 위치의 특징 벡터를 추출
3. 학습 가능한 AU Positional Embedding을 더하여 각 AU region의 고유 정보를 인코딩

**8개 AU Region 정의:**

| # | 영역 | MediaPipe Landmark | FACS 대응 |
|---|------|-------------------|-----------|
| 0 | Forehead (이마) | avg(69, 299, 9) | AU1+AU2 (내/외측 눈썹올림) |
| 1 | Eyes Left (왼눈) | 159 | AU5 (상안검 거상), AU7 (안검 긴장) |
| 2 | Eyes Right (오른눈) | 386 | AU5, AU7 |
| 3 | Nose (코) | 195 | AU9 (코 주름), AU10 (상순 거상) |
| 4 | Cheek Left (왼볼) | 186 | AU6 (볼 올림) |
| 5 | Cheek Right (오른볼) | 410 | AU6 |
| 6 | Mouth (입) | 13 | AU12 (입꼬리 당김), AU25 (입술 벌림) |
| 7 | Chin (턱) | 18 | AU17 (턱 올림) |

### 3.3 Cross-Attention의 설계 의도

기존 ViT 기반 감정 인식 모델은 전역 Self-Attention만 사용하여, 감정에 중요한
국소 영역(눈, 입 등)의 fine-grained 특징이 전역 특징에 묻히는 문제가 있다.

K-FER의 Cross-Attention은 이를 두 단계로 해결한다:

**Step 1 — Cross-Attention: "어떤 AU가 현재 감정에 중요한가?"**
- CLS/Global 토큰이 Query가 되어, 8개 AU 토큰에서 **선택적으로** 정보를 추출
- 분노 표정에서는 눈썹(AU0)·입(AU6)에 높은 attention weight
- 행복 표정에서는 볼(AU4,5)·입(AU6)에 높은 attention weight
- **Gated Residual**: `σ(g) ⊙ cross_out`으로 정보 흐름을 per-dimension 제어
  - `g`는 0으로 초기화되어, 학습 초기에는 skip connection이 지배적
  - 학습이 진행되면서 점차 cross-attention 정보를 수용

**Step 2 — Self-Attention: "전체 맥락에서 감정을 종합적으로 판단"**
- 10개 전체 토큰이 서로 attend하여 global context를 형성
- AU 간 상호작용 (예: 눈썹 올림 + 입 벌림 = 놀람)을 학습

이 **Cross-first, Self-second** 순서는 일반적인 transformer 구조와 반대이며,
AU의 국소 정보를 먼저 CLS에 집약한 후 전역 판단을 내리는 것이 더 효과적임을
실험적으로 확인하였다.

### 3.4 학습 및 성능

| 항목 | 값 |
|------|-----|
| 데이터셋 | AI Hub 한국인 감정인식 |
| 학습 | 413,122 images |
| 검증 | 51,804 images |
| Backbone LR | base_lr × 0.1 (차별적 학습률) |
| Head LR | base_lr |
| Accuracy | 79.7% |
| Macro F1 | 0.7953 |
| Checkpoint | `emotion_system/result/best.pth` (84MB) |

---

## 4. Stage 3: Audio Expert — 음성 감정 인식

음성 모달리티는 두 개의 **frozen pre-trained encoder**를 사용하여
상호 보완적인 정서 정보를 추출한다.

### 4.1 emotion2vec — 이산적 감정 범주

| 항목 | 값 |
|------|-----|
| 모델 | emotion2vec_plus_large (Ma et al., ACL 2024) |
| 사전학습 | 160K hours 음성 감정 데이터 |
| 출력 | 1024-dim embedding → 9-class softmax |
| 인터페이스 | FunASR |

emotion2vec는 **자기지도 사전학습(Self-Supervised Pre-Training)** 기반의 음성 감정
표현 모델로, 대규모 음성 데이터에서 학습된 표현을 9가지 감정 범주로 분류한다.

**9-class 감정:**
```
angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown
```

K-MER에서는 이 9-class probability distribution을 그대로 토큰 T4로 사용한다.
Fine-tuning 없이 frozen encoder로 사용하는 이유는, K-EMocon 데이터셋의
음성 세그먼트 수(~2,259)가 encoder 학습에 불충분하기 때문이다.

### 4.2 audeering wav2vec2 — 연속적 정서 차원

| 항목 | 값 |
|------|-----|
| 모델 | wav2vec2-large-robust-12-ft-emotion-msp-dim |
| Backbone | Wav2Vec2 (1024d hidden states) |
| Head | Linear(1024→1024) → Tanh → Linear(1024→3) |
| 출력 | Arousal, Valence, Dominance ∈ [0, 1] |

Russell의 **Circumplex Model of Affect** (1980)에 따르면, 감정은
Valence(쾌-불쾌)와 Arousal(각성-이완)의 2차원 연속 공간에서 표현된다.
audeering 모델은 이 연속적 정서 차원을 음성에서 직접 예측한다.

이산적 감정 범주(emotion2vec)와 연속적 정서 차원(audeering)을 동시에 사용함으로써,
**범주적 판별력**과 **차원적 세밀함**을 모두 확보한다.

### 4.3 Audio Quality 추정

```
T6: audio_quality [3d]
├─ RMS energy → 음성 존재 여부
├─ Voicing ratio → 유성음 비율 (음성 vs 침묵)
└─ SNR estimation → 신호 대 잡음비
```

이 품질 지표는 K-MER Fusion의 validity masking에 활용되어,
음성이 없거나 잡음이 심한 구간에서 audio 토큰의 attention 기여를 자동으로 줄인다.

---

## 5. Stage 4: Bio Expert — 생체 신호 기반 각성도

Bio Expert는 **학습 파라미터 없이** neurokit2 라이브러리 기반 hand-crafted feature
engineering으로 생체 신호에서 15차원 특징 벡터를 추출한다.

### 5.1 BVP → HRV Features (T7: 4d)

심박변이도(Heart Rate Variability)는 자율신경계 활성의 핵심 생리 지표이다.

```
BVP signal (64Hz)
    │
    ▼ neurokit2.ppg_clean() → bandpass filtering
    │
    ▼ neurokit2.ppg_findpeaks() → R-peak detection
    │
    ▼ Inter-Beat Interval (IBI) 계산
    │
    ├─ mean_hr = 60000 / mean(IBI)        ← 평균 심박수
    ├─ sdnn = std(IBI)                     ← 전체 HRV (시간 영역)
    ├─ rmssd = √(mean(diff(IBI)²))        ← 연속 차이의 RMS (부교감 지표)
    └─ lf_hf = PSD_LF / PSD_HF            ← 교감/부교감 균형 (주파수 영역)
              LF: 0.04~0.15Hz (교감+부교감)
              HF: 0.15~0.40Hz (부교감)
```

**이론적 배경:**
- **SDNN**: 전체 자율신경 활성도의 지표. 스트레스 시 감소 (Malik et al., 1996)
- **RMSSD**: 미주신경(부교감) 톤의 지표. 이완 시 증가 (Task Force, 1996)
- **LF/HF ratio**: 교감-부교감 균형. 스트레스/불안 시 LF 증가 (Kreibig, 2010)

### 5.2 EDA → SCR Features (T8: 5d)

피부전도도(Electrodermal Activity)는 **교감신경 전용 지표**로,
정서적 각성(arousal)에 가장 직접적으로 반응하는 생리 신호이다.

```
EDA signal (4Hz)
    │
    ├─ mean_scl = mean(EDA)               ← 기저 피부전도 수준
    ├─ std_scl = std(EDA)                  ← 변동성
    ├─ n_peaks = count(peaks > μ+0.5σ)     ← SCR(피부전도반응) 발생 빈도
    ├─ mean_amp = mean(EDA[peaks] - μ)     ← SCR 평균 진폭
    └─ auc = ∫|EDA - μ|dt                  ← 곡선하면적 (총 각성량)
```

**이론적 배경:**
- SCR 빈도와 진폭은 정서적 각성도(arousal)와 강한 양의 상관 (Boucsein, 2012)
- 운전 중 스트레스 감지에서 EDA는 가장 신뢰성 높은 생리 지표 (Healey & Picard, 2005)

### 5.3 HR/Temperature Features (T9: 6d)

```
├─ hr_mean, hr_std, hr_range     ← 심박수 통계 (HR 센서, 1Hz)
└─ temp_mean, temp_slope, temp_range  ← 피부 온도 추세
```

피부 온도의 **slope**(기울기)은 이완/긴장 상태 전이를 포착한다.
긴장(fight-or-flight) 시 말초 혈관 수축으로 피부 온도가 감소하며,
이완 시 말초 혈관 확장으로 온도가 상승한다 (Kreibig, 2010).

### 5.4 Bio Quality (T10: 3d)

```
bio_quality = [bvp_valid, eda_valid, hr_valid]

각 채널별:
  quality = 0.5 × (1 - NaN비율) + 0.5 × (유효채널 수 / 3)
  valid = (유효채널 ≥ 1)
```

센서 탈착이나 움직임 아티팩트로 인한 데이터 손실을 정량화하여,
K-MER Fusion에서 bio 토큰의 신뢰도를 동적으로 조절한다.

---

## 6. Stage 5: FACS Auxiliary — 안면 기하 특징 + 졸음

### 6.1 EAR (Eye Aspect Ratio) — 눈 감김 검출

```
         p2         p3
    ─────●──────────●─────
   /                       \
  ● p1                   p4 ●
   \                       /
    ─────●──────────●─────
         p6         p5

EAR = (‖p2 - p6‖ + ‖p3 - p5‖) / (2 × ‖p1 - p4‖)
```

- **눈 뜸**: EAR ≈ 0.25~0.35
- **눈 감김**: EAR < 0.21 (threshold)
- 두 눈의 EAR 평균을 사용

**MediaPipe Landmark:**
- LEFT_EYE: [362, 385, 387, 263, 373, 380]
- RIGHT_EYE: [33, 160, 158, 133, 153, 144]

### 6.2 PERCLOS — 졸음 판정

PERCLOS(Percentage of Eye Closure)는 NHTSA/FHWA 표준 졸음 지표이다
(Dinges & Grace, 1998).

```
              1초 윈도우 (30 frames @ 30fps)

  EAR:  ──●──●──●──●──●──●──●──●──●──●──
         0.3 0.28 0.12 0.10 0.09 0.11 0.25 0.30 0.31 0.29
                   ▲ ─ ─ ─ ─ ─ ▲
                   EAR < 0.21 (눈 감김 구간)

  PERCLOS = (눈 감김 프레임 수) / (전체 프레임 수)
         = 4 / 10 = 0.40

  판정:
  ┌──────────────────┬──────────────┬──────────────────┐
  │ PERCLOS < 0.2    │ 0.2 ≤ P < 0.4│ PERCLOS ≥ 0.4    │
  │ Alert (정상)     │ Low Attention │ Drowsy (졸음)    │
  │                  │ (주의 분산)   │                  │
  └──────────────────┴──────────────┴──────────────────┘
```

**Arousal 연동 보정:**
```python
if arousal < 0.3:  # K-MER Fusion의 arousal이 매우 낮으면
    threshold_drowsy *= 0.7   # 졸음 기준을 더 민감하게
    threshold_sleeping *= 0.8
```

이는 "저각성 + 눈 감김 증가"가 졸음의 강력한 복합 신호이므로,
멀티모달 정보를 활용하여 판정 정밀도를 높이는 설계이다.

### 6.3 FACS Geometric Scores (T12: 6d)

MediaPipe FaceMesh의 landmark 좌표에서 직접 계산하는 기하학적 표정 지표:

| # | 특징 | 계산 | 감정 연관 |
|---|------|------|----------|
| 0 | Mouth Openness | \|lip_top - lip_bottom\| / face_h | 놀람, 공포 |
| 1 | Left Brow Raise | \|brow_L - eye_L\| / face_h | 놀람, 공포 |
| 2 | Right Brow Raise | \|brow_R - eye_R\| / face_h | 놀람, 공포 |
| 3 | Left Lip Corner | (lip_center - corner_L).y / face_h | 행복(+), 슬픔(-) |
| 4 | Right Lip Corner | (lip_center - corner_R).y / face_h | 행복(+), 슬픔(-) |
| 5 | Nose Wrinkle | \|nasion - nose_tip\| / face_h | 혐오, 분노 |

이 특징들은 K-FER의 CNN 기반 특징과 상호 보완적이며, 기하학적 변형에 강인하다.

### 6.4 Cross-Modal Agreement (T13: 3d)

서로 다른 모달리티의 예측이 얼마나 일치하는지 측정하는 메타 특징:

**1. Face-Audio Agreement:**
```
kfer_valence = P(happy) + 0.5×P(surprised) - P(angry) - P(anxious) - P(hurt) - P(sad)
e2v_valence  = P(happy) + 0.5×P(surprised) - P(angry) - P(fearful) - P(disgusted) - P(sad)
agreement = max(0, 1.0 - |kfer_valence - e2v_valence| / 2.0)
```

**2. Arousal-Emotion Consistency:**
```
arousal = audeering_avd[0]          # 연속적 각성도
e2v_intensity = 1 - P(neutral)      # 감정 강도
consistency = 1 - |arousal - e2v_intensity|
```

**3. Entropy Gap:**
```
kfer_H = -Σ P_kfer(i) log P_kfer(i)   # K-FER 엔트로피
hse_H  = -Σ P_hse(j) log P_hse(j)     # HSEmotion 엔트로피
entropy_gap = max(0, 1 - |kfer_H - hse_H| / 2.0)
```

두 얼굴 감정 모델의 엔트로피가 유사하면 둘 다 확신적이거나 둘 다 불확실한 것이므로,
예측 신뢰도를 교차 검증하는 효과가 있다.

---

## 7. Stage 6: K-MER Fusion — 멀티모달 토큰 퓨전

K-MER Fusion은 14개 feature 토큰 + 1개 [CLS] 토큰 = **15개 토큰을 64차원으로
통합**하는 경량 Transformer 기반 퓨전 모듈이다. EfficientFormer 스타일의
Pool-FFN으로 모달리티 내 지역 혼합을 수행한 후, MHSA로 모달리티 간 전역
교차 주의를 수행한다.

### 7.1 Token Formation & Projection

```
각 토큰 t_i (원본 차원 d_i):
    t̂_i = LayerNorm(W_i · t_i + b_i)     ← Linear(d_i → 64) + LN
    t̂_i = t̂_i + E_mod(type_i)            ← Modality Type Embedding

Modality Type Embedding: nn.Embedding(5, 64)
    Type 0: Face  (T1, T2, T3)
    Type 1: Audio (T4, T5, T6)
    Type 2: Bio   (T7, T8, T9, T10)
    Type 3: Aux   (T11, T12, T13)
    Type 4: Meta  (T14, CLS)

CLS Token: nn.Parameter([1, 1, 64]), trunc_normal(σ=0.02)
```

### 7.2 Intra-Modal Pool-FFN

각 모달리티 그룹 내에서 **지역적 토큰 혼합(local token mixing)**을 수행한다.
이는 EfficientFormer (Li et al., CVPR 2022)에서 영감을 받은 설계로,
Self-Attention보다 연산량이 매우 적으면서 유사한 효과를 달성한다.

```
Pool-FFN(x) = x + FFN(LayerNorm(AvgPool1d(x)))

┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  Face Group (T1, T2, T3):    AvgPool1d(k=3, s=1, p=1)       │
│  Audio Group (T4, T5, T6):   AvgPool1d(k=3, s=1, p=1)       │
│  Bio Group (T7~T10):         AvgPool1d(k=4, s=1, p=2)       │
│  Aux/Meta (T11~T15):         passthrough (혼합 없이 직통)    │
│                                                               │
│  FFN = Linear(64 → 128) → GELU → Linear(128 → 64)           │
│  (expansion ratio = 2)                                        │
│                                                               │
│  연산 비용: 그룹당 < 0.1ms (Jetson Orin 기준)                │
└─────────────────────────────────────────────────────────────┘
```

**이론적 근거:**
AvgPool1d는 인접 토큰 간의 이동 평균으로, 모달리티 내부의 **관련 특징들을
부드럽게 혼합**한다. 예를 들어, Face 그룹에서는 K-FER 확률(T1)과 메타 정보(T2),
얼굴 통계(T3)가 서로의 맥락을 공유한다. 이 과정이 Self-Attention 대비 O(1)
복잡도로 수행되므로, 경량화에 핵심적인 역할을 한다.

### 7.3 Global Cross-Modal MHSA

모달리티 내 혼합이 끝난 후, **전체 15개 토큰에 대해 1-layer Multi-Head
Self-Attention**을 수행하여 모달리티 간 교차 정보를 학습한다.

```
MHSA(X) = MultiHead(Q, K, V) where Q = K = V = LayerNorm(X)

MultiHead(Q, K, V) = Concat(head₁, ..., head₄) · W^O

headᵢ = Attention(Q·W^Q_i, K·W^K_i, V·W^V_i)

Attention(Q, K, V) = softmax(QKᵀ / √d_k + M) · V

M: key_padding_mask
   M[b, j] = -∞  if token j is invalid (해당 모달리티 센서 부재)
   M[b, j] = 0    if token j is valid

┌─────────────────────────────────────────────────────────┐
│  Configuration:                                           │
│    embed_dim = 64                                        │
│    num_heads = 4 (d_k = 16 per head)                     │
│    dropout = 0.1                                          │
│    batch_first = True                                    │
│                                                           │
│  Pre-Norm → MHSA → Residual                             │
│  Pre-Norm → FFN(64→128→64, GELU) → Residual             │
└─────────────────────────────────────────────────────────┘
```

**Validity Masking — 모달리티 결손 대응:**

실제 운용 환경에서는 센서 탈착, 소음, 움직임 아티팩트 등으로 특정 모달리티가
일시적으로 사용 불가할 수 있다. K-MER은 이를 **key_padding_mask**로 처리한다:

```
Token Range     Valid Condition
─────────────────────────────
T1~T3  (Face)   face_valid
T4~T6  (Audio)  audio_valid
T7~T10 (Bio)    bio_valid
T11~T12 (Aux)   face_valid
T13    (XMD)    face_valid
T14, CLS        항상 valid
```

Invalid 토큰의 Key에 -∞ 마스크를 적용하면, softmax에서 해당 토큰의 attention
weight가 0이 되어 **자연스럽게 무시**된다. 이로써 모델은 사용 가능한 모달리티만으로
예측을 수행할 수 있다 (graceful degradation).

### 7.4 CLS Pooling & Output Heads

```
fused_repr = tokens[:, -1, :]    # CLS token (last position) → [B, 64]

┌─────────────────────────────────────────────────────┐
│  Arousal Head:                                        │
│    Linear(64→32) → ReLU → Dropout(0.1) → Linear(32→1) → Sigmoid     │
│    출력: arousal ∈ [0, 1]                            │
│                                                       │
│  Valence Head:                                        │
│    Linear(64→32) → ReLU → Dropout(0.1) → Linear(32→1) → Sigmoid     │
│    출력: valence ∈ [0, 1]                            │
│                                                       │
│  Drowsiness Head:                                     │
│    Input: [fused_repr.detach() ‖ perclos ‖ ear_mean]  │
│    Linear(66→32) → ReLU → Dropout(0.1) → Linear(32→3)│
│    출력: {alert, drowsy, sleeping}                   │
│    ※ fused_repr를 detach하여 각성도 gradient 차단     │
└─────────────────────────────────────────────────────┘
```

**Drowsiness Head의 detach 설계:**
졸음 판정은 PERCLOS/EAR이라는 직접적 지표가 있으므로, fused_repr의 gradient가
졸음 분류에 의해 왜곡되지 않도록 detach한다. 이로써 arousal/valence head는
순수하게 정서 차원 학습에 집중할 수 있다.

### 7.5 학습 설정

| 항목 | 값 |
|------|-----|
| 데이터셋 | K-EMocon (3,577 segments, 11 participants) |
| 교차검증 | GroupKFold 6-fold (participant ID로 그룹화) |
| Optimizer | AdamW (weight_decay) |
| 학습률 | 1e-3 |
| Epochs | 100~150 (early stopping) |
| 파라미터 | ~97K (teacher) |
| 최고 성능 | Arousal UAR 60.02% |

---

## 8. Stage 7: Post-Processing — 시간적 안정화 및 복합감정 해석

### 8.1 Temporal Smoothing — 7-Frame Majority Vote

실시간 추론에서 프레임 단위 예측은 불가피하게 노이즈를 포함한다.
7-frame sliding window의 majority vote로 이를 완화한다.

```
Time:   t-6  t-5  t-4  t-3  t-2  t-1   t
Pred:   행복 행복 행복 분노 행복 행복 행복
                                        ↑
Vote Result = 행복 (6/7 = 85.7%)
```

단일프레임 79.7% → 운용시(W=7) 평균 **98.8%** (6종 병합 기준)

이 극적인 개선은 K-FER의 AU 기반 예측이 프레임 간 일관성이 높기 때문이다.
오류는 주로 비연속적(sporadic)이므로, majority vote로 효과적으로 제거된다.

### 8.2 StableEmotionDetector — 4중 안정화 전략

Temporal Smoothing을 넘어, 4가지 안정화 전략을 결합하여 추론 안정성을 극대화한다:

**Strategy 1: Emotion-Group-Aware Majority Vote**
```
SUSTAINED emotions (neutral, happy, sad):  min_votes = 6  (높은 안정성 요구)
TRANSIENT emotions (surprised, angry):     min_votes = 3  (빠른 반응 허용)
```
지속적 감정(행복, 중립)은 빈번하게 바뀌면 안 되지만,
일시적 감정(놀람, 분노)은 빠르게 반응해야 하므로, 차별적 임계값을 적용한다.

**Strategy 2: EMA (Exponential Moving Average)**
```
P_ema(t) = α · P_new(t) + (1-α) · P_ema(t-1)      (α = 0.3)
```
확률 분포 자체에 지수 이동 평균을 적용하여 부드러운 전이를 유도한다.

**Strategy 3: Confidence Threshold (0.5)**
```
if max(P_ema) < confidence_threshold:
    keep previous prediction (불확실하면 이전 결과 유지)
```

**Strategy 4: Minimum Hold Time (0.5초)**
```
감정 변경 후 최소 0.5초간 유지 → 초당 2회 이상 변경 방지
```

### 8.3 Compound Emotion Refiner — 7 × Arousal → 13 Label

Russell의 Circumplex Model을 기반으로, K-FER의 7-class 이산 감정에
Arousal 연속 값을 결합하여 13가지 세분화된 복합 감정으로 확장한다.

```
Arousal 이산화:
  Low:  arousal < 0.33
  Mid:  0.33 ≤ arousal < 0.66
  High: arousal ≥ 0.66

매핑 예시:
┌──────────┬────────────┬────────────────┬───────────────┐
│ K-FER    │ Low Arousal │ Mid Arousal    │ High Arousal  │
├──────────┼────────────┼────────────────┼───────────────┤
│ neutral  │ calm       │ neutral        │ neutral       │
│ happy    │ happy      │ positive_engaged│ excited      │
│ sad      │ depressed  │ sad            │ sad           │
│ anxious  │ anxious    │ anxious        │ stressed      │
│ angry    │ angry      │ angry          │ angry         │
│ hurt     │ hurt       │ hurt           │ hurt          │
│ surprised│ surprised  │ surprised      │ surprised     │
└──────────┴────────────┴────────────────┴───────────────┘

최종 13-class:
  0:neutral  1:calm  2:happy  3:positive_engaged  4:excited
  5:sad  6:depressed  7:anxious  8:stressed  9:angry
  10:hurt  11:surprised  12:drowsy
```

이 확장은 단순 "행복"과 "고각성 행복(excited)"을 구분하여,
운전자의 정서 상태를 더 세밀하게 파악할 수 있게 한다.

---

## 9. Stage 8: Gateway Packet Encoding — 차량 게이트웨이 출력

최종 분석 결과를 **8-byte USB 패킷**으로 인코딩하여 차량 게이트웨이에 전송한다.

### 9.1 패킷 비트 레이아웃

```
Byte 0    Byte 1    Byte 2    Byte 3    Byte 4         Byte 5           Byte 6    Byte 7
┌────────┐┌────────┐┌────────┐┌────────┐┌──────────────┐┌────────────────┐┌────────┐┌────────┐
│  0xAA  ││  0x01  ││  SEQ   ││  0x02  ││[EmoCode][Flg]││[EmoI][StI][N][R]││  CRC8  ││  0xFE  │
│  SOF   ││  TYPE  ││ 0~255  ││  LEN   ││  4bit   4bit ││ 3b   3b  1  1  ││ B1~B5  ││  EOF   │
└────────┘└────────┘└────────┘└────────┘└──────────────┘└────────────────┘└────────┘└────────┘

Byte 4 상세:
  bit[7:4]  Emotion Code (0~5, 6종)
  bit[3]    Stress Flag
  bit[2]    Low Attention Flag
  bit[1]    Drowsy Flag
  bit[0]    END_FLAG (이벤트 종료)

Byte 5 상세:
  bit[7:5]  Emotion Intensity (0~7, confidence → 3bit 양자화)
  bit[4:2]  State Intensity (0~7, arousal → 3bit 양자화)
  bit[1]    Negative Emotion Flag (부정감정)
  bit[0]    Reserved
```

### 9.2 Emotion Code 매핑 (6종, sad+hurt 병합)

```
K-FER 7-class          Protocol 6종
─────────────          ──────────────
angry(0)     ────→     Code 2 (분노)
anxious(1)   ────→     Code 0 (공포)
happy(2)     ────→     Code 4 (행복)
hurt(3)      ────┐
                 ├──→  Code 3 (슬픔/혐오)    ← 병합
sad(5)       ────┘
neutral(4)   ────→     Code 5 (중립)
surprised(6) ────→     Code 1 (놀람)
```

**sad+hurt 병합 근거:**
AI Hub 한국인 감정 데이터에서 sad↔hurt 간 혼동률이 32.7%로 매우 높다.
이는 한국 문화권에서 "슬픔"과 "서운함/상처"의 표정이 유사하기 때문이며,
두 클래스를 병합하면 단일프레임 68.4% → 운용시 95.7%로 크게 개선된다.

### 9.3 Status Flag 판정 (4종)

| Flag | 위치 | 판정 논리 | 이론적 근거 |
|------|------|---------|-----------|
| **Stress** | Byte4 bit3 | arousal > 0.6 AND K-FER ∈ {angry, anxious} | 고각성 + 부정감정 = 스트레스 (Lazarus, 1991) |
| **Low Attention** | Byte4 bit2 | PERCLOS ∈ [0.2, 0.4) | 졸음 전 단계, 주의력 저하 시작 |
| **Drowsy** | Byte4 bit1 | PERCLOS ≥ 0.4 | NHTSA 표준 (Dinges & Grace, 1998) |
| **Negative Emotion** | Byte5 bit1 | K-FER ∈ {angry, anxious, hurt, sad} | 안전 운전에 부정적 영향을 미치는 감정군 |

### 9.4 Intensity 양자화

```
quantize(v) = min(⌊clamp(v, 0, 1) × 8⌋, 7)

Emotion Intensity: K-FER softmax max probability → 0~7
  0.0~0.125 → 0 (매우 낮음)
  0.875~1.0 → 7 (매우 높음)

State Intensity: Arousal prediction → 0~7
  0.0~0.125 → 0 (이완)
  0.875~1.0 → 7 (고각성)
```

### 9.5 CRC8 무결성 검증

```
Polynomial: 0x07, Init: 0x00
범위: Byte1 ~ Byte5

for byte in payload:
    crc ^= byte
    for _ in range(8):
        crc = ((crc << 1) ^ 0x07) if (crc & 0x80) else (crc << 1)
        crc &= 0xFF
```

### 9.6 10개 인식 항목 종합 성능

| No. | 항목 | 단일프레임 | 운용시(W=7) |
|-----|------|-----------|------------|
| 1 | Code 0: 공포 | 71.0% | 97.9% |
| 2 | Code 1: 놀람 | 84.5% | 99.4% |
| 3 | Code 2: 분노 | 87.8% | 100.0% |
| 4 | Code 3: 슬픔/혐오 | 68.4% | 95.7% |
| 5 | Code 4: 행복 | 97.2% | 100.0% |
| 6 | Code 5: 중립 | 92.9% | 100.0% |
| 7 | 스트레스 | — | 95.0% |
| 8 | 주의분산 | — | 95.0% |
| 9 | 졸음 | — | 96.0% |
| 10 | 부정감정 | — | 99.8% |
| | **종합 Macro Average** | | **98.0%** |

---

## 10. Knowledge Distillation — 경량 학생 모델

Jetson Orin 엣지 배포를 위해, K-MER Fusion(교사, ~97K params)의 지식을
KMERStudent(학생, ~12K params)로 증류한다.

### 10.1 Student Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    KMERStudent (~12K params)                      │
│                                                                   │
│  Face path:  [kfer_probs(7) + meta(2) + face_stats(3)] = 12d   │
│              → Linear(12→64) → LN → GELU → Dropout → Linear(64→64) │
│                                                                   │
│  Bio path:   [bvp(4) + eda(5) + hr_temp(6) + bio_q(3)] = 18d  │
│              → Linear(18→64) → LN → GELU → Dropout → Linear(64→64) │
│                                                                   │
│  Audio path: [e2v(9) + avd(3) + aq(3)] = 15d                   │
│              → Linear(15→64) → LN → GELU → Dropout → Linear(64→64) │
│                                                                   │
│  Gating:     concat(192d)                                        │
│              → Linear(192→96) → ReLU → Linear(96→192) → Sigmoid │
│              gated = concat ⊙ gate_weights                      │
│                                                                   │
│  Fusion:     gated(192d) → Linear(192→64) → LN → GELU          │
│              → fused_repr (64d)                                  │
│                                                                   │
│  Arousal:    Linear(64→32) → ReLU → Dropout → Linear(32→1) → σ │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 KD Loss — 3단 지식 증류

```
L_total = λ_task · L_task + λ_feat · L_feature + λ_logit · L_logit

L_task    = BinaryFocalLoss(student_arousal, label)
L_feature = MSE(normalize(s_repr), normalize(t_repr))
L_logit   = KL(log_softmax(s_logit/τ), softmax(t_logit/τ)) · τ²

최적 가중치 (Student_heavy_KD):
  λ_task = 0.3,  λ_feat = 0.4,  λ_logit = 0.3,  τ = 4.0
```

**Feature-level KD (MSE):**
학생의 64d fused_repr이 교사의 64d fused_repr과 동일한 표현 공간을
학습하도록 유도한다. L2 정규화 후 MSE를 계산하여 크기에 무관하게
방향성(direction)을 정렬한다.

**Logit-level KD (KL):**
교사의 soft prediction에 담긴 "dark knowledge"를 전달한다.
온도 τ=4.0으로 softmax를 부드럽게 만들어, 교사가 "분노와 공포가 비슷하게
어려운 케이스"라고 판단한 정보까지 학생에게 전달한다 (Hinton et al., 2015).

### 10.3 KD Ablation 결과

| 실험 | Arousal UAR | 교사 대비 |
|------|-------------|---------|
| Student_no_KD (task loss only) | 57.55% | 95.9% |
| Student_standard_KD | 58.72% | 97.8% |
| **Student_heavy_KD** | **59.18%** | **98.6%** |
| Student_feature_only | 58.45% | 97.4% |
| Student_logit_only | 57.82% | 96.3% |
| Student_no_audio | 56.91% | 94.8% |

Feature KD와 Logit KD의 조합이 가장 효과적이며,
학생 모델이 교사 성능의 98.6%를 달성하면서 파라미터는 12.4%에 불과하다.

---

## 11. 손실 함수 및 학습 전략

### 11.1 Uncertainty-Weighted Multi-Task Learning

K-MER Fusion은 3가지 태스크(Arousal, Valence, Drowsy)를 동시에 학습한다.
Kendall et al. (CVPR 2018)의 동종 불확실성 가중 기법을 적용한다:

```
L_total = Σᵢ (1 / 2σᵢ²) · Lᵢ + log(σᵢ)

σᵢ: 태스크 i의 학습 가능한 불확실성 파라미터
  - σᵢ가 작으면 → 해당 태스크 가중치 증가 (확실한 태스크에 집중)
  - σᵢ가 크면 → 해당 태스크 가중치 감소 (불확실한 태스크 완화)
  - log(σᵢ) 항이 σᵢ → ∞를 방지 (정규화 역할)
```

이 기법은 태스크 간 가중치를 수동으로 튜닝할 필요 없이,
**학습 과정에서 자동으로 최적 균형**을 찾는다.

### 11.2 Binary Focal Loss

클래스 불균형 문제를 해결하기 위한 Focal Loss (Lin et al., ICCV 2017):

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

p_t = p     if y = 1
    = 1-p   if y = 0

γ = 2.0: 쉬운 샘플(p_t ≈ 1)의 기여를 급격히 감소
α = 0.65 (arousal): 양성(high arousal) 클래스에 더 높은 가중치
```

운전 중 고각성(스트레스, 분노) 상태는 저각성(이완, 졸음) 대비 빈도가 낮으므로,
Focal Loss가 희소한 고각성 상태의 학습에 효과적이다.

---

## 12. 설계 근거 및 이론적 배경

### 12.1 왜 Single Backbone Forward인가?

기존 AU 기반 FER 모델은 8개 AU 영역에 대해 각각 backbone forward를 수행하여
8× 이상의 추론 비용이 발생한다. K-FER은 **feature map level에서 RoI를 추출**하는
방식으로 backbone을 1회만 실행하면서도 동등한 AU 정보를 획득한다.

```
기존 방식:  8 × backbone(crop_i) → 8 × [B, C]    ← O(8N)
K-FER:     1 × backbone(face) → grid_sample(8)    ← O(N + 8)
```

Jetson Orin에서 MobileViTv2-100 1회 forward는 ~8ms이므로,
기존 방식은 ~64ms → K-FER은 ~9ms로 **7× 속도 향상**을 달성한다.

### 12.2 왜 Pool-FFN + MHSA 2단 구조인가?

| 구조 | 장점 | 단점 |
|------|------|------|
| MHSA only | 전역 교차 학습 | 15×15 attention → 연산량 |
| Pool-FFN only | 매우 빠름 | 모달리티 간 교차 불가 |
| **Pool-FFN + MHSA** | **지역 + 전역 최적 조합** | 약간의 추가 연산 |

Pool-FFN이 모달리티 내 중복 정보를 먼저 압축하므로,
후속 MHSA는 이미 정제된 토큰에 대해 더 효율적으로 교차 주의를 수행한다.

### 12.3 왜 Modality Validity Masking인가?

차량 환경에서는 센서 결손이 빈번하다:
- 야간: 카메라 품질 저하 → face 토큰 invalid
- 통화 중: 마이크에 대화 음성 → audio 감정 신뢰도 감소
- 장갑 착용: E4 센서 접촉 불량 → bio 토큰 invalid

key_padding_mask로 invalid 토큰을 자동 차단함으로써,
**사용 가능한 모달리티만으로 최선의 예측을 수행**하는 graceful degradation을
보장한다. 이는 실증 환경에서의 **시스템 가용성(availability)**을 높이는 핵심
설계이다.

### 12.4 참고 문헌

| 번호 | 인용 | 활용 위치 |
|------|------|---------|
| 1 | Ekman & Friesen (1978), "Facial Action Coding System" | AU 영역 정의 |
| 2 | Russell (1980), "A Circumplex Model of Affect" | Arousal-Valence 2차원 |
| 3 | Kreibig (2010), "Autonomic nervous system activity in emotion" | 생체 특징 설계 |
| 4 | Healey & Picard (2005), "Detecting stress during real-world driving" | EDA/HRV 운전 스트레스 |
| 5 | Dinges & Grace (1998), "PERCLOS: A valid psychophysiological measure" | 졸음 판정 표준 |
| 6 | Kendall et al. (CVPR 2018), "Multi-task learning using uncertainty" | MTL 가중치 |
| 7 | Lin et al. (ICCV 2017), "Focal Loss for dense object detection" | 클래스 불균형 |
| 8 | Hinton et al. (2015), "Distilling the knowledge in a neural network" | Knowledge Distillation |
| 9 | Ma et al. (ACL 2024), "emotion2vec" | 음성 감정 인코더 |
| 10 | Li et al. (CVPR 2022), "EfficientFormer" | Pool-FFN 구조 |
| 11 | Malik et al. (1996), "Heart rate variability standards" | HRV 분석 표준 |
| 12 | Boucsein (2012), "Electrodermal Activity" | EDA 신호 처리 |
| 13 | Mehta et al. (2022), "MobileViTv2" | Backbone 선정 |
| 14 | Lazarus (1991), "Emotion and Adaptation" | 스트레스 이론 |

---

> **문서 버전**: v1.0 (2026.03)
> **대상**: 시스템 이해 및 논문 작성 참고
