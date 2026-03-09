# K-MER: Multimodal Driver Emotion Recognition — System Pipeline & Theoretical Foundations

> **Korean Multimodal Emotion Recognition for Real-Time Driver Monitoring**
> Korean Ministry of Trade, Industry and Energy R&D Program | NVIDIA Jetson Orin

---

## Table of Contents

1. [System Objective](#1-system-objective)
2. [End-to-End Pipeline Overview](#2-end-to-end-pipeline-overview)
3. [Modality-Specific Experts](#3-modality-specific-experts)
4. [Tokenized Multimodal Fusion](#4-tokenized-multimodal-fusion)
5. [Deployment-Level Decision Layer](#5-deployment-level-decision-layer)
6. [Knowledge Distillation](#6-knowledge-distillation)
7. [Training Objectives](#7-training-objectives)
8. [Design Rationale](#8-design-rationale)
9. [Experimental Notes & Operational Metrics](#9-experimental-notes--operational-metrics)

---

## 1. System Objective

### 1.1 Problem Statement

기존 Driver Monitoring System(DMS)은 졸음 검출 또는 시선 추적에 국한되어, 운전자의 **정서적 상태**(스트레스, 분노, 불안, 위축 등)가 주행 안전에 미치는 영향을 반영하지 못한다. 그러나 정서적 각성(emotional arousal)이 높은 상태에서의 위험 판단 오류율은 정상 상태 대비 2~4배 증가하며 (Lazarus, 1991), 이는 졸음만큼이나 사고 유발의 주요 요인이다.

### 1.2 Design Goal

K-MER 시스템은 다음 네 가지 제약 조건 하에서 운전자의 감정 및 상태를 실시간으로 인식하고, 차량 시스템에 전달하는 것을 목표로 한다:

| 제약 조건 | 설계 대응 |
|-----------|---------|
| Edge 연산 제한 (Jetson Orin) | Frozen expert + 경량 fusion (~144K params) |
| 센서 결손 가능성 (야간, 소음, 센서 탈착) | Validity masking 기반 graceful degradation |
| 차량 게이트웨이 인터페이스 | 8-byte USB 패킷 프로토콜 |
| 한국인 표정 특성 반영 | AI Hub 한국인 감정인식 데이터 학습 |

### 1.3 System Scope

K-MER 시스템은 얼굴, 음성, 생체, 안면기하 보조 분기의 **네 입력 경로로부터 서로 이질적인 affective evidence를 추출**한 뒤, 이를 **토큰 단위로 정렬**하여 경량 Transformer 기반 fusion 모듈에서 통합한다. 최종 출력은 감정 6종 + 상태 4종 = 10개 인식 항목으로 구성되며, 8-byte USB 패킷으로 차량 게이트웨이에 전송된다.

---

## 2. End-to-End Pipeline Overview

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         K-MER END-TO-END PIPELINE                            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  [Layer 1]  Sensor Input                                                     ║
║  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                      ║
║  │ RGB Camera   │   │ E4 Wristband │   │ Microphone   │                      ║
║  │ (RealSense)  │   │ (BVP/EDA/HR) │   │ (16kHz)      │                      ║
║  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                      ║
║         │                  │                   │                              ║
║  [Layer 2]  Modality-Specific Feature Extraction (Frozen)                    ║
║  ┌──────▼───────┐   ┌──────▼───────┐   ┌──────▼───────┐   ┌──────────────┐  ║
║  │ K-FER Expert │   │ Bio Expert   │   │ Audio Expert │   │ Auxiliary    │  ║
║  │ MobileViTv2  │   │ neurokit2    │   │ emotion2vec  │   │ Geometry     │  ║
║  │ +AU Cross-   │   │ hand-craft   │   │ +audeering   │   │ EAR/PERCLOS  │  ║
║  │  Attention   │   │ 0 params     │   │ frozen enc.  │   │ +FACS 6d     │  ║
║  │ ~5M params   │   │              │   │              │   │              │  ║
║  ├──────────────┤   ├──────────────┤   ├──────────────┤   ├──────────────┤  ║
║  │ T1: probs 7d │   │ T7: BVP  4d │   │ T4: e2v  9d │   │ T11: PER 2d │  ║
║  │ T2: meta  2d │   │ T8: EDA  5d │   │ T5: AVD  3d │   │ T12: FAC 6d │  ║
║  │ T3: face  3d │   │ T9: HRT  6d │   │ T6: AQ   3d │   │ T13: XMD 3d │  ║
║  │              │   │ T10: BQ  3d │   │              │   │ T14: FLG 3d │  ║
║  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘  ║
║         │                  │                   │                  │           ║
║         └──────────────────┴───────────────────┴──────────────────┘           ║
║                                     │                                        ║
║  [Layer 3]  Tokenized Multimodal Fusion (~144K params)                      ║
║  ┌──────────────────────────────────▼──────────────────────────────────┐      ║
║  │               K-MER Fusion Module                                    │      ║
║  │  14 Expert Tokens + [CLS] → 64d Projection → Modality Embedding    │      ║
║  │  → Pool-FFN (intra-modal) → MHSA (cross-modal, validity masking)  │      ║
║  │  → CLS Pooling → Arousal | Valence | Drowsy Heads                 │      ║
║  └──────────────────────────────────┬──────────────────────────────────┘      ║
║                                     │                                        ║
║  [Layer 4]  Deployment-Level Decision                                       ║
║  ┌──────────────────────────────────▼──────────────────────────────────┐      ║
║  │  Temporal Smoothing (7-frame MV) | StableEmotionDetector (4-strat) │      ║
║  │  Compound Emotion Refiner (7×A→13) | Status Flag Determination    │      ║
║  └──────────────────────────────────┬──────────────────────────────────┘      ║
║                                     │                                        ║
║  [Layer 5]  Vehicle Interface                                               ║
║  ┌──────────────────────────────────▼──────────────────────────────────┐      ║
║  │  Gateway Packet Encoder: 8-byte USB Protocol                        │      ║
║  │  [SOF][TYPE][SEQ][LEN][EmoCode+Flags][Intensity+NegEmo][CRC][EOF]  │      ║
║  └────────────────────────────────────────────────────────────────────┘      ║
║                                     │                                        ║
║                                     ▼                                        ║
║                           Vehicle Gateway (USB)                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

시스템은 **모델 수준(Layer 2~3)**과 **시스템 후처리 수준(Layer 4~5)**이 명확히 분리된다. 모델 수준은 학습된 파라미터에 의한 추론이며, 시스템 후처리 수준은 rule-based stabilization과 protocol encoding으로 구성된다. 이 분리는 성능 해석 시 frame-wise classification accuracy와 deployment-level operational accuracy를 구분하는 데 핵심적이다.

---

## 3. Modality-Specific Experts

### 3.1 Face Expert: K-FER (Korean Facial Emotion Recognition)

K-FER은 **AU RoI Cross-Attention**을 핵심 메커니즘으로 하는 얼굴 감정 분류기로, 단일 이미지에서 Action Unit(AU) 기반의 선택적 주의를 통해 7-class 감정을 분류한다.

#### 3.1.1 Backbone & Feature Extraction

MobileViTv2-100 (Mehta et al., 2022)을 backbone으로 사용한다. ImageNet pretrained weights로부터 fine-tuning하며, 단일 forward pass로 feature map `F ∈ R^(B×384×7×7)`과 global feature `g ∈ R^(B×384)`를 동시에 추출한다.

기존 AU 기반 FER 모델(POSTER++, DAtt-Net 등)은 각 AU 영역에 대해 독립적인 backbone forward를 수행하여 K개 AU에 대해 O(K·N)의 연산을 요구한다. K-FER은 이를 단일 forward의 feature map 위에서 `F.grid_sample(bilinear)`로 AU 영역 특징을 추출함으로써 O(N + K)로 축소한다.

#### 3.1.2 AU RoI Extraction

MediaPipe FaceMesh의 468개 landmark로부터 8개 AU 관심 영역의 중심 좌표를 계산하고, 이를 feature map 좌표계([-1, 1])로 정규화한 뒤 bilinear grid sampling으로 해당 위치의 특징 벡터를 추출한다.

| # | 영역 | FACS 대응 |
|---|------|-----------|
| 0 | Forehead (이마) | AU1+AU2 (내/외측 눈썹올림) |
| 1 | Eyes Left (왼눈) | AU5 (상안검 거상), AU7 (안검 긴장) |
| 2 | Eyes Right (오른눈) | AU5, AU7 |
| 3 | Nose (코) | AU9 (코 주름), AU10 (상순 거상) |
| 4 | Cheek Left (왼볼) | AU6 (볼 올림) |
| 5 | Cheek Right (오른볼) | AU6 |
| 6 | Mouth (입) | AU12 (입꼬리 당김), AU25 (입술 벌림) |
| 7 | Chin (턱) | AU17 (턱 올림) |

각 AU 영역에는 학습 가능한 positional embedding `E_au ∈ R^(8×384)`가 더해져, region identity를 인코딩한다.

#### 3.1.3 Cross-Attention Fusion Layer

10개 토큰 `[CLS, Global, AU_1, ..., AU_8]`으로 구성된 시퀀스에 대해, Cross-first, Self-second 순서의 attention을 1-layer 수행한다.

**Step 1 — Cross-Attention (Pre-Norm):**
- Q = [CLS, Global] (2 tokens), K/V = [AU_1, ..., AU_8] (8 tokens)
- CLS/Global 토큰이 AU 토큰에서 선택적으로 정보를 수집
- 분노 표정에서는 눈썹(AU0)·입(AU6)에 높은 attention weight, 행복 표정에서는 볼(AU4,5)·입(AU6)에 높은 attention weight가 관찰됨

**Gated Residual:**
```
gate = sigmoid(W_gate)        # Per-dimension learnable, shape=[384], init=0.0
output = Q + gate * cross_attn_out
```

`W_gate`가 0으로 초기화되므로 `sigmoid(0) = 0.5`에서 시작하여, 학습이 진행되면서 각 차원별로 global 정보(skip connection)와 local AU 정보(cross-attention output)의 최적 비율을 자동으로 학습한다. 이는 standard residual connection이 전체 차원에 동일한 가중치를 부여하는 것과 대비되는, **per-dimension adaptive information routing**이다.

**Step 2 — Self-Attention (Pre-Norm):**
- Q = K = V = all 10 tokens
- 전체 토큰 간 정보 교환, AU 간 상호작용(예: 눈썹 올림 + 입 벌림 = 놀람)을 학습
- Standard residual (gate 없음)

**Step 3 — Feed-Forward Network:**
```
LayerNorm(384) → Linear(384→1536) → GELU → Dropout(0.1) → Linear(1536→384) → Residual
```

이 **Cross-first, Self-second** 순서는 일반적인 transformer 구조와 반대이다. AU의 국소 정보를 먼저 CLS에 집약한 후 전역 판단을 내리는 것이 더 효과적임을 실험적으로 확인하였으며, 이는 facial expression이 본질적으로 local muscle action의 조합으로 정의된다는 FACS 이론(Ekman & Friesen, 1978)과 부합한다.

#### 3.1.4 Classification Head

```
CLS token [B, 384]
  → LayerNorm(384)
  → Linear(384→384) → GELU
  → Dropout(0.2)
  → Linear(384→7)
  → logits [B, 7] → softmax → probs [B, 7]
```

7-class: angry(0), anxious(1), happy(2), hurt(3), neutral(4), sad(5), surprised(6)

#### 3.1.5 Auxiliary Outputs

K-FER은 감정 확률 외에도 다음을 동시에 출력한다:
- **FACS 6d**: 기하학적 AU activation scores (입 벌림, 눈썹 올림, 입꼬리, 코 주름)
- **EAR**: Eye Aspect Ratio (눈 감김 정도)
- **PERCLOS**: Percentage of Eye Closure (졸음 지표)

이들은 fusion 모듈의 보조 토큰(T11~T13)으로 사용된다.

### 3.2 Audio Expert

음성 모달리티는 두 개의 frozen pre-trained encoder를 사용하여 상호 보완적인 정서 정보를 추출한다. 두 모델 모두 fine-tuning 없이 frozen encoder로 사용하는데, 이는 K-EMocon 데이터셋의 음성 세그먼트 수(~2,259)가 대규모 encoder의 재학습에 불충분하기 때문이다.

**emotion2vec (Ma et al., ACL 2024):**
- 자기지도 사전학습 기반 음성 감정 표현 모델
- 160K hours 음성 데이터에서 학습
- 1024-dim embedding → 9-class softmax (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown)
- K-MER에서는 9-class probability distribution을 그대로 토큰 T4로 사용

**audeering wav2vec2:**
- Wav2Vec2-large backbone (1024d) → Dense(1024,1024) + Tanh → Linear(1024,3)
- Arousal, Valence, Dominance ∈ [0, 1] 연속 출력
- Russell의 Circumplex Model of Affect (1980)에 기반한 연속적 정서 차원 추정

이산적 감정 범주(emotion2vec)와 연속적 정서 차원(audeering)을 동시에 사용함으로써, **범주적 판별력**과 **차원적 세밀함**을 모두 확보한다.

**Audio Quality (T6: 3d):**
- RMS energy (음성 존재 여부), Voicing ratio (유성음 비율), SNR estimation
- 품질 미달 시 fusion 모듈의 validity masking에 활용

### 3.3 Bio Expert (Physiological Features)

Bio Expert는 **학습 파라미터 없이(0 learnable params)** neurokit2 기반 hand-crafted feature engineering으로 생체 신호에서 정서적 각성의 생리학적 근거를 추출한다.

**BVP → HRV Features (T7: 4d):**

심박변이도(Heart Rate Variability)는 자율신경계 활성의 핵심 생리 지표이다.

| 특징 | 계산 | 생리학적 의미 |
|------|------|-------------|
| mean_hr | 60000 / mean(IBI) | 평균 심박수 |
| sdnn | std(IBI) | 전체 자율신경 활성도. 스트레스 시 감소 (Malik et al., 1996) |
| rmssd | sqrt(mean(diff(IBI)^2)) | 미주신경(부교감) 톤. 이완 시 증가 (Task Force, 1996) |
| lf_hf | PSD_LF / PSD_HF | 교감/부교감 균형. 스트레스 시 LF 증가 (Kreibig, 2010) |

**EDA → SCR Features (T8: 5d):**

피부전도도(Electrodermal Activity)는 **교감신경 전용 지표**로, 정서적 각성에 가장 직접적으로 반응한다 (Boucsein, 2012). 운전 중 스트레스 감지에서 가장 신뢰성 높은 생리 지표로 확인되었다 (Healey & Picard, 2005).

| 특징 | 의미 |
|------|------|
| mean_scl | 기저 피부전도 수준 |
| std_scl | 변동성 |
| n_peaks | SCR 발생 빈도 |
| mean_amp | SCR 평균 진폭 |
| auc | 곡선하면적 (총 각성량) |

**HR/Temperature Features (T9: 6d):**
- hr_mean, hr_std, hr_range: 심박수 통계
- temp_mean, temp_slope, temp_range: 피부 온도 추세
- 피부 온도의 slope은 이완/긴장 상태 전이를 포착 — 긴장(fight-or-flight) 시 말초 혈관 수축으로 피부 온도 감소 (Kreibig, 2010)

**Bio Quality (T10: 3d):**
- 각 채널별 유효성 (bvp_valid, eda_valid, hr_valid)
- 센서 탈착이나 움직임 아티팩트로 인한 데이터 손실을 정량화

### 3.4 Auxiliary Geometry Branch

K-FER의 CNN 기반 특징과 상호 보완적인, 기하학적 변형에 강인한 보조 특징을 제공한다.

**PERCLOS + EAR (T11: 2d):**
- EAR: Eye Aspect Ratio — 눈 감김 정도의 연속 지표
- PERCLOS: NHTSA/FHWA 표준 졸음 지표 (Dinges & Grace, 1998)

**FACS Geometric Scores (T12: 6d):**
- MediaPipe landmark 좌표에서 직접 계산하는 표정 지표
- Mouth openness, brow raise (L/R), lip corner (L/R), nose wrinkle

**Cross-Modal Agreement (T13: 3d):**
- Face-Audio valence agreement: 얼굴-음성 감정의 valence 일치도
- Arousal-Emotion consistency: 연속 각성도와 감정 강도의 일관성
- Entropy gap: 두 얼굴 감정 모델(K-FER, HSEmotion)의 확신도 교차 검증

이 메타 특징은 모달리티 간 예측의 합치 정도를 정량화하여, fusion 모듈이 불일치 상황에서 어떤 모달리티를 더 신뢰할지 학습하는 데 기여한다.

---

## 4. Tokenized Multimodal Fusion

### 4.1 Token Design Philosophy

K-MER Fusion은 각 모달리티의 출력을 하나의 벡터로 조기 결합(early fusion)하지 않고, **의미적으로 분해된 14개의 feature token**과 **1개의 [CLS] token**으로 유지한다. 이는 모달리티 내부의 정보 유형 차이(예: 감정 확률 vs 품질 지표)와 센서 결손 상황을 보존한 채, **token-level selective interaction**을 가능하게 하기 위함이다.

이 설계는 세 가지 이점을 제공한다:
1. **Fine-grained attention**: Fusion 모듈이 "K-FER 확률과 Audio AVD 사이의 관계"처럼 구체적인 cross-modal 상호작용을 학습할 수 있다
2. **Graceful degradation**: 센서 결손 시 해당 토큰만 masking하면 나머지 토큰의 정보 흐름에 영향을 주지 않는다
3. **Interpretability**: 어떤 토큰 간의 attention이 높은지 분석하여 모달리티 기여도를 파악할 수 있다

### 4.2 Token Formation & Projection

```
14 Expert Tokens + 1 [CLS] Token = 15 Tokens
각 토큰: Linear(d_in → 64) + LayerNorm(64) + Modality Type Embedding
```

| Token | Name | Input Dim | Modality Type (5종) |
|-------|------|-----------|-------------------|
| T1 | kfer_probs | 7 | Face (0) |
| T2 | kfer_meta | 2 | Face (0) |
| T3 | face_stats | 3 | Face (0) |
| T4 | emo2vec_probs | 9 | Audio (1) |
| T5 | audeering_avd | 3 | Audio (1) |
| T6 | audio_quality | 3 | Audio (1) |
| T7 | bvp_features | 4 | Bio (2) |
| T8 | eda_features | 5 | Bio (2) |
| T9 | hr_temp_features | 6 | Bio (2) |
| T10 | bio_quality | 3 | Bio (2) |
| T11 | perclos_ear | 2 | Aux (3) |
| T12 | facs_scores | 6 | Aux (3) |
| T13 | cross_modal | 3 | Aux (3) |
| T14 | validity_flags | 3 | Meta (4) |
| CLS | learnable | 64 | Meta (4) |

**Modality Type Embedding:** `nn.Embedding(5, 64)` — 같은 모달리티의 토큰이 공통된 modality context를 공유하도록 한다. 이는 ViT의 positional embedding에 대응하며, 토큰의 순서가 아닌 **소속 모달리티**를 인코딩한다.

**CLS Token:** `nn.Parameter([1, 1, 64])`, `trunc_normal(std=0.02)` — 모든 모달리티의 정보가 집약되는 global representation anchor이다.

### 4.3 Intra-Modal Pool-FFN

각 모달리티 그룹 내에서 지역적 토큰 혼합(local token mixing)을 수행한다. 이는 EfficientFormer (Li et al., CVPR 2022)에서 영감을 받은 설계로, Self-Attention보다 연산량이 매우 적으면서(O(1) per token) 유사한 intra-modal context aggregation 효과를 달성한다.

```
Pool-FFN(x) = x + FFN(LayerNorm(AvgPool1d(x)))

Face Group (T1, T2, T3):    AvgPool1d(kernel=3, stride=1, padding=1)
Audio Group (T4, T5, T6):   AvgPool1d(kernel=3, stride=1, padding=1)
Bio Group (T7~T10):         AvgPool1d(kernel=4, stride=1, padding=2)
Aux/Meta (T11~CLS):         Passthrough (혼합 없이 직통)

FFN = Linear(64 → 128) → GELU → Linear(128 → 64)
```

AvgPool1d는 인접 토큰 간의 이동 평균으로, 모달리티 내부의 관련 특징들을 부드럽게 혼합한다. 예를 들어, Face 그룹에서는 K-FER 확률(T1)과 메타 정보(T2), 얼굴 통계(T3)가 서로의 맥락을 공유한다. Aux/Meta 토큰은 이미 cross-modal 성격을 가지므로 intra-modal mixing을 적용하지 않는다.

### 4.4 Global Cross-Modal MHSA

모달리티 내 혼합이 끝난 후, **전체 15개 토큰에 대해 1-layer Multi-Head Self-Attention**을 수행하여 모달리티 간 교차 정보를 학습한다.

```
MHSA Configuration:
  embed_dim = 64
  num_heads = 4 (d_k = 16 per head)
  dropout = 0.1
  batch_first = True

Pre-Norm → MHSA → Residual
Pre-Norm → FFN(64→128→64, GELU, Dropout) → Residual
```

### 4.5 Validity Masking

실제 운용 환경에서는 센서 탈착, 소음, 움직임 아티팩트 등으로 특정 모달리티가 일시적으로 사용 불가할 수 있다.

```
Token Range      Valid Condition
──────────────────────────────
T1~T3   (Face)   face_valid
T4~T6   (Audio)  audio_valid
T7~T10  (Bio)    bio_valid
T11~T12 (Aux)    face_valid
T13     (XMD)    face_valid
T14, CLS         항상 valid
```

Invalid token은 **key-padding mask**를 통해 attention 대상에서 제외되며, 그 결과 모델은 사용 가능한 evidence만으로 추론을 수행한다. 이 설계는 missing modality를 별도 대체값(zero-imputation, mean-imputation 등)으로 채우는 방식보다, **잘못된 신호 전파를 줄이고** 실제 운용 환경에서의 **graceful degradation을 보장**한다.

대체값 기반 접근은 결측 모달리티에 "가짜 정보"를 주입하여 모델의 attention distribution을 왜곡할 수 있다. 반면, key-padding mask는 attention weight 계산에서 해당 토큰을 완전히 배제하여, 나머지 valid 토큰들의 상대적 attention weight가 자연스럽게 재분배되도록 한다.

### 4.6 CLS Pooling & Output Heads

```
fused_repr = tokens[:, -1, :]    # CLS token → [B, 64]

Arousal Head:
  Linear(64→32) → ReLU → Dropout(0.1) → Linear(32→1) → Sigmoid
  → arousal ∈ [0, 1]

Valence Head:
  Linear(64→32) → ReLU → Dropout(0.1) → Linear(32→1) → Sigmoid
  → valence ∈ [0, 1]

Drowsiness Head:
  Input: [fused_repr.detach() || perclos || ear_mean] = [B, 66]
  Linear(66→32) → ReLU → Dropout(0.1) → Linear(32→3)
  → {alert, drowsy, sleeping} logits
```

**Drowsiness Head의 `.detach()` 설계:**

졸음 판정은 PERCLOS/EAR이라는 직접적이고 해석 가능한 지표가 주도한다. `fused_repr`을 `.detach()`하는 것은 **gradient isolation** — 졸음 분류의 loss gradient가 fusion 모듈의 arousal/valence 표현 학습에 역전파되지 않도록 차단한다.

이는 multi-task learning에서 task 간 gradient interference 문제를 해결하는 설계로, 감정 인식(arousal/valence)과 졸음 판정이라는 본질적으로 다른 task의 학습 목표가 서로를 방해하지 않도록 보장한다. Drowsy head는 fused_repr의 정보는 활용하되, 그 학습 방향에는 영향을 주지 않는 **read-only consumer** 역할을 한다.

---

## 5. Deployment-Level Decision Layer

이 레이어는 학습된 모델의 frame-level 출력을 **운용 환경에 적합한 안정적 결정**으로 변환한다. 여기서 적용되는 모든 전략은 rule-based이며, 학습 파라미터를 포함하지 않는다.

### 5.1 Temporal Smoothing & StableEmotionDetector

단일 프레임 수준의 예측은 순간적 노이즈에 민감하므로, 실제 운용 단계에서는 **7-frame temporal voting**을 적용하여 이벤트 수준의 안정성을 높인다.

StableEmotionDetector는 4가지 안정화 전략을 계층적으로 적용한다:

**Strategy 1: Emotion-Group-Aware Majority Vote (W=7)**
```
SUSTAINED emotions (neutral, happy, sad):   min_votes = 6 (높은 안정성 요구)
TRANSIENT emotions (surprised, angry, ...): min_votes = 3 (빠른 반응 허용)
```
지속적 감정(행복, 중립)은 빈번하게 바뀌면 안 되지만, 일시적 감정(놀람, 분노)은 빠르게 반응해야 하므로 차별적 임계값을 적용한다.

**Strategy 2: EMA (Exponential Moving Average, alpha=0.3)**
```
P_ema(t) = 0.3 * P_new(t) + 0.7 * P_ema(t-1)
```
확률 분포 자체에 지수 이동 평균을 적용하여 부드러운 전이를 유도한다.

**Strategy 3: Confidence Threshold (0.5)**
```
if max(P_ema) < 0.5: keep previous prediction
```
불확실한 예측은 이전 결과를 유지하여, 모호한 전이 구간에서의 불안정한 출력을 방지한다.

**Strategy 4: Minimum Hold Time (0.5s)**
```
감정 변경 후 최소 0.5초간 유지 → 초당 2회 이상 변경 방지
```

### 5.2 Deployment-Level Accuracy의 해석

본 시스템에서 보고하는 운용 정확도(예: 98.8%)는 **temporal stabilization이 포함된 deployment-level operational accuracy**로 해석해야 하며, 모델의 frame-wise classification accuracy(79.7%)와 직접 비교되는 값이 아니다.

이 극적인 개선은 K-FER의 AU 기반 예측이 프레임 간 높은 일관성을 보이기 때문에 가능하다. 오류는 주로 비연속적(sporadic)이므로, majority vote로 효과적으로 제거된다. 이는 temporal voting의 효과가 모델의 기저 성능에 의존한다는 점을 함의한다 — 기저 모델의 오류가 체계적(systematic)이라면 temporal voting으로 개선할 수 없다.

### 5.3 Compound Emotion Refiner

Russell의 Circumplex Model을 기반으로, K-FER의 7-class 이산 감정에 Arousal 연속 값을 결합하여 13가지 세분화된 복합 감정으로 확장한다.

```
Arousal 이산화: Low(<0.33), Mid(0.33~0.66), High(>=0.66)

매핑 예시:
  happy + high arousal → excited
  happy + low arousal  → calm_happy
  angry + high arousal → enraged
  sad + high arousal   → distressed
  neutral + any        → neutral / calm
```

이 확장은 단순 "행복"과 "고각성 행복(excited)"을 구분하여, 운전자의 정서 상태를 더 세밀하게 파악하고, 위험 수준을 차등 판단하는 데 활용된다.

### 5.4 Status Flag Determination

| Flag | 위치 | 판정 논리 | 이론적 근거 |
|------|------|---------|-----------|
| Stress | Byte4 bit3 | arousal > 0.6 AND K-FER ∈ {angry, anxious} | 고각성 + 부정감정 = 스트레스 (Lazarus, 1991) |
| Low Attention | Byte4 bit2 | PERCLOS ∈ [0.2, 0.4) | 졸음 전 단계, 주의력 저하 시작 |
| Drowsy | Byte4 bit1 | PERCLOS >= 0.4 | NHTSA 표준 (Dinges & Grace, 1998) |
| Negative Emotion | Byte5 bit1 | K-FER ∈ {angry, anxious, hurt, sad} | 안전 운전에 부정적 영향을 미치는 감정군 |

### 5.5 Gateway Packet Encoding

최종 분석 결과를 8-byte USB 패킷으로 인코딩한다.

```
Byte 0    Byte 1    Byte 2    Byte 3    Byte 4          Byte 5            Byte 6    Byte 7
[0xAA]    [0x01]    [SEQ]     [0x02]    [EmoCode|Flags]  [EmoI|StI|NE|R]   [CRC8]    [0xFE]
 SOF       TYPE     0~255      LEN      4bit    4bit     3b  3b  1b  1b    B1~B5      EOF
```

**Emotion Code 매핑 (7-class → 6 codes, sad+hurt 병합):**

| K-FER Class | Protocol Code | 감정 |
|-------------|--------------|------|
| anxious(1) | Code 0 | 공포 |
| surprised(6) | Code 1 | 놀람 |
| angry(0) | Code 2 | 분노 |
| sad(5), hurt(3) | Code 3 | 슬픔/혐오 (병합) |
| happy(2) | Code 4 | 행복 |
| neutral(4) | Code 5 | 중립 |

sad+hurt 병합 근거: AI Hub 한국인 감정 데이터에서 sad↔hurt 간 혼동률이 32.7%로 매우 높으며, 이는 한국 문화권에서 "슬픔"과 "상처"의 표정이 유사하기 때문이다.

**Intensity 양자화:**
```
quantize(v) = min(floor(clamp(v, 0, 1) * 8), 7)

Emotion Intensity: K-FER softmax max probability → 3-bit (0~7)
State Intensity:   Arousal prediction → 3-bit (0~7)
```

---

## 6. Knowledge Distillation

Jetson Orin 엣지 배포를 위해, K-MER Fusion 교사(~144K params)의 지식을 KMERStudent(~12K params)로 증류한다.

### 6.1 Student Architecture

```
Face path:  [kfer_probs(7) + meta(2) + face_stats(3)] = 12d
            → Linear(12→64) → LN → GELU → Dropout → Linear(64→64)

Bio path:   [bvp(4) + eda(5) + hr_temp(6) + bio_q(3)] = 18d
            → Linear(18→64) → LN → GELU → Dropout → Linear(64→64)

Audio path: [e2v(9) + avd(3) + aq(3)] = 15d
            → Linear(15→64) → LN → GELU → Dropout → Linear(64→64)

Gating:     concat(192d)
            → Linear(192→96) → ReLU → Linear(96→192) → Sigmoid
            gated = concat * gate_weights

Fusion:     gated(192d) → Linear(192→64) → LN → GELU
            → fused_repr (64d) → Arousal Head
```

Student는 Pool-FFN과 MHSA를 제거하고, 단순 MLP + learned gating으로 대체한다. 이 구조적 차이에도 불구하고 교사 대비 98.6%의 성능을 유지하는 것은, KD loss의 representation-level과 logit-level 정렬이 구조적 차이를 보상함을 시사한다.

### 6.2 3-Component KD Loss

```
L_total = lambda_task * L_task + lambda_feat * L_feature + lambda_logit * L_logit

L_task    = BinaryFocalLoss(student_arousal, label)
L_feature = MSE(normalize(s_repr), normalize(t_repr))
L_logit   = KL(log_softmax(s_logit/tau), softmax(t_logit/tau)) * tau^2
```

- **L_task**: Student의 자체 task performance 유지
- **L_feature**: Teacher-Student의 64d representation 방향성 정렬 (L2 정규화 후 MSE)
- **L_logit**: Teacher의 soft prediction에 담긴 dark knowledge 전달 (Hinton et al., 2015). 온도 tau=4.0으로 softmax를 부드럽게 만들어, 교사가 "분노와 공포가 유사하게 어려운 케이스"라고 판단한 정보까지 학생에게 전달

최적 가중치: lambda_task=0.3, lambda_feat=0.4, lambda_logit=0.3, tau=4.0

---

## 7. Training Objectives

### 7.1 Uncertainty-Weighted Multi-Task Learning

K-MER Fusion은 3가지 태스크(Arousal, Valence, Drowsy)를 동시에 학습한다. Kendall et al. (CVPR 2018)의 동종 불확실성(homoscedastic uncertainty) 가중 기법을 적용한다:

```
L_total = SUM_i (1 / (2 * sigma_i^2)) * L_i + log(sigma_i)
```

sigma_i는 태스크 i의 학습 가능한 불확실성 파라미터이다:
- sigma_i가 작으면 → 해당 태스크 가중치 증가 (확실한 태스크에 집중)
- sigma_i가 크면 → 해당 태스크 가중치 감소 (불확실한 태스크 완화)
- log(sigma_i) 항이 sigma_i → infinity를 방지 (정규화 역할)

이 기법은 태스크 간 가중치를 수동 튜닝할 필요 없이, 학습 과정에서 자동으로 최적 균형을 찾는다.

### 7.2 Binary Focal Loss

클래스 불균형 문제를 해결하기 위한 Focal Loss (Lin et al., ICCV 2017):

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

gamma = 2.0: 쉬운 샘플(p_t ≈ 1)의 기여를 급격히 감소
alpha = 0.65 (arousal): 양성(high arousal) 클래스에 더 높은 가중치
```

운전 중 고각성(스트레스, 분노) 상태는 저각성(이완, 졸음) 대비 빈도가 낮으므로, Focal Loss가 희소한 고각성 상태의 학습에 효과적이다.

### 7.3 K-FER Training

| 항목 | 값 |
|------|-----|
| 데이터셋 | AI Hub 한국인 감정인식 (413K train / 52K val) |
| Loss | Focal Loss (gamma=2) + Label Smoothing (0.1) |
| Optimizer | AdamW (backbone LR × 0.1 차별적 학습률) |
| Scheduler | CosineAnnealing |

### 7.4 K-MER Fusion Training

| 항목 | 값 |
|------|-----|
| 데이터셋 | K-EMocon (3,577 segments, 11 participants) |
| 교차검증 | GroupKFold 6-fold (participant ID 그룹화, pair-aware) |
| Optimizer | AdamW |
| 학습률 | 1e-3 |
| Epochs | 100~150 (early stopping) |

---

## 8. Design Rationale

이 섹션에서는 K-MER 시스템의 주요 설계 결정에 대한 이론적 정당성을 기술한다.

### 8.1 Single-Forward AU Extraction

기존 AU 기반 FER 모델은 K개 AU 영역에 대해 각각 backbone forward를 수행한다. K-FER은 단일 forward의 feature map 위에서 `F.grid_sample`로 AU 영역을 추출함으로써, 동등한 AU 정보를 O(N + K) 연산으로 획득한다.

```
기존:  K × backbone(crop_i) → K × [B, C]    ← O(K·N)
K-FER: 1 × backbone(face) → grid_sample(K)  ← O(N + K)
```

Jetson Orin에서 MobileViTv2-100 1회 forward는 ~8ms이므로, 기존 방식(K=8)은 ~64ms → K-FER은 ~9ms로 약 7배 속도 향상을 달성한다. 이는 edge deployment에서의 실시간성 요구와 직접적으로 관련된다.

### 8.2 Per-Dimension Gated Residual

Standard residual connection(`x + f(x)`)은 전체 차원에 동일한 비율로 original과 transformed 정보를 결합한다. K-FER의 gated residual(`x + sigmoid(g) * f(x)`)은 각 차원별로 결합 비율을 학습하여, global feature의 어떤 차원은 유지하고 어떤 차원은 AU cross-attention 정보로 대체하는 **fine-grained information routing**을 수행한다.

초기화를 0으로 설정하는 것은, 학습 초기에 모델이 잘 학습된 pretrained backbone의 global feature를 과도하게 왜곡하지 않도록 보장하는 안전 장치이다.

### 8.3 Pool-FFN + MHSA 2단 구조

| 구조 | 모달리티 내 혼합 | 모달리티 간 교차 | 연산 복잡도 |
|------|---------------|---------------|-----------|
| MHSA only | O(N^2) | O(N^2) | 15^2 = 225 |
| Pool-FFN only | O(N) | 불가 | ~N |
| **Pool-FFN + MHSA** | O(N) | O(N^2) | **N + N^2 (정제된 토큰)** |

Pool-FFN이 모달리티 내 중복 정보를 먼저 압축하므로, 후속 MHSA는 이미 정제된 토큰에 대해 더 효율적으로 교차 주의를 수행한다. 이는 Transformer의 연산을 "local → global" 순서로 구조화하는 EfficientFormer의 철학과 일맥상통한다.

### 8.4 Validity Masking vs. Imputation

센서 결손 처리의 두 가지 대안적 접근:

| 접근 | 방식 | 문제점 |
|------|------|-------|
| Zero-imputation | 결측 토큰을 0으로 채움 | 0이 "신호 없음"이 아닌 "낮은 값"으로 해석될 수 있음 |
| Mean-imputation | 결측 토큰을 학습 평균으로 채움 | 실제 상태와 무관한 "평균적 정보"가 attention에 참여 |
| **Key-padding mask** | 결측 토큰을 attention에서 배제 | Valid 토큰의 attention weight 자연 재분배 |

K-MER은 key-padding mask 방식을 채택하여, 결측 모달리티의 "가짜 정보"가 다른 모달리티의 attention distribution을 왜곡하는 것을 원천 차단한다.

### 8.5 Detached Drowsy Head

Multi-task learning에서 auxiliary task의 gradient가 shared representation을 왜곡하는 문제는 널리 알려져 있다 (Yu et al., NeurIPS 2020). K-MER에서는 졸음 판정이 arousal/valence 표현 학습을 방해하지 않도록, drowsy head의 입력에서 fused_repr을 `.detach()`한다.

이 설계의 핵심 통찰은, 졸음 판정에는 PERCLOS/EAR이라는 **직접적이고 해석 가능한 지표**가 이미 존재하므로, fused_repr은 보조 맥락 정보로만 활용하되 그 학습에는 영향을 주지 않아야 한다는 것이다.

### 8.6 Frozen Expert 전략

Expert 모듈을 frozen으로 유지하는 것은 다음의 이론적·실용적 근거에 기반한다:

1. **데이터 효율성**: K-EMocon (~3,577 segments)은 소규모이므로, emotion2vec(160K hours 학습) 등의 대규모 encoder를 fine-tuning하면 과적합 위험이 크다
2. **도메인 지식 보존**: 각 expert가 보유한 전문 지식(음성 감정, 생리학적 각성 등)을 온전히 보존
3. **연산 효율**: Expert 출력을 사전 캐싱하면, Fusion 모듈만 반복 학습 가능 → 실험 주기 단축
4. **모듈성**: 새 센서/모델 추가 시 토큰만 추가하면 됨 (plug-and-play architecture)

---

## 9. Experimental Notes & Operational Metrics

### 9.1 K-FER Frame-Level Performance

| 항목 | 값 |
|------|-----|
| Dataset | AI Hub 한국인 감정인식 (413K train / 52K val) |
| Frame-level accuracy | 79.7% |
| Frame-level Macro F1 | 0.7953 |

### 9.2 K-MER Fusion Ablation (K-EMocon 6-fold CV)

| Experiment | Arousal UAR | 비고 |
|-----------|-------------|------|
| LGBM 4-expert baseline | 56.47% | Non-neural baseline |
| 3-Expert Pool-FFN | 57.83% | Face + Bio + Audio |
| 5-Expert 15-Token | 59.14% | + Aux + Meta |
| **Full KMERFusion** | **60.02%** | Pool-FFN + MHSA |

### 9.3 Knowledge Distillation Ablation

| Experiment | Arousal UAR | Teacher Fidelity |
|-----------|-------------|-----------------|
| Student (no KD, task only) | 57.55% | 95.9% |
| Student (standard KD) | 58.72% | 97.8% |
| **Student (heavy KD)** | **59.18%** | **98.6%** |
| Student (feature only) | 58.45% | 97.4% |
| Student (logit only) | 57.82% | 96.3% |
| Student (no audio) | 56.91% | 94.8% |

Feature KD와 Logit KD의 조합이 가장 효과적이며, 학생 모델이 교사 성능의 98.6%를 달성하면서 파라미터는 12.4%에 불과하다.

### 9.4 Deployment-Level Operational Accuracy

7-frame temporal voting + StableEmotionDetector 적용 후의 운용 수준 정확도:

| No. | 항목 | Frame-level | Operational (W=7) |
|-----|------|-------------|-------------------|
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
| | **Macro Average** | | **98.0%** |

이 수치는 **deployment-level operational accuracy**이며, frame-wise classification accuracy와는 구분되어야 한다.

### 9.5 References

| # | Citation | 활용 위치 |
|---|---------|---------|
| 1 | Ekman & Friesen (1978), "Facial Action Coding System" | AU 영역 정의, Cross-first 설계 근거 |
| 2 | Russell (1980), "A Circumplex Model of Affect" | Compound emotion, Arousal-Valence |
| 3 | Kreibig (2010), "Autonomic nervous system activity in emotion" | 생체 특징 설계, HRV/EDA/Temperature |
| 4 | Healey & Picard (2005), "Detecting stress during real-world driving" | EDA 운전 스트레스 검증 |
| 5 | Dinges & Grace (1998), "PERCLOS: A valid psychophysiological measure" | 졸음 판정 표준 |
| 6 | Kendall et al. (CVPR 2018), "Multi-task learning using uncertainty" | Uncertainty-weighted MTL |
| 7 | Lin et al. (ICCV 2017), "Focal Loss for dense object detection" | 클래스 불균형 처리 |
| 8 | Hinton et al. (2015), "Distilling the knowledge in a neural network" | Knowledge Distillation |
| 9 | Ma et al. (ACL 2024), "emotion2vec" | 음성 감정 인코더 |
| 10 | Li et al. (CVPR 2022), "EfficientFormer" | Pool-FFN 구조 영감 |
| 11 | Malik et al. (1996), "Heart rate variability standards" | HRV 분석 표준 |
| 12 | Boucsein (2012), "Electrodermal Activity" | EDA 신호 처리 |
| 13 | Mehta et al. (2022), "MobileViTv2" | Backbone 선정 |
| 14 | Lazarus (1991), "Emotion and Adaptation" | 스트레스 이론 |
| 15 | Yu et al. (NeurIPS 2020), "Gradient Surgery for Multi-Task Learning" | Detached head 설계 근거 |

---

> **Document Version**: v2.0 (2026.03)
> **Purpose**: 논문 Method / System Overview 전환용 이론서
