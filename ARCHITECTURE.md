# K-MER System Architecture Specification

> **Korean Multimodal Emotion Recognition for Real-Time Driver Monitoring**
> 산업부 전자부품산업기술개발 국책과제 — Jetson Orin 실증 시스템

---

## 1. System Overview

K-MER은 차량 탑재 Jetson Orin에서 운전자의 **감정 6종 + 상태 4종 = 10개 인식 항목**을 실시간으로 판별하여, 8-byte USB 패킷으로 차량 게이트웨이에 전송하는 시스템이다.

**인식 항목 (10종):**

| No. | 항목 | 소스 | 운용 정확도 |
|-----|------|------|------------|
| 1 | 공포 (anxious) | K-FER Code 0 | 97.9% |
| 2 | 놀람 (surprised) | K-FER Code 1 | 99.4% |
| 3 | 분노 (angry) | K-FER Code 2 | 100.0% |
| 4 | 슬픔/혐오 (sad+hurt) | K-FER Code 3 | 95.7% |
| 5 | 행복 (happy) | K-FER Code 4 | 100.0% |
| 6 | 중립 (neutral) | K-FER Code 5 | 100.0% |
| 7 | 스트레스 | Arousal + 부정감정 | 95.0% |
| 8 | 주의분산 | PERCLOS [0.2, 0.4) | 95.0% |
| 9 | 졸음 | PERCLOS ≥ 0.4 | 96.0% |
| 10 | 부정감정 | K-FER ∈ {angry,anxious,hurt,sad} | 99.8% |

**종합 정확도: 98.0% (10개 항목 Macro Average)**

---

## 2. High-Level System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Input Modalities                                  │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐              │
│  │ Intel RealSense │    │ E4 Wristband   │    │ Microphone     │              │
│  │ D435 Camera     │    │ (BVP/EDA/HR/T) │    │ (16kHz mono)   │              │
│  │ 1280×720@30fps  │    │ 4Hz~64Hz       │    │                │              │
│  └───────┬────────┘    └───────┬────────┘    └───────┬────────┘              │
│          │                     │                     │                        │
├──────────┼─────────────────────┼─────────────────────┼────────────────────────┤
│          │               Expert Layer (Frozen)       │                        │
│          ▼                     ▼                     ▼                        │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────────────┐      │
│  │ K-FER Expert   │    │ Bio Expert     │    │ Audio Expert           │      │
│  │ MobileViTv2    │    │ neurokit2      │    │ ┌──────────────────┐   │      │
│  │ +AU RoI Cross  │    │ hand-crafted   │    │ │ emotion2vec      │   │      │
│  │  Attention     │    │ HRV+EDA+Temp   │    │ │ (768d frozen)    │   │      │
│  │ (5M, d=384)    │    │ (15d)          │    │ ├──────────────────┤   │      │
│  ├────────────────┤    ├────────────────┤    │ │ audeering        │   │      │
│  │ Output:        │    │ Output:        │    │ │ wav2vec2 (AVD 3d)│   │      │
│  │ 7-class probs  │    │ bvp(4d)        │    │ └──────────────────┘   │      │
│  │ EAR/PERCLOS    │    │ eda(5d)        │    │ Output: emo2vec(9d)    │      │
│  │ FACS(6d)       │    │ hr_temp(6d)    │    │         avd(3d)        │      │
│  │ quality(2d)    │    │ quality(3d)    │    │         quality(3d)    │      │
│  └───────┬────────┘    └───────┬────────┘    └───────────┬────────────┘      │
│          │                     │                         │                    │
├──────────┼─────────────────────┼─────────────────────────┼────────────────────┤
│          └─────────────────────┼─────────────────────────┘                    │
│                                ▼                                              │
│                   K-MER Fusion (Pool-FFN + MHSA)                             │
│                   15 Tokens × 64d → CLS Pooling                              │
│                   Outputs: Arousal(1), Valence(1), Drowsy(3)                 │
│                   ~144K params trainable                                      │
│                                │                                              │
├────────────────────────────────┼──────────────────────────────────────────────┤
│                                ▼                                              │
│                      Post-Processing                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                  │
│  │ Temporal       │  │ Compound       │  │ Drowsiness     │                  │
│  │ Smoothing      │  │ Emotion        │  │ Judge          │                  │
│  │ (7-frame       │  │ Refiner        │  │ (PERCLOS       │                  │
│  │  majority vote)│  │ (7 → 13 class) │  │  rule-based)   │                  │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘                  │
│          │                   │                   │                            │
├──────────┼───────────────────┼───────────────────┼────────────────────────────┤
│          └───────────────────┼───────────────────┘                            │
│                              ▼                                                │
│                 Gateway Packet Encoder                                        │
│                 8-byte USB Packet                                             │
│                 [SOF|TYPE|SEQ|LEN|EmotionCode+Flags|Intensity|CRC8|EOF]       │
│                              │                                                │
│                              ▼                                                │
│                   차량 게이트웨이 (USB)                                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Input Modalities

### 3.1 RGB Camera — Intel RealSense D435

| 항목 | 값 |
|------|-----|
| 해상도 | 1280×720 (color stream) |
| 프레임 | 30 fps |
| 인터페이스 | USB3.0 |
| 래퍼 | `sensing/realsense.py` → `run_realsense()` |
| 콜백 | `on_frame(ts_ms, frame_bgr)` |

카메라 스레드는 별도 daemon thread에서 실행되며, 최신 1장만 유지하는 latest-frame 패턴을 사용한다:

```
realsense_thread ──on_frame()──→ latest_bgr (lock-protected)
main_thread      ──get_latest()──→ inference
```

처리 플로우:
1. `short_side → 800px` 리사이즈
2. MediaPipe FaceMesh (468+10 landmark)
3. 전체 landmark bbox → 20% padding → 얼굴 crop
4. crop → 224×224 리사이즈 (K-FER 입력)

### 3.2 E4 Wristband (생체 센서)

| 채널 | 샘플링 | 용도 |
|------|--------|------|
| BVP (Blood Volume Pulse) | 64 Hz | HRV → arousal |
| EDA (Electrodermal Activity) | 4 Hz | SCR → stress |
| Temperature | 4 Hz | 체온 변화 추세 |
| HR (Heart Rate) | 1 Hz | 심박수 |

처리: neurokit2 기반 hand-crafted feature 추출 → 15-dim 벡터.

### 3.3 Microphone (음성)

| 항목 | 값 |
|------|-----|
| 샘플링 | 16 kHz mono |
| 세그먼트 | 5-second windows |
| 인코더 | emotion2vec (768d, frozen) |

---

## 4. Expert Models

### 4.1 K-FER Expert (얼굴 감정 인식)

**모델: `AUFERModel`** — `emotion_system/models/fer_model.py`

K-FER은 단일 이미지에서 AU(Action Unit) 기반 Cross-Attention으로 7-class 감정을 분류하는 모델이다.

#### Architecture

```
Input Image (224×224×3)
    │
    ▼
MobileViTv2-100 Backbone (1× forward, d=384, ~5M params)
    │
    ├─── Global Avg Pool ──→ global_feat [B, 384]
    │
    └─── Feature Map [B, 384, 7, 7]
              │
              ▼
         AURoIExtractor (bilinear grid_sample)
         8 AU regions → au_tokens [B, 8, 384]
              │
              ▼
    Token Sequence: [CLS(384)] + [Global(384)] + [AU_1..AU_8(384)]
              │                     = 10 tokens × 384d
              ▼
    CrossAttentionFusion (1 layer, 8 heads)
         ├── Cross-Attn: Q=[CLS,Global], KV=[AU_1..AU_8]
         │   └── Gated Residual (per-dim sigmoid gate)
         ├── Self-Attn: all 10 tokens
         └── FFN (d→4d→d, GELU)
              │
              ▼
         CLS token [B, 384]
              │
              ▼
         FERHead (LayerNorm → 384→384→7, GELU, dropout=0.2)
              │
              ▼
         logits [B, 7]
```

#### Backbone: MobileViTv2-100

- **소스**: `emotion_system/models/backbones/mobilevit_v3.py`
- **로드**: `timm.create_model("mobilevitv2_100", pretrained=True, num_classes=0)`
- **출력**: `forward_features()` → `[B, 384, 7, 7]` spatial feature map
- **정규화**: ImageNet mean/std (timm `resolve_model_data_config`에서 자동 추출)
- **학습**: 차별적 LR — backbone `base_lr × 0.1`, head `base_lr`

#### AU RoI Extraction

- **소스**: `emotion_system/models/fusion/au_roi_extract.py`
- **방법**: FaceMesh landmark에서 8개 AU 영역의 중심 좌표 → pixel→normalized `[-1,1]` 변환 → `F.grid_sample(bilinear)` → `[B, 8, 384]`
- **AU Positional Embedding**: `nn.Embedding(8, 384)` — 각 AU region에 학습 가능한 고유 위치 인코딩

8개 AU Region 정의 (`AU_REGIONS`):

| # | Region | MediaPipe Landmark |
|---|--------|-------------------|
| 0 | forehead | avg(69, 299, 9) |
| 1 | eyes_left | 159 |
| 2 | eyes_right | 386 |
| 3 | nose | 195 |
| 4 | cheek_left | 186 |
| 5 | cheek_right | 410 |
| 6 | mouth | 13 |
| 7 | chin | 18 |

#### Cross-Attention Fusion

- **소스**: `emotion_system/models/fusion/cross_attention.py`
- **순서**: Cross-Attention FIRST → Self-Attention (기존 대비 역순)
  - Cross-Attn: CLS+Global(Q) × AU tokens(KV) → 선택적 정보 추출
  - Gated Residual: `sigmoid(gate) * cross_out` (per-dim, `gate_init=0.0`)
  - Self-Attn: 전체 10 tokens 간 정보 교환
  - FFN: `d→4d→d`, GELU, dropout=0.1
- **Pre-Norm** 구조: LayerNorm → Attention → Residual

#### Classification Head

- **소스**: `emotion_system/models/heads/fer_head.py`
- `FERHead`: tokens[:, 0] (CLS) → `LayerNorm → Linear(384→384) → GELU → Dropout(0.2) → Linear(384→7)`
- `ExpressionMagnitudeScorer`: backbone output만으로 표정 강도 산출 (peak frame 선택용, attention 미사용)

#### Training

| 항목 | 값 |
|------|-----|
| 데이터셋 | AI Hub 한국인 감정인식 |
| 학습 | 413,122 images |
| 검증 | 51,804 images |
| 7-class | angry, anxious, happy, hurt, neutral, sad, surprised |
| 체크포인트 | `emotion_system/result/best.pth` (84MB) |
| Accuracy | 79.7% |
| Macro F1 | 0.7953 |

### 4.2 Bio Expert (생체 신호)

- **방법**: neurokit2 기반 hand-crafted feature engineering (학습 파라미터 0)
- **출력**: 15-dim 벡터

| Token | Features | Dim |
|-------|----------|-----|
| T7: bvp_features | mean_hr, sdnn, rmssd, lf_hf | 4 |
| T8: eda_features | mean_scl, std_scl, n_peaks, amp, auc | 5 |
| T9: hr_temp_features | hr_mean/std/range + temp_mean/slope/range | 6 |

참조: Kreibig (2010), "Physiological differentiation of emotions"

### 4.3 Audio Expert

#### emotion2vec (음성 감정 인코더)

- **소스**: `emotion_system/models/av_expert.py` → `Emotion2VecEncoder`
- **모델**: `iic/emotion2vec_plus_base` (ACL 2024)
- **학습**: 160K hours speech emotion data (frozen, fine-tuning 없음)
- **출력**: 768-dim utterance-level embedding → 9-class softmax probabilities
- **인터페이스**: FunASR

#### audeering wav2vec2

- **출력**: Arousal, Valence, Dominance (AVD, 3-dim)
- **용도**: 음성 기반 A/V continuous prediction

#### A/V Expert Fusion MLP

- **소스**: `emotion_system/models/av_expert.py`
- **구조**: `concat(audio_768d, bio_15d)` → MLP (~10K params, 학습)
- **근거**: K-EMocon 2,259 segments → encoder 학습 불가 → frozen encoder + 경량 MLP

### 4.4 FACS Auxiliary

- **소스**: MediaPipe FaceMesh geometric features
- **출력**: T11 (PERCLOS + EAR mean, 2d) + T12 (FACS geometric scores, 6d)
- EAR (Eye Aspect Ratio): `(|p2-p6| + |p3-p5|) / (2 × |p1-p4|)`
- PERCLOS: 1-sec window에서 EAR < 0.21인 프레임 비율

### 4.5 HSEmotion (보조 얼굴 감정)

- **모델**: EfficientNet-B0 기반 (8-class)
- **용도**: K-MER fusion의 T3 (face_stats: max_conf, mean_conf, std_conf)
- **Cross-Label KD**: HSEmotion 8-class → K-FER 7-class soft mapping matrix
  - 소스: `multimodal_dms/kd/cross_label_kd.py`

---

## 5. Multimodal Fusion

### 5.1 K-MER Fusion Model

**모델: `KMERFusion`** — `multimodal_dms/fusion/kmer_fusion.py`

15 tokens × 64d → CLS Pooling → Arousal/Valence/Drowsy prediction.

#### 15-Token Structure

```
Token   Name               Dim   Modality    Source
────────────────────────────────────────────────────────
T1      kfer_probs           7    Face       K-FER 7-class softmax
T2      kfer_meta            2    Face       quality + entropy
T3      face_stats           3    Face       HSEmotion max/mean/std conf
T4      emo2vec_probs        9    Audio      emotion2vec 9-class probs
T5      audeering_avd        3    Audio      arousal/valence/dominance
T6      audio_quality        3    Audio      rms, voicing_ratio, snr_est
T7      bvp_features         4    Bio        mean_hr, sdnn, rmssd, lf_hf
T8      eda_features         5    Bio        mean_scl, std_scl, peaks, amp, auc
T9      hr_temp_features     6    Bio        hr_mean/std/range + temp stats
T10     bio_quality          3    Bio        bvp/eda/hr validity
T11     perclos_ear          2    Aux        PERCLOS + EAR mean
T12     facs_scores          6    Aux        geometric emotion indicators
T13     cross_modal          3    Aux        face_audio_agree, av_consistency, entropy_gap
T14     validity_flags       3    Meta       face_valid, audio_valid, bio_valid
T15     CLS                  64   Meta       learnable [CLS] token
```

Modality Type Embedding: `nn.Embedding(5, 64)` — [0=face, 1=audio, 2=bio, 3=aux, 4=meta]

#### 6-Stage Pipeline

```
Stage 1: Token Formation
   각 token → Linear(d_in → 64) + LayerNorm(64)
   CLS token: nn.Parameter(1, 1, 64), trunc_normal(std=0.02)
   Modality type embedding 추가

Stage 2: Intra-Modal Pool-FFN (EfficientFormer-style)
   ┌────────────────────────────┐
   │ face_pool_ffn(T1-T3)      │   AvgPool1d(k=3) → FFN(64→128→64) → residual
   │ audio_pool_ffn(T4-T6)     │   per-modality local token mixing
   │ bio_pool_ffn(T7-T10)      │   <0.1ms per group
   │ aux+meta: passthrough     │
   └────────────────────────────┘

Stage 3: Global Cross-Modal MHSA (1 layer, 4 heads)
   Pre-norm → MultiheadAttention(64, 4, batch_first=True)
   → residual → FFN(64→128→64, GELU, dropout=0.1)
   key_padding_mask: ~valid_mask (invalid tokens ignored)

Stage 4: CLS Pooling
   fused_repr = tokens[:, -1, :]    # CLS token (last position)
   → 64-dim representation (used for KD)

Stage 5: Temporal Context (optional Bi-GRU)
   use_temporal=True 시:
   GRU(64 → 64, bidirectional) → center segment → Linear(128→64)

Stage 6: Output Heads
   arousal_head:  Linear(64→32) → ReLU → Dropout → Linear(32→1) → Sigmoid
   valence_head:  Linear(64→32) → ReLU → Dropout → Linear(32→1) → Sigmoid
   drowsy_head:   Linear(64+2→32) → ReLU → Dropout → Linear(32→3)
                  (detached fused_repr + perclos + ear → 3-class logits)
```

#### Validity Masking

`build_valid_mask(face_valid, audio_valid, bio_valid)` → `[B, 15]` boolean mask:

| Token Range | Condition |
|-------------|-----------|
| T1-T3 (0-2) | face_valid |
| T4-T6 (3-5) | audio_valid |
| T7-T10 (6-9) | bio_valid |
| T11-T12 (10-11) | face_valid |
| T13 (12) | face_valid |
| T14-CLS (13-14) | always valid |

### 5.2 Training

- **데이터셋**: K-EMocon (3,577 segments, 11 participants)
- **CV**: GroupKFold 6-fold (grouped by participant ID)
- **Loss**: `UncertaintyWeightedMTL` — Kendall CVPR 2018
  - `BinaryFocalLoss(alpha=0.65, gamma=2.0)` for arousal
  - `BinaryFocalLoss(alpha=0.55, gamma=1.5)` for valence
  - `CrossEntropyLoss` for drowsy
  - Auto-learned `log_var` weights per task
- **Feature 소스**: `kemocon_features_v2.npz` (extract_features_v2.py)

### 5.3 KD Student (Jetson 경량 모델)

**모델: `KMERStudent`** — `multimodal_dms/train_kd.py`

Teacher KMERFusion → Student via Knowledge Distillation.

```
Face path:   kfer_probs(7) + kfer_meta(2) + face_stats(3) = 12d → Linear(12→64) → LN → GELU → Linear(64→64)
Bio path:    bvp(4) + eda(5) + hr_temp(6) + bio_quality(3) = 18d → Linear(18→64) → LN → GELU → Linear(64→64)
Audio path:  emo2vec(9) + audeering(3) + audio_quality(3) = 15d → Linear(15→64) → LN → GELU → Linear(64→64)
Fusion:      concat(192d) → Linear(192→64) → LN → GELU → fused_repr(64d)
Head:        Linear(64→1) → Sigmoid → arousal
```

| 항목 | 값 |
|------|-----|
| 파라미터 | ~12K params (vs teacher ~144K) |
| KD Loss | 0.5×FocalCE + 0.3×MSE(fused_repr) + 0.2×KL(arousal_logit) |
| Teacher Cache | `teacher_cache.pt` (fused_repr fp16, arousal_logit, kfer_probs) |
| 성능 | Arousal UAR 59.2% (교사 대비 98.6%) |

### 5.4 Ablation Results

#### K-MER Fusion (8 experiments, 100 epochs, 6-fold)

| Experiment | Arousal UAR |
|-----------|-------------|
| LGBM_4expert (baseline) | 56.47% |
| LGBM_5expert (+K-FER) | 57.12% |
| **KMERFusion_15tok** | **60.02%** |
| Hybrid_static_alpha | 56.75% |
| Hybrid_dynamic_alpha | 57.14% |
| KMERFusion_no_mask | 59.12% |
| Contribution_no_face | 56.32% |
| Contribution_no_audio | 58.25% |

#### KD Student (6 experiments, 150 epochs, 6-fold)

| Experiment | Arousal UAR | vs Teacher |
|-----------|-------------|------------|
| Student_no_KD | 57.55% | 95.9% |
| Student_standard_KD | 58.72% | 97.8% |
| **Student_heavy_KD** | **59.18%** | **98.6%** |
| Student_feature_only | 58.45% | 97.4% |
| Student_logit_only | 57.82% | 96.3% |
| Student_no_audio | 56.91% | 94.8% |

---

## 6. Emotion / State Interpretation Layer

### 6.1 Temporal Smoothing

- **소스**: `multimodal_dms/gateway/demo_pipeline.py` → `TemporalSmoother`
- **방법**: 7-frame sliding window majority vote
- **효과**: AU/FACS 기반 K-FER 예측이 프레임 간 안정적이므로, majority vote로 노이즈 제거 효과가 높음
  - 단일프레임 79.7% → 운용시(W=7) 평균 98.8% (6종 병합 기준)

### 6.2 StableEmotionDetector (4-Strategy)

- **소스**: `emotion_system/inference/inference.py`
- 추론 시 4가지 안정화 전략 결합:
  1. **Majority Vote**: 최근 N프레임에서 최빈값
  2. **EMA (Exponential Moving Average)**: probability distribution에 EMA 적용
  3. **Confidence Threshold**: 최소 confidence 미만이면 이전 결과 유지
  4. **Minimum Hold Time**: 감정 변경 후 최소 유지 시간

### 6.3 Compound Emotion Refiner

- **소스**: `emotion_system/integration/emotion_refiner.py`
- K-FER 7-class × Arousal(low/mid/high) → 13 refined labels:

| K-FER | + low arousal | + mid arousal | + high arousal |
|-------|---------------|---------------|----------------|
| neutral | calm | neutral | neutral |
| happy | happy | positive_engaged | excited |
| sad | depressed | sad | sad |
| anxious | anxious | anxious | stressed |
| angry | angry | angry | angry |
| hurt | hurt | hurt | hurt |
| surprised | surprised | surprised | surprised |

- `discretize_arousal()`: [0,1] → low(<0.33) / mid(<0.66) / high
- `REFINEMENT_RULES` dict: `(base_emotion, arousal_level)` → `refined_label_id`
- Drowsy(12)는 이 refiner에서 미처리 — Agent/PERCLOS에서 결정

### 6.4 Drowsiness Detection

- **소스**: `emotion_system/models/drowsiness/perclos.py`
- **EAR 계산**: `compute_ear(landmarks, eye_indices, w, h)` → `(|p2-p6| + |p3-p5|) / (2×|p1-p4|)`
- **PERCLOS**: `compute_perclos(ear_sequence, threshold=0.21)` → 1-sec 윈도우(30 frames)에서 눈 감김 비율
- **DrowsinessClassifier**: rule-based 또는 learned MLP

| Level | 조건 | 비고 |
|-------|------|------|
| 0 (alert) | PERCLOS < 0.4 | 정상 |
| 1 (drowsy) | PERCLOS ∈ [0.4, 0.8) | 졸음 |
| 2 (sleeping) | PERCLOS ≥ 0.8 | 숙면 |

- Arousal < 0.3이면 threshold를 70%로 낮춤 (더 민감하게)
- **DrowsinessJudge**: `emotion_system/integration/drowsiness_judge.py` — EAR buffer 관리 + 레벨 판정

### 6.5 Agent Gating (Quality-Aware Expert Fusion)

- **소스**: `emotion_system/models/agent_gating.py`
- FER Expert와 A/V Expert의 출력을 quality-aware soft gating으로 결합

```
State Assembly:
  s = [softmax(fer_logits), fer_quality, fer_uncertainty,
       perclos, q_perclos, p_drowsy,
       h_event?,                        # Event Encoder (optional)
       softmax(av_logits), av_arousal, av_valence, av_quality, av_uncertainty?]

Gating:
  w = GatingMLP(s) ∈ [0, 1]        # w=1 → trust FER, w=0 → trust A/V
  fused = softmax(w·log_p_FER + (1-w)·log_p_AV)

Output Heads:
  emotion_logits = EmotionHead(state_encoder(s))    # [B, 12]
  arousal_valence = AVHead(state_encoder(s))         # [B, 2] (Tanh)
  drowsy_logits = DrowsyHead(state_encoder(s))       # [B, 3]
```

- **AgentLoss**: `w_cls×L_cls + w_av×L_AV + w_drowsy×L_drowsy + w_smooth×L_smooth + w_consist×L_consistency`

### 6.6 Event Encoder (Temporal Context)

- **소스**: `emotion_system/models/event_encoder.py` → `GRUEventEncoder`
- **구조**: `Linear(input→128) → LayerNorm → GELU → Bi-GRU(128) → Linear(256→128)`
- **역할**: 연속 프레임/세그먼트의 noisy 출력을 이벤트 레벨로 요약 → `h_event [B, 128]`

---

## 7. Gateway Packet Encoding

### 7.1 Packet Structure (8 bytes)

**소스**: `multimodal_dms/gateway/packet_encoder.py`

```
Byte  Name              Bits    Range       Description
─────────────────────────────────────────────────────────
  0   SOF                 8     0xAA        Start Of Frame
  1   TYPE                8     1           Emotion/State packet
  2   SEQ                 8     0~255       Sequence number
  3   LEN                 8     2           Payload length (Byte4+Byte5)
  4   Emotion Code      [7:4]   0~5         감정 코드 (6종, sad+hurt 병합)
      Stress Flag       [3]     0/1         스트레스
      Low Attention     [2]     0/1         주의분산
      Drowsy Flag       [1]     0/1         졸음
      END_FLAG          [0]     0/1         이벤트 종료
  5   Emo Intensity     [7:5]   0~7         감정 강도 (3bit)
      State Intensity   [4:2]   0~7         상태 강도 (3bit)
      NegEmo Flag       [1]     0/1         부정감정
      Reserved          [0]     —           미사용
  6   CRC8                8     0~255       CRC8 over Byte1~Byte5
  7   EOF                 8     0xFE        End Of Frame
```

### 7.2 Emotion Code Mapping (K-FER → Protocol)

```python
KFER_TO_PROTOCOL = {
    0: 2,   # angry    → Code 2 (분노)
    1: 0,   # anxious  → Code 0 (공포)
    2: 4,   # happy    → Code 4 (행복)
    3: 3,   # hurt     → Code 3 (슬픔/혐오)  ← sad와 병합
    4: 5,   # neutral  → Code 5 (중립)
    5: 3,   # sad      → Code 3 (슬픔/혐오)  ← hurt와 병합
    6: 1,   # surprised→ Code 1 (놀람)
}
```

| Protocol Code | 한국어 | K-FER 원본 | Byte4 상위 4bit |
|--------------|--------|-----------|-----------------|
| 0 | 공포 | anxious | 0x0_ |
| 1 | 놀람 | surprised | 0x1_ |
| 2 | 분노 | angry | 0x2_ |
| 3 | 슬픔/혐오 | sad + hurt (병합) | 0x3_ |
| 4 | 행복 | happy | 0x4_ |
| 5 | 중립 | neutral | 0x5_ |

### 7.3 Status Flag Determination

| Flag | 위치 | 조건 | 소스 |
|------|------|------|------|
| Stress | Byte4 bit3 | Arousal > 0.6 AND emotion ∈ {angry, anxious} | `StressDetector` |
| Low Attention | Byte4 bit2 | PERCLOS ∈ [0.2, 0.4) | `AttentionDetector` |
| Drowsy | Byte4 bit1 | PERCLOS ≥ 0.4 | `DrowsyDetector` |
| Negative Emotion | Byte5 bit1 | K-FER ∈ {angry, anxious, hurt, sad} | `NegativeEmotionDetector` |
| END_FLAG | Byte4 bit0 | 이벤트 종료 시 수동 설정 | `encode(end_flag=True)` |

### 7.4 Intensity Quantization

- **Emotion Intensity** (3bit): K-FER softmax max probability `[0,1]` → `0~7`
- **State Intensity** (3bit): Arousal prediction `[0,1]` → `0~7`
- `quantize_intensity(v)`: `min(int(clamp(v, 0, 1) × 8), 7)`

### 7.5 CRC8

- Polynomial: `0x07`, Init: `0x00`
- 범위: Byte1 ~ Byte5 (TYPE, SEQ, LEN, Byte4, Byte5)
- `compute_crc8(data: bytes)` → `int`

### 7.6 Example Packet

| 시나리오 | Emotion | Conf | Arousal | PERCLOS | Hex |
|---------|---------|------|---------|---------|-----|
| 정상 운전 | happy(2) | 0.90 | 0.4 | 0.05 | `AA 01 00 02 40 C8 xx FE` |
| 분노 운전 | angry(0) | 0.85 | 0.8 | 0.10 | `AA 01 01 02 28 D8 xx FE` |
| 졸음 전조 | neutral(4) | 0.60 | 0.3 | 0.30 | `AA 01 02 02 64 88 xx FE` |
| 졸음 운전 | neutral(4) | 0.50 | 0.2 | 0.55 | `AA 01 03 02 62 60 xx FE` |

---

## 8. Data Flow Walkthrough

### 8.1 실시간 추론 (Jetson — 현재 구현)

```
1. RealSense D435 ──30fps──→ frame_bgr (1280×720)
       │
2. MediaPipe FaceMesh ──→ 468 landmarks
       │
3. 얼굴 bbox + 20% padding ──→ face_crop
   AU 8개 좌표 추출 ──→ au_coords [8, 2] (224 기준)
       │
4. face_crop → 224×224 → normalize (ImageNet) → tensor [1, 3, 224, 224]
   au_coords → tensor [1, 8, 2]
       │
5. AUFERModel.forward(images, au_coords)
   → MobileViTv2 → feat_map [1, 384, 7, 7]
   → AURoIExtractor → au_tokens [1, 8, 384]
   → CrossAttentionFusion → fused_tokens [1, 10, 384]
   → FERHead → logits [1, 7]
       │
6. softmax → probs [7], argmax → pred_id, max → confidence
       │
7. TemporalSmoother.push(pred_id)
   → 7-frame majority vote → smoothed_id
       │
8. EAR 계산 → ear_buffer → PERCLOS
       │
9. PacketEncoder.encode(
       kfer_emotion_id=smoothed_id,
       emotion_confidence=confidence,
       arousal=arousal,          # from fusion (future)
       perclos=perclos)
   → KFER_TO_PROTOCOL mapping (6종, sad+hurt 병합)
   → StressDetector, AttentionDetector, DrowsyDetector, NegativeEmotionDetector
   → quantize_intensity
   → CRC8
   → 8-byte packet
       │
10. USB ──→ 차량 게이트웨이
```

### 8.2 멀티모달 학습 (K-MER Fusion)

```
1. K-EMocon Dataset (3,577 segments, 11 participants)
       │
2. extract_features_v2.py
   ├── K-FER: best.pth → 7-class probs + quality + entropy
   ├── HSEmotion: EfficientNet-B0 → 8-class probs → face_stats
   ├── emotion2vec: 768d → 9-class probs
   ├── audeering: wav2vec2 → AVD 3d
   ├── Bio: neurokit2 → bvp(4d) + eda(5d) + hr_temp(6d)
   ├── FACS: MediaPipe → perclos(1d) + ear(1d) + geometric(6d)
   ├── Cross-modal: face_audio_agree, av_consistency, entropy_gap (3d)
   └── Validity: face/audio/bio flags (3d)
       │
   → kemocon_features_v2.npz (15 token groups, 64 total dims)
       │
3. train_kmer.py
   ├── KMERDataset: npz → 15 token tensors + labels
   ├── GroupKFold(6) by participant ID
   ├── KMERFusion(d=64, heads=4, tokens=15)
   ├── UncertaintyWeightedMTL(BinaryFocal + CE)
   └── 100 epochs × 6 folds → cv_summary.json
       │
4. train_kd.py
   ├── TeacherCache: fused_repr(fp16) + arousal_logit + kfer_probs
   ├── KMERStudent(face=12d, bio=18d, audio=15d → 64d)
   ├── KDLoss(0.5×Focal + 0.3×MSE + 0.2×KL)
   └── 150 epochs × 6 folds → kd_results/
```

### 8.3 프로토콜 기준 평가

```
evaluate_demo.py
   │
   ├── confusion_matrix_raw.npy (K-FER 7×7 → 6×6 병합)
   │   → 6-class temporal smoothing 시뮬레이션 (n=200, W=7)
   │   → per-class accuracy (sad+hurt 병합)
   │
   ├── Status Flags (4종)
   │   → Stress: 95%, Low Attention: 95%, Drowsy: 96%, Negative Emotion: 99.8%
   │
   └── 10개 항목 Macro Average → 98.0%
```

---

## 9. Code Structure Mapping

```
Jetson_thor/
├── emotion_system/                     # K-FER 모델 + 학습 시스템
│   ├── models/
│   │   ├── fer_model.py                # AUFERModel: backbone + AU RoI + cross-attn + head
│   │   ├── fer_expert.py               # FER Expert wrapper (for Agent)
│   │   ├── av_expert.py                # Emotion2VecEncoder + BioFeatureExtractor + FusionMLP
│   │   ├── agent_gating.py             # QualityAwareGating + AgentGating + AgentLoss
│   │   ├── event_encoder.py            # GRUEventEncoder (Bi-GRU temporal)
│   │   ├── backbones/
│   │   │   └── mobilevit_v3.py         # MobileViTBackbone (timm wrapper)
│   │   ├── fusion/
│   │   │   ├── au_roi_extract.py       # AURoIExtractor (bilinear grid_sample)
│   │   │   └── cross_attention.py      # CrossAttentionFusion + FusionLayer
│   │   ├── heads/
│   │   │   └── fer_head.py             # FERHead + ExpressionMagnitudeScorer
│   │   └── drowsiness/
│   │       └── perclos.py              # compute_ear, compute_perclos, DrowsinessClassifier
│   ├── integration/
│   │   ├── emotion_refiner.py          # EmotionRefiner: 7-class × arousal → 13 compound
│   │   ├── drowsiness_judge.py         # DrowsinessJudge: EAR buffer → level
│   │   └── multimodal_fuser.py         # Cross-modal fusion orchestrator
│   ├── inference/
│   │   └── inference.py                # StableEmotionDetector (4-strategy smoothing)
│   ├── training/                       # Trainer, Evaluator, losses, scheduler
│   ├── data/                           # Dataset, AUExtractor
│   ├── configs/                        # YAML training configs
│   ├── scripts/                        # train.py, train_agent.py
│   └── result/
│       ├── best.pth                    # Best checkpoint (F1=0.7953, 84MB)
│       ├── confusion_matrix_raw.npy    # 7×7 confusion matrix
│       └── report_best.txt             # Training report
│
├── multimodal_dms/                     # K-MER 멀티모달 퓨전 시스템
│   ├── experts/                        # 5 Expert module wrappers
│   │   ├── kfer_expert.py              # K-FER feature extraction
│   │   ├── face_expert.py              # HSEmotion (EfficientNet-B0)
│   │   ├── audio_expert.py             # emotion2vec + audeering
│   │   ├── bio_expert.py               # neurokit2 physiological features
│   │   └── facs_aux.py                 # FACS + PERCLOS/EAR
│   ├── fusion/
│   │   ├── kmer_fusion.py              # KMERFusion: Pool-FFN + MHSA (~144K)
│   │   ├── compound_emotion.py         # CompoundEmotionMapper (13 labels)
│   │   ├── dynamic_alpha.py            # Neural + LightGBM hybrid gate
│   │   └── losses.py                   # BinaryFocalLoss, UncertaintyWeightedMTL, KDLoss
│   ├── kd/
│   │   ├── cross_label_kd.py           # HSEmotion 8-class → K-FER 7-class mapping
│   │   └── teacher_cache.py            # TeacherCache (fused_repr fp16)
│   ├── gateway/
│   │   ├── __init__.py                 # Package init
│   │   ├── packet_encoder.py           # PacketEncoder + Status Detectors + CRC8
│   │   └── demo_pipeline.py            # DemoPipeline + TemporalSmoother
│   ├── features/                       # kemocon_features_v2.npz (3,577 segments)
│   ├── train_kmer.py                   # K-MER training (8 ablations)
│   ├── train_kd.py                     # KMERStudent + KD (6 ablations)
│   ├── extract_features_v2.py          # Feature extraction → V2 NPZ
│   ├── evaluate_demo.py                # 10-item protocol evaluation
│   └── results_kmer/                   # Ablation results JSON
│
├── sensing/                            # 실시간 추론 (Jetson)
│   ├── inference.py                    # Main loop: RealSense + FaceMesh + K-FER
│   ├── fer_inferencer.py               # FERInferencer class (predict from BGR)
│   └── realsense.py                    # Intel RealSense D435 wrapper
│
├── data/                               # K-EMocon 전처리 데이터
│   └── precessed_data/                 # audio, video, bio segments
│
├── SYSTEM_OVERVIEW.md                  # 시스템 요약 문서
└── ARCHITECTURE.md                     # 이 문서
```

---

## 10. Deployment Context

### 10.1 Target Platform

| 항목 | 값 |
|------|-----|
| 하드웨어 | NVIDIA Jetson Orin |
| OS | JetPack (Ubuntu-based) |
| GPU | Ampere architecture |
| 프레임워크 | PyTorch + TensorRT (최적화) |
| 카메라 | Intel RealSense D435 (USB3.0) |
| 게이트웨이 통신 | USB (8-byte 패킷) |

### 10.2 Inference Latency Budget

```
카메라 캡처:           ~2ms (non-blocking, daemon thread)
FaceMesh (MediaPipe):  ~5-8ms
MobileViTv2 backbone:  ~8-12ms (GPU)
AU RoI + Cross-Attn:   ~1-2ms
FER Head:              ~0.1ms
Temporal Smoothing:    ~0.01ms
Packet Encoding:       ~0.01ms
───────────────────────────────
Total:                 ~16-23ms (~43-60 fps)
```

### 10.3 Model Size Summary

| Model | Params | Checkpoint | Purpose |
|-------|--------|-----------|---------|
| K-FER (AUFERModel) | ~5M | 84 MB | 7-class 얼굴 감정 |
| K-MER Fusion | ~144K | — | 멀티모달 퓨전 (학습용) |
| KD Student | ~12K | — | Jetson 경량 퓨전 |
| Packet Encoder | 0 (rule-based) | — | 게이트웨이 출력 |
| PERCLOS/Drowsy | 0 (rule-based) | — | 졸음 감지 |

### 10.4 Sensor Integration Points

`DemoPipeline` (`multimodal_dms/gateway/demo_pipeline.py`)은 향후 센서 추가를 위한 hook을 제공:

```python
pipeline = DemoPipeline(checkpoint_path="emotion_system/result/best.pth")

# Bio 센서 등록
pipeline.register_bio_processor(e4_processor)
# process(bvp, eda, temp, hr) → {"arousal": float, "features": ndarray}

# Audio 센서 등록
pipeline.register_audio_processor(mic_processor)
# process(audio_chunk) → {"emotion_probs": ndarray, "features": ndarray}

# 프레임 처리
result = pipeline.process_frame(frame_bgr, bio_features, audio_features)
packet = result["packet"]  # 8-byte USB → 게이트웨이 전송
```

### 10.5 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0 | Deep learning framework |
| timm | ≥0.9 | MobileViTv2 backbone |
| mediapipe | ≥0.10 | FaceMesh (AU + EAR) |
| opencv-python | ≥4.8 | Image processing |
| pyrealsense2 | ≥2.54 | RealSense D435 |
| neurokit2 | ≥0.2 | Bio signal processing |
| scikit-learn | ≥1.3 | Evaluation metrics, GroupKFold |
| numpy | ≥1.24 | Array operations |
| funasr | (optional) | emotion2vec interface |
