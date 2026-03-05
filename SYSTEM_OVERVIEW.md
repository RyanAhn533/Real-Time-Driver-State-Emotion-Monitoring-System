# K-MER: 멀티모달 운전자 감정인식 시스템

## 1. 프로젝트 개요

**한국형 멀티모달 감정인식 (Korean Multimodal Emotion Recognition)**
- 산업부 전자부품산업기술개발 국책과제
- 운전자 모니터링 시스템 (DMS) 실증
- 타겟 플랫폼: NVIDIA Jetson Orin

**시스템 목표:**
- 10개 인식 항목 종합 정확도 **94.4%**
- 감정 7종 + 졸음/스트레스/주의분산 감지
- 차량 게이트웨이 USB 패킷 실시간 전송

---

## 2. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                        센서 입력 (Sensing)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ RGB Camera   │  │ E4 Wristband │  │ Microphone   │              │
│  │ (RealSense)  │  │ (BVP/EDA/HR) │  │ (16kHz)      │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
├─────────┼──────────────────┼──────────────────┼─────────────────────┤
│         ▼                  ▼                  ▼                     │
│                     Expert Layer (Frozen)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ K-FER        │  │ Bio Expert   │  │ Audio Expert │              │
│  │ MobileViTv2  │  │ BVP/EDA/HR   │  │ emotion2vec  │              │
│  │ + AU Cross   │  │ feature eng. │  │ + audeering  │              │
│  │ Attention    │  │              │  │              │              │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤              │
│  │ 7-class prob │  │ bio features │  │ emo probs    │              │
│  │ EAR/PERCLOS  │  │ (15-dim)     │  │ A/V/D        │              │
│  │ AU/FACS      │  │              │  │ (1024-dim)   │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
├─────────┼──────────────────┼──────────────────┼─────────────────────┤
│         ▼                  ▼                  ▼                     │
│                   K-MER Fusion (~97K params)                       │
│  ┌─────────────────────────────────────────────────────┐           │
│  │ 15 Tokens × 64d                                     │           │
│  │ Pool-FFN (intra-modal) → MHSA (cross-modal)        │           │
│  │ → Arousal Head / Valence Head                       │           │
│  └──────────────────────────┬──────────────────────────┘           │
├─────────────────────────────┼───────────────────────────────────────┤
│                             ▼                                      │
│                    Post-Processing                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ Temporal     │  │ Compound     │  │ Drowsiness   │             │
│  │ Smoothing    │  │ Emotion      │  │ Judge        │             │
│  │ (7-frame MV) │  │ (13-class)   │  │ (PERCLOS)    │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
├─────────┼──────────────────┼──────────────────┼─────────────────────┤
│         ▼                  ▼                  ▼                     │
│               Gateway Packet Encoder                               │
│  ┌─────────────────────────────────────────────────────┐           │
│  │ 8-byte USB Packet                                   │           │
│  │ [SOF][TYPE][SEQ][LEN][EmotionCode+Flags][Int][CRC]  │           │
│  └──────────────────────────┬──────────────────────────┘           │
│                             ▼                                      │
│                    차량 게이트웨이 (USB)                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 모델 구성

### 3.1 K-FER (얼굴 감정 인식)

| 항목 | 내용 |
|------|------|
| Backbone | MobileViTv2-100 (5M params, d=384) |
| 핵심 기법 | AU RoI Cross-Attention (8 AU regions) |
| 학습 데이터 | AI Hub 한국인 감정인식 (413,122 학습 / 51,804 검증) |
| 출력 | 7-class (angry, anxious, happy, hurt, neutral, sad, surprised) |
| 성능 | Accuracy 79.7%, F1 0.7953 |
| 체크포인트 | `emotion_system/result/best.pth` (84MB) |

### 3.2 K-MER Fusion (멀티모달 퓨전)

| 항목 | 내용 |
|------|------|
| 구조 | Pool-FFN (intra-modal) + MHSA (cross-modal, 4 heads) |
| 입력 | 15 tokens × 64d (K-FER + HSEmotion + emotion2vec + audeering + Bio + FACS) |
| 파라미터 | ~97K params |
| 출력 | Arousal (binary), Valence (binary) |
| 성능 | Arousal UAR 60.0% (K-EMocon 6-fold CV) |

### 3.3 KD Student (Jetson 경량 모델)

| 항목 | 내용 |
|------|------|
| 구조 | Face(12→64) + Bio(18→64) + Audio(15→64) + Gated Fusion |
| 파라미터 | ~68K params |
| KD Loss | 0.3×FocalCE + 0.4×MSE(repr) + 0.3×KL(logit) |
| 성능 | Arousal UAR 59.2% (교사 대비 98.6%) |

### 3.4 졸음 감지 (PERCLOS)

| 항목 | 내용 |
|------|------|
| 방법 | Eye Aspect Ratio (EAR) + PERCLOS rule-based |
| 기준 | EAR < 0.21 → 눈 감김, PERCLOS ≥ 0.4 → 졸음 |
| 단계 | 0=alert, 1=drowsy, 2=sleeping |
| 참고 | NHTSA/FHWA 표준 (Dinges & Grace, 1998) |

---

## 4. 게이트웨이 USB 패킷 프로토콜

### 4.1 패킷 구조 (8 bytes)

| Byte | 이름 | Bit | 값/범위 | 설명 |
|------|------|-----|---------|------|
| 0 | SOF | 8 | 0xAA | Start Of Frame |
| 1 | TYPE | 8 | 1 | 감정/상태 패킷 |
| 2 | SEQ | 8 | 0~255 | 순번 |
| 3 | LEN | 8 | 2 | Payload 길이 |
| 4 | Emotion Code | 4 (bit7~4) | 0~6 | 감정 코드 |
|   | Stress Flag | 1 (bit3) | 0/1 | 스트레스 |
|   | Low Attention | 1 (bit2) | 0/1 | 주의 분산 |
|   | Drowsy Flag | 1 (bit1) | 0/1 | 졸음 |
|   | END_FLAG | 1 (bit0) | 0/1 | 이벤트 종료 |
| 5 | Emotion Intensity | 3 (bit7~5) | 0~7 | 감정 강도 |
|   | State Intensity | 3 (bit4~2) | 0~7 | 상태 강도 |
|   | Reserved | 2 (bit1~0) | - | 미사용 |
| 6 | CRC8 | 8 | 0~255 | Byte1~5 CRC |
| 7 | EOF | 8 | 0xFE | End Of Frame |

### 4.2 Emotion Code 매핑

| Code | 프로토콜 | K-FER 원본 | Byte4 상위 |
|------|---------|-----------|-----------|
| 0 | 공포 | anxious | 0x00 |
| 1 | 놀람 | surprised | 0x10 |
| 2 | 분노 | angry | 0x20 |
| 3 | 슬픔 | sad | 0x30 |
| 4 | 행복 | happy | 0x40 |
| 5 | 혐오 | hurt | 0x50 |
| 6 | 중립 | neutral | 0x60 |

### 4.3 Status Flag 판정 기준

| Flag | 조건 | 소스 |
|------|------|------|
| Stress (bit3) | 고각성 (arousal > 0.6) + 부정감정 (분노/공포) | K-MER Arousal + K-FER |
| Low Attention (bit2) | PERCLOS ∈ [0.2, 0.4) | FaceMesh EAR |
| Drowsy (bit1) | PERCLOS ≥ 0.4 | FaceMesh EAR |

### 4.4 Intensity 양자화

- Emotion Intensity (3bit): K-FER softmax max prob → [0,1] → 0~7
- State Intensity (3bit): Arousal 예측값 → [0,1] → 0~7

---

## 5. 실증 검증 성능 (10개 인식 항목)

### 5.1 항목별 정확도

| No. | 패킷 필드 | 인식 항목 | 단일프레임 | 운용시(W=7) |
|-----|----------|----------|-----------|------------|
| 1 | Code 0 | 공포 | 71.0% | 97.9% |
| 2 | Code 1 | 놀람 | 84.5% | 99.4% |
| 3 | Code 2 | 분노 | 87.8% | 100.0% |
| 4 | Code 3 | 슬픔 | 52.0% | 63.9% |
| 5 | Code 4 | 행복 | 97.2% | 100.0% |
| 6 | Code 5 | 혐오 | 72.2% | 96.8% |
| 7 | Code 6 | 중립 | 92.9% | 100.0% |
| 8 | bit3 | 스트레스 | — | 95.0% |
| 9 | bit2 | 주의 분산 | — | 95.0% |
| 10 | bit1 | 졸음 | — | 96.0% |

### 5.2 종합

| 그룹 | 평균 정확도 |
|------|-----------|
| Emotion Code 7종 | 94.0% |
| Status Flag 3종 | 95.3% |
| **종합 (10개 항목 Macro Average)** | **94.4%** |

### 5.3 멀티모달 퓨전 참조 성능 (K-EMocon 6-fold CV)

| 항목 | 성능 |
|------|------|
| Valence 정확도 | 90.5% |
| Valence F1 | 94.8% |
| Arousal UAR (K-MER) | 60.0% |
| KD Student Arousal UAR | 59.2% (교사 대비 98.6%) |

---

## 6. 디렉토리 구조

```
Jetson_thor/
├── emotion_system/              # K-FER 모델 + 학습 시스템
│   ├── models/
│   │   ├── fer_model.py         # AUFERModel (MobileViTv2 + AU Cross-Attention)
│   │   ├── fer_expert.py        # FER Expert wrapper
│   │   ├── av_expert.py         # Audio/Bio fusion expert
│   │   ├── agent_gating.py      # Quality-aware expert fusion
│   │   ├── backbones/           # MobileViTv2 backbone
│   │   ├── fusion/              # AU RoI extraction + Cross-Attention
│   │   ├── heads/               # FER classification head
│   │   └── drowsiness/perclos.py # PERCLOS + EAR
│   ├── integration/
│   │   ├── emotion_refiner.py   # 7-class → 13 compound
│   │   ├── drowsiness_judge.py  # Drowsiness level judge
│   │   └── multimodal_fuser.py  # Cross-modal fusion
│   ├── training/                # Trainer, evaluator, losses, scheduler
│   ├── data/                    # Dataset, AU extractor
│   ├── configs/                 # YAML training configs
│   ├── scripts/                 # train.py, train_agent.py, etc.
│   └── result/
│       ├── best.pth             # Best checkpoint (F1=0.7953, 84MB)
│       ├── confusion_matrix_raw.npy
│       └── report_best.txt
│
├── multimodal_dms/              # K-MER 멀티모달 퓨전 시스템
│   ├── experts/                 # 5 Expert modules
│   │   ├── kfer_expert.py       # K-FER wrapper
│   │   ├── face_expert.py       # HSEmotion (EfficientNet-B0, 8-class)
│   │   ├── audio_expert.py      # emotion2vec + audeering
│   │   ├── bio_expert.py        # Physiological features
│   │   └── facs_aux.py          # FACS auxiliary (EAR/PERCLOS + geometric)
│   ├── fusion/                  # Fusion models
│   │   ├── kmer_fusion.py       # KMERFusion (Pool-FFN + MHSA, ~97K)
│   │   ├── compound_emotion.py  # 13-class compound mapper
│   │   ├── dynamic_alpha.py     # Neural+LightGBM hybrid gate
│   │   └── losses.py            # MTL + KD losses
│   ├── kd/                      # Knowledge Distillation
│   │   ├── cross_label_kd.py    # HSEmotion→K-FER soft mapping
│   │   └── teacher_cache.py     # Teacher prediction cache
│   ├── gateway/                 # 게이트웨이 통신
│   │   ├── packet_encoder.py    # 8-byte USB 패킷 인코더
│   │   └── demo_pipeline.py     # End-to-end 데모 파이프라인
│   ├── features/                # Feature cache (V2, 3577 segments)
│   ├── train_kmer.py            # K-MER 학습 (8 ablations)
│   ├── train_kd.py              # KD Student 학습 (6 ablations)
│   ├── evaluate_demo.py         # 10개 항목 실증 평가
│   └── results_*/               # Ablation & demo results
│
├── sensing/                     # 실시간 추론 (Jetson)
│   ├── inference.py             # RealSense + FaceMesh + K-FER loop
│   ├── fer_inferencer.py        # FERInferencer class
│   └── realsense.py             # Intel RealSense wrapper
│
├── data/                        # K-EMocon 전처리 데이터
│   └── precessed_data/          # audio, video, bio segments
│
└── SYSTEM_OVERVIEW.md           # 이 문서
```

---

## 7. 센싱 파이프라인

### 7.1 현재 구현 (sensing/)

- **RealSense D435** → BGR 프레임 (1280×720@30fps)
- **MediaPipe FaceMesh** → AU 8개 좌표 + EAR
- **K-FER** → 7-class 감정 추론
- **Temporal Smoothing** → Majority vote (window=7)
- **Packet Encoding** → 8-byte USB → 게이트웨이

### 7.2 확장 포인트 (센싱 코드 추가 시)

```python
from gateway.demo_pipeline import DemoPipeline

pipeline = DemoPipeline(checkpoint_path="emotion_system/result/best.pth")

# Bio 센서 프로세서 등록
pipeline.register_bio_processor(e4_processor)

# Audio 프로세서 등록
pipeline.register_audio_processor(mic_processor)

# 프레임 처리
result = pipeline.process_frame(
    frame_bgr,
    bio_features=e4_processor.get_latest(),
    audio_features=mic_processor.get_latest(),
)
packet = result["packet"]  # 8-byte USB 패킷
```

추가 예정:
- **E4 Wristband**: BVP, EDA, TEMP, HR → Bio expert features
- **마이크 (16kHz)**: emotion2vec embedding → Audio expert features
- **K-MER Fusion**: 멀티모달 퓨전 → Arousal/Valence → Stress flag 정확도 향상

---

## 8. 주요 실험 결과

### 8.1 K-MER Ablation (8 experiments, 100 epochs, 6-fold)

| 실험 | A_UAR |
|------|-------|
| LGBM_4expert (baseline) | 56.47% |
| LGBM_5expert (+K-FER) | 57.12% |
| KMERFusion_15tok | **60.02%** |
| Hybrid_static_alpha | 56.75% |
| Hybrid_dynamic_alpha | 57.14% |
| KMERFusion_no_mask | 59.12% |
| Contribution_no_face | 56.32% |
| Contribution_no_audio | 58.25% |

### 8.2 KD Student Ablation (6 experiments, 150 epochs, 6-fold)

| 실험 | A_UAR | 교사 대비 |
|------|-------|---------|
| Student_no_KD | 57.55% | 95.9% |
| Student_standard_KD | 58.72% | 97.8% |
| Student_heavy_KD | **59.18%** | **98.6%** |
| Student_feature_only | 58.45% | 97.4% |
| Student_logit_only | 57.82% | 96.3% |
| Student_no_audio | 56.91% | 94.8% |
