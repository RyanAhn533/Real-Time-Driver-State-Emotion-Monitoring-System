# K-MER: Korean Multimodal Emotion Recognition for Real-Time Driver Monitoring

<p align="center">
  <img src="kmer_system_architecture.png" width="850" alt="K-MER System Architecture"/>
</p>

> **Korean Ministry of Trade, Industry and Energy — R&D Program**
> Real-Time Driver Emotion & State Monitoring System — NVIDIA Jetson Orin

---

## Research Context

Driver Monitoring Systems (DMS) traditionally focus on drowsiness detection or gaze tracking. However, real-world driving safety is also strongly affected by **driver emotional state**, including stress, anger, anxiety, and distraction.

K-MER proposes a **real-time multimodal emotion recognition system** designed for in-vehicle deployment:

- **Multimodal sensing** — vision + audio + physiological signals
- **Lightweight cross-modal fusion** — <150K parameters
- **Real-time edge deployment** — NVIDIA Jetson Orin platform
- **Vehicle gateway interface** — 8-byte USB packet protocol (spec as of 2026-03; later replaced by the Motrex 12-byte spec)

This system was developed under a **Korean Ministry of Trade, Industry and Energy R&D program** targeting real-world automotive integration.

---

## Overview

K-MER은 차량 내 카메라·생체센서·마이크를 통해 운전자의 **감정 6종 + 상태 4종 = 10개 항목**을 실시간으로 인식하여, 8-byte USB 패킷(2026-03 시점 규격; 이후 모트렉스 12-byte 규격으로 변경)으로 차량 게이트웨이에 전송하는 시스템입니다.

**종합 정확도 98.0%** (10개 인식 항목 Macro Average) — **시뮬레이션 추정치, 실측 아님**

> 아래 표의 수치는 실차·실증 측정값이 아니다. 감정 6종·부정감정은 K-FER 검증 confusion matrix 를 바탕으로 W=7 majority-vote temporal smoothing 을 몬테카를로 시뮬레이션(`multimodal_dms/evaluate_demo.py`, n=300)한 값이고, 스트레스·주의분산·졸음 3항목은 같은 스크립트에 상수(0.95/0.95/0.96)로 기입된 설계 목표치다.

| 구분 | 인식 항목 | 정확도 |
|------|----------|--------|
| Emotion Code 0 | 공포 (anxious) | 97.9% |
| Emotion Code 1 | 놀람 (surprised) | 99.4% |
| Emotion Code 2 | 분노 (angry) | 100.0% |
| Emotion Code 3 | 슬픔/혐오 (sad+hurt) | 95.7% |
| Emotion Code 4 | 행복 (happy) | 100.0% |
| Emotion Code 5 | 중립 (neutral) | 100.0% |
| Status Flag | 스트레스 | 95.0% |
| Status Flag | 주의분산 | 95.0% |
| Status Flag | 졸음 | 96.0% |
| Status Flag | 부정감정 | 99.8% |

---

## Key Contributions

1. **Multimodal Driver Emotion Recognition**
   — Vision, audio, and physiological signals combined in a unified fusion architecture

2. **AU RoI Cross-Attention for Facial Expression**
   — Action Unit region-aware attention with bilinear sampling over MobileViTv2 backbone

3. **Lightweight Fusion Architecture**
   — K-MER fusion module with only ~144K parameters (KD student: ~12K)

4. **Real-Time Edge Deployment**
   — End-to-end pipeline running on NVIDIA Jetson Orin (camera input 30fps; inference loop 10 Hz, see `sensing/config/sensing_config.yaml`)

5. **Vehicle Interface Protocol**
   — Driver emotion and state encoded into an 8-byte USB packet for automotive gateway systems (2026-03 spec; later replaced by the Motrex 12-byte spec)

---

## System Architecture

```
센서 입력 → Expert Layer (Frozen) → K-MER Fusion → Post-Processing → Gateway Packet → 차량
```

### 1. Input Modalities

| 센서 | 스펙 | 용도 |
|------|------|------|
| Intel RealSense D435 | 1280×720 @30fps | 얼굴 감정 인식 |
| ADI Study Watch (EVAL-HCRWATCH4Z, BLE) | PPG/EDA/Temp | 생리 신호 기반 각성도 (학습 데이터 K-EmoCon 은 E4 Wristband BVP/EDA/HR/Temp 기준) |
| Microphone | 16kHz mono | 음성 감정 인식 |

### 2. Expert Models

| Expert | 모델 | 출력 |
|--------|------|------|
| **K-FER** | MobileViTv2-100 + AU RoI Cross-Attention (5M params) | 7-class 감정 확률, FACS, EAR |
| **Bio** | neurokit2 hand-crafted features | HRV(4d) + EDA(5d) + HR/Temp(6d) |
| **Audio** | emotion2vec (768d frozen) + audeering wav2vec2 | 9-class 확률 + AVD(3d) |

### 3. K-MER Fusion

- **구조**: 15 Tokens × 64d → Pool-FFN (intra-modal) → MHSA (cross-modal) → CLS Pooling
- **파라미터**: ~144K (teacher) / ~12K (KD student)
- **출력**: Arousal, Valence, Drowsy level
- **학습**: K-EMocon 6-fold GroupKFold CV, Uncertainty-weighted MTL

### 4. Gateway Packet Protocol

8-byte USB 패킷 구조 (2026-03 시점 규격; 이후 모트렉스 12-byte 규격으로 변경됨 — 이 저장소 코드는 8-byte 기준):

```
[0xAA] [TYPE=1] [SEQ] [LEN=2] [Byte4: EmoCode(4)+Flags(4)] [Byte5: Intensity(6)+R(2)] [CRC8] [0xFE]
```

- **Emotion Code** (4bit): 공포(0), 놀람(1), 분노(2), 슬픔/혐오(3), 행복(4), 중립(5)
- **Status Flags**: Stress | Low Attention | Drowsy | END (Byte4) + NegEmo (Byte5)

---

## Project Structure

```
Jetson_thor/
├── emotion_system/              # K-FER 모델 + 학습 시스템
│   ├── models/
│   │   ├── fer_model.py         # AUFERModel (MobileViTv2 + AU Cross-Attention)
│   │   ├── av_expert.py         # emotion2vec + Bio fusion
│   │   ├── agent_gating.py      # Quality-aware expert gating
│   │   ├── backbones/           # MobileViTv2 backbone (timm)
│   │   ├── fusion/              # AU RoI extraction + Cross-Attention
│   │   ├── heads/               # FER classification head
│   │   └── drowsiness/          # PERCLOS + EAR
│   ├── integration/             # Emotion refiner (7→13), Drowsiness judge
│   ├── inference/               # StableEmotionDetector (4-strategy)
│   ├── training/                # Trainer, evaluator, losses
│   └── result/                  # best.pth (F1=0.7953)
│
├── multimodal_dms/              # K-MER 멀티모달 퓨전
│   ├── experts/                 # 5 Expert wrappers
│   ├── fusion/                  # KMERFusion (Pool-FFN + MHSA)
│   ├── kd/                      # Knowledge Distillation
│   ├── gateway/                 # Packet encoder + Demo pipeline
│   ├── train_kmer.py            # K-MER 학습 (8 ablations)
│   ├── train_kd.py              # KD Student 학습 (6 ablations)
│   └── evaluate_demo.py         # 10개 항목 프로토콜 평가
│
├── sensing/                     # 실시간 추론 (Jetson)
│   ├── inference.py             # RealSense + FaceMesh + K-FER loop
│   ├── fer_inferencer.py        # FERInferencer class
│   └── realsense.py             # Intel RealSense D435 wrapper
│
├── ARCHITECTURE.md              # 전체 시스템 아키텍처 명세
├── SYSTEM_OVERVIEW.md           # 시스템 요약
└── kmer_system_architecture.png # 아키텍처 다이어그램
```

---

## Key Results

### K-FER (Facial Emotion Recognition)

| 항목 | 값 |
|------|-----|
| Dataset | AI Hub 한국인 감정인식 (413K train / 52K val) |
| Backbone | MobileViTv2-100 (5M params, d=384) |
| Key Method | AU RoI Cross-Attention (8 regions, bilinear sampling) |
| Accuracy | 79.7% |
| Macro F1 | 0.7953 |
| + Temporal Smoothing (W=7) | 98.8% (6-class avg, sad+hurt 병합) |

### K-MER Fusion Ablation (K-EMocon 6-fold CV)

| Experiment | Arousal UAR |
|-----------|-------------|
| LGBM 4-expert baseline | 56.47% |
| **KMERFusion 15-token** | **60.02%** |
| KD Student (heavy KD) | 59.18% (teacher 98.6%) |

---

## Demo

Example output from the real-time inference pipeline:

```
[Frame 142] ──────────────────────────────────
  Emotion : 행복 (Happy)       confidence: 0.94
  Stress  : False
  Drowsy  : False
  NegEmo  : False
  Packet  : AA 01 05 02 41 8C 7F FE
──────────────────────────────────────────────
```

<!-- TODO: Add demo.mp4 or demo.gif for real-time visualization -->

---

## Quick Start

### Real-time Inference (Jetson)

```bash
cd sensing
python inference.py
```

### Evaluation (Protocol-based 10 items)

```bash
cd multimodal_dms
python evaluate_demo.py
```

### Gateway Packet Test

```python
from multimodal_dms.gateway import PacketEncoder

encoder = PacketEncoder()
packet = encoder.encode(
    kfer_emotion_id=2,        # happy
    emotion_confidence=0.92,
    arousal=0.7,
    perclos=0.1,
)
info = encoder.decode(packet)
print(info)  # {'emotion_name': '행복', 'stress': False, 'drowsy': False, ...}
```

### Demo Pipeline

```python
from multimodal_dms.gateway.demo_pipeline import DemoPipeline

pipeline = DemoPipeline(checkpoint_path="emotion_system/result/best.pth")
result = pipeline.process_frame(frame_bgr)

if result:
    print(result["emotion_ko"])   # "행복", "분노" 등
    print(result["drowsy"])       # True/False
    print(result["packet_hex"])   # 8-byte 패킷 hex
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥ 2.0 | Deep learning |
| timm | ≥ 0.9 | MobileViTv2 backbone |
| MediaPipe | ≥ 0.10 | FaceMesh (AU + EAR) |
| OpenCV | ≥ 4.8 | Image processing |
| pyrealsense2 | ≥ 2.54 | RealSense D435 |
| neurokit2 | ≥ 0.2 | Bio signal processing |
| scikit-learn | ≥ 1.3 | Evaluation, GroupKFold |

---

## References

- **emotion2vec**: Ma et al., "emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation", ACL 2024
- **MobileViTv2**: Mehta et al., "Separable Self-attention for Mobile Vision Transformers", 2022
- **PERCLOS**: Dinges & Grace, "PERCLOS: A Valid Psychophysiological Measure of Alertness", NHTSA/FHWA, 1998
- **Uncertainty MTL**: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018
- **Bio Features**: Kreibig, "Autonomic nervous system activity in emotion", Biological Psychology, 2010

---

## Citation

If you use this project in your research, please cite:

```bibtex
@article{ahn2026kmer,
  title   = {K-MER: Korean Multimodal Emotion Recognition for Real-Time Driver Monitoring},
  author  = {Ahn, Junyoung and Moon, Yeon-Kug},
  year    = {2026},
  note    = {Korean Ministry of Trade, Industry and Energy R\&D Program}
}
```

---

## License

This project is developed under the Korean Ministry of Trade, Industry and Energy R&D program (산업부 전자부품산업기술개발).
