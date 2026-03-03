# Multimodal Emotion & Drowsiness Recognition System

## Architecture Overview

```
Input Sources
├── RealSense Camera (RGB) ──→ FER Pipeline (NOW)
│                            ──→ Drowsiness/PERCLOS (NOW, from FaceMesh)
├── Microphone (Audio)       ──→ Arousal/Valence (NEXT)
├── Bio Sensors              ──→ Arousal/Valence (NEXT)
└── External Camera (YOLO)   ──→ Context Embedding (FUTURE)

Fusion
├── FER: 7 basic emotions (angry, disgust, fear, happy, neutral, sad, surprise)
├── A/V:  Arousal × Valence from audio + bio
├── Combined: 7 emotions × A/V scaling → 10 refined labels
├── Drowsiness: PERCLOS + Arousal → alert/drowsy/sleeping
└── Context: external situation text embedding → attention merge
```

## Directory Structure

```
emotion_system/
│
├── configs/                     # All configuration files
│   ├── default.yaml             # Base training config
│   ├── mobilevit_fer.yaml       # MobileViTv3 FER model config
│   └── jetson_deploy.yaml       # Jetson inference config (later)
│
├── data/                        # Data pipeline
│   ├── preprocessors/           # Stage 1-2: face crop + AU extraction
│   │   ├── face_detector.py     # YOLOv8 face detection & crop
│   │   ├── au_extractor.py      # MediaPipe FaceMesh → AU coordinates + PERCLOS features
│   │   └── build_csv.py         # Orchestrator: raw images → training CSV
│   │
│   ├── validators/              # Stage 3: data QC
│   │   └── csv_validator.py     # NaN/inf/missing file checks
│   │
│   ├── dataset.py               # PyTorch Dataset (CSV → tensors)
│   └── transforms.py            # Augmentation pipelines
│
├── models/                      # Model definitions (NO training logic)
│   ├── backbones/
│   │   ├── __init__.py
│   │   └── mobilevit_v3.py      # MobileViTv3 backbone wrapper (timm)
│   │
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── au_roi_extract.py    # Feature-map level AU RoI extraction
│   │   └── cross_attention.py   # Lightweight cross-attention fusion
│   │
│   ├── heads/
│   │   ├── __init__.py
│   │   ├── fer_head.py          # Emotion classification head
│   │   └── expr_magnitude.py    # Expression magnitude scorer (for peak selection)
│   │
│   ├── drowsiness/              # Drowsiness detection module
│   │   ├── __init__.py
│   │   └── perclos.py           # PERCLOS computation + drowsiness classifier
│   │
│   └── fer_model.py             # Full FER model assembly
│
├── training/                    # Training loop & utilities
│   ├── trainer.py               # Main training loop
│   ├── scheduler.py             # LR schedulers (warmup + cosine)
│   ├── losses.py                # Focal loss, class-balanced loss
│   └── evaluator.py             # Metrics, confusion matrix, reports
│
├── inference/                   # Deployment (Jetson / real-time)
│   ├── realtime_pipeline.py     # Full real-time pipeline (later)
│   ├── peak_selector.py         # 1-sec window peak frame selection
│   └── export_tensorrt.py       # ONNX → TensorRT conversion (later)
│
├── modalities/                  # Future multimodal branches
│   ├── audio/                   # Audio → Arousal/Valence
│   │   ├── feature_extract.py
│   │   └── av_predictor.py
│   │
│   ├── bio/                     # Biosensor → Arousal/Valence
│   │   ├── signal_process.py
│   │   └── av_predictor.py
│   │
│   └── context/                 # External YOLO → text → embedding
│       ├── scene_detector.py
│       └── context_encoder.py
│
├── integration/                 # Final multimodal fusion
│   ├── emotion_refiner.py       # 7 emotions × A/V → 10 refined labels
│   ├── drowsiness_judge.py      # PERCLOS + Arousal → drowsiness level
│   └── multimodal_fuser.py      # All modalities → final output
│
├── utils/                       # Shared utilities
│   ├── seed.py
│   ├── logging.py
│   └── checkpoint.py
│
├── scripts/                     # CLI entry points
│   ├── preprocess.py            # Run full preprocessing pipeline
│   ├── train.py                 # Run training
│   ├── evaluate.py              # Run evaluation only
│   └── demo.py                  # Quick demo / visualization
│
└── requirements.txt
```

## Quick Start (FER Training)

```bash
# 1. Preprocess: face crop + AU extraction + validation
python scripts/preprocess.py --config configs/default.yaml

# 2. Train
python scripts/train.py --config configs/mobilevit_fer.yaml

# 3. Evaluate
python scripts/evaluate.py --checkpoint outputs/best.pth --config configs/mobilevit_fer.yaml
```
