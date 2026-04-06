# K-MER 전체 파이프라인 상세 정리

---

## Stage 0: 센서 데이터 수집 (3개 스레드, 병렬)

### 0-1. 카메라 (realsense.py)

```
Intel RealSense D435
  시리얼: 021222070391
  해상도: 1280 × 720
  FPS: 30
  출력: BGR numpy array (720, 1280, 3) uint8

동작:
  pyrealsense2 파이프라인으로 color stream 캡처
  매 프레임마다 콜백 호출:
    on_frame(ts_ms=1710000000, frame_bgr=np.array(720,1280,3))
          │
          ▼
    Latest.set(ts_ms, frame_bgr)
    → 항상 최신 1장만 보관 (이전 프레임 덮어씀)
    → inference_loop이 10Hz로 읽어감
    → 30fps 중 20fps는 자연스럽게 버려짐 (최신만 사용)
```

### 0-2. 마이크 (rode.py)

```
RODE Wireless GO II
  샘플레이트: 48,000 Hz
  블록사이즈: 8,192 samples (약 0.17초)
  채널: 스테레오 → 모노 변환
  출력: float32 numpy array (8192,)

동작:
  sounddevice InputStream 콜백 방식
  디바이스 자동 검색: "Wireless GO II", "RØDE", "RODE" 키워드 매칭
  매 블록마다 콜백 호출:
    on_audio_chunk(ts_ms, mono_float32(8192,), sr=48000)
          │
          ▼
    Ring1D.push(ts_ms, mono_float32)
    → 링 버퍼 크기: 48000 × 2 = 96,000 samples (2초분)
    → inference_loop이 10Hz로 snapshot 호출
    → snapshot 시 최근 2초 오디오를 시간순으로 반환

  연결 끊김 시: 2회 재연결 시도 후 대기
```

### 0-3. 워치 (watch.py)

```
ADI Study Watch (Analog Devices)
  연결: BLE 동글 (VID=0x0456, PID=0x2CFE)
  MAC: F1-18-1C-93-7C-42
  프로토콜: adi_study_watch SDK

  3가지 데이터 스트림:

  PPG (광용적맥파):
    ADPD 센서 → 포토다이오드 2채널
    콜백: on_ppg(ts_ms, d1=12500.0, d2=8300.0)
           │
           ▼
    BioQueues.push_ppg(ts, d1, d2) → deque (최대 512개)

  EDA (전기피부반응):
    4-wire 임피던스 측정
    콜백: on_eda(ts_ms, real=0.4523)
           │
           ▼
    BioQueues.push_eda(ts, real) → deque (최대 512개)

  Temperature (피부 온도):
    서미스터 센서
    콜백: on_temp(ts_ms, skin_c=36.5)
           │
           ▼
    BioQueues.push_temp(ts, skin_c) → deque (최대 512개)

  BLE 연결 시: monkey-patch (_patched_ble_open)로 리눅스 시리얼 호환
```

---

## Stage 1: 공유 버퍼 → 추론 루프 진입 (10Hz)

```
ModelInputs (공유 상태 객체)
  ├── frame_main: Latest     → 최신 BGR 프레임 1장
  ├── audio:      Ring1D     → 최근 2초 오디오 (96,000 samples)
  └── bio:        BioQueues  → PPG/EDA/Temp 큐 (각 최대 512개)

inference_loop (10Hz = 0.1초마다 1회):
  매 사이클:
    1. frame = frame_main.get()        → (ts, BGR ndarray) or (0, None)
    2. ts, audio_buf = audio.snapshot() → (ts, float32 array up to 96000)
    3. ppg, eda, temp = bio.snapshot()  → (list of tuples × 3)
    4. result = inferencer.forward(frame, audio_buf, ppg, eda, temp)
```

---

## Stage 2: KMERInferencer.forward() — 핵심 추론

### 2-1. 얼굴 처리 (_process_face)

```
입력: BGR 프레임 (720, 1280, 3) uint8

Step 2-1-1: MediaPipe FaceMesh
  ┌────────────────────────────────────────────┐
  │ 짧은 변 기준 800px로 리사이즈              │
  │ (720→800 = 비율 1.11, 결과 약 800×1422)    │
  │                                            │
  │ FaceMesh(max_num_faces=1) 실행             │
  │ → 468개 랜드마크 좌표 검출                  │
  │                                            │
  │ 랜드마크에서 얼굴 bbox 추출:               │
  │   x_min, y_min, x_max, y_max              │
  │   + 20% padding                            │
  │                                            │
  │ 얼굴 영역 crop → 224×224로 리사이즈        │
  │ BGR→RGB 변환, /255.0 정규화                │
  │ → face_chw (3, 224, 224) float32           │
  │                                            │
  │ AU 좌표 8개 추출 (224×224 스케일):          │
  │   [0] 이마     = avg(landmark 69,299,9)    │
  │   [1] 왼눈     = landmark 159              │
  │   [2] 오른눈   = landmark 386              │
  │   [3] 코       = landmark 195              │
  │   [4] 왼볼     = landmark 186              │
  │   [5] 오른볼   = landmark 410              │
  │   [6] 입       = landmark 13               │
  │   [7] 턱       = landmark 18               │
  │ → au_coords (8, 2) float32                │
  └────────────────────────────────────────────┘

Step 2-1-2: KFERExpert.extract(face_chw, au_coords)
  ┌────────────────────────────────────────────┐
  │ AUFERModel (AU-guided FER 모델)            │
  │ 체크포인트: emotion_system/result/best.pth │
  │                                            │
  │ 입력: face_chw (3,224,224), au_coords(8,2) │
  │                                            │
  │ 내부:                                      │
  │   ResNet backbone → feature map            │
  │   AU attention (8개 영역에 집중)             │
  │   7-class classifier                       │
  │                                            │
  │ 출력:                                      │
  │   kfer_probs (7,) = softmax 확률           │
  │     [0] angry    = 0.03                    │
  │     [1] anxious  = 0.02                    │
  │     [2] happy    = 0.85   ← top-1          │
  │     [3] hurt     = 0.01                    │
  │     [4] neutral  = 0.05                    │
  │     [5] sad      = 0.02                    │
  │     [6] surprised= 0.02                    │
  │                                            │
  │   kfer_meta (2,) = [quality, entropy]      │
  │     quality = max(probs) = 0.85            │
  │     entropy = -Σ(p·log(p)) = 0.72          │
  │                                            │
  │   face_stats (3,) = [max, mean, std]       │
  │     = [0.85, 0.143, 0.29]                  │
  └────────────────────────────────────────────┘

Step 2-1-3: FACSAuxExpert.extract(face_chw, au_coords)
  ┌────────────────────────────────────────────┐
  │ 입력: face_chw (3,224,224), au_coords(8,2) │
  │                                            │
  │ EAR (Eye Aspect Ratio) 계산:               │
  │   왼눈 랜드마크 → EAR_left                  │
  │   오른눈 랜드마크 → EAR_right               │
  │   mean_EAR = (EAR_left + EAR_right) / 2   │
  │                                            │
  │ PERCLOS 계산:                              │
  │   최근 N프레임 중 EAR < 0.21인 비율         │
  │   예: 100프레임 중 15프레임 눈 감음          │
  │   PERCLOS = 0.15                           │
  │                                            │
  │ FACS AU 강도 계산:                          │
  │   6개 얼굴 영역의 표정 근육 활성화 강도      │
  │                                            │
  │ 출력:                                      │
  │   perclos_ear (2,) = [PERCLOS, mean_EAR]   │
  │     = [0.15, 0.28]                         │
  │                                            │
  │   facs_scores (6,) = AU 강도               │
  │     = [이마 0.3, 눈 0.5, 코 0.1,           │
  │        볼 0.2, 입 0.7, 턱 0.1]             │
  └────────────────────────────────────────────┘

얼굴 처리 최종 출력 (→ Stage 3으로):
  face_valid = True
  kfer_probs    (7,)  → Token T1
  kfer_meta     (2,)  → Token T2
  face_stats    (3,)  → Token T3
  perclos_ear   (2,)  → Token T11
  facs_scores   (6,)  → Token T12
```

### 2-2. 오디오 처리 (_process_audio)

```
입력: audio_buf (최대 96,000 samples, 48kHz, float32)

Step 2-2-1: Resample
  ┌────────────────────────────────────────────┐
  │ librosa.resample(audio_48k, 48000, 16000)  │
  │ 96,000 samples → 32,000 samples (2초@16kHz)│
  └────────────────────────────────────────────┘

Step 2-2-2: Audio Quality 계산
  ┌────────────────────────────────────────────┐
  │ audio_quality (3,):                        │
  │   [0] RMS  = √(mean(x²)) = 0.045          │
  │       → 소리 크기 (0=무음, 1=최대)          │
  │   [1] ZCR  = zero crossing rate = 0.12     │
  │       → 주파수 특성 (높을수록 고주파)        │
  │   [2] SNR  = signal-to-noise = 15.3 dB     │
  │       → 음질 지표                           │
  └────────────────────────────────────────────┘

Step 2-2-3: Emotion2VecExpert.extract(audio_16k)
  ┌────────────────────────────────────────────┐
  │ emotion2vec 모델 (FunASR 기반)             │
  │                                            │
  │ 입력: 16kHz 오디오 waveform                │
  │                                            │
  │ 출력:                                      │
  │   emo2vec_probs (9,) = softmax 확률        │
  │     [0] angry      = 0.05                  │
  │     [1] disgusted  = 0.02                  │
  │     [2] fearful    = 0.01                  │
  │     [3] happy      = 0.10                  │
  │     [4] neutral    = 0.65   ← top-1        │
  │     [5] other      = 0.05                  │
  │     [6] sad        = 0.03                  │
  │     [7] surprised  = 0.04                  │
  │     [8] unknown    = 0.05                  │
  └────────────────────────────────────────────┘

Step 2-2-4: AudeeringExpert.extract(audio_16k)
  ┌────────────────────────────────────────────┐
  │ wav2vec2 기반 audeering 모델               │
  │ (transformers 라이브러리)                   │
  │                                            │
  │ 입력: 16kHz 오디오 waveform                │
  │                                            │
  │ 출력:                                      │
  │   audeering_avd (3,):                      │
  │     [0] arousal   = 0.45  (각성도)         │
  │     [1] valence   = 0.62  (긍부정)         │
  │     [2] dominance = 0.55  (지배감)         │
  │   범위: 각각 [0, 1]                        │
  └────────────────────────────────────────────┘

오디오 처리 최종 출력 (→ Stage 3으로):
  audio_valid = True
  emo2vec_probs   (9,)  → Token T4
  audeering_avd   (3,)  → Token T5
  audio_quality   (3,)  → Token T6
```

### 2-3. 생체신호 처리 (_process_bio)

```
입력:
  ppg  = [(ts, d1, d2), (ts, d1, d2), ...]  최대 512개
  eda  = [(ts, real), ...]                   최대 512개
  temp = [(ts, skin_c), ...]                 최대 512개

Step 2-3-1: NPZ 변환
  ┌────────────────────────────────────────────┐
  │ _build_bio_npz():                          │
  │   PPG → numpy array (N, 3) [ts, d1, d2]   │
  │   EDA → numpy array (M, 2) [ts, real]     │
  │   Temp → numpy array (K, 2) [ts, skin_c]  │
  └────────────────────────────────────────────┘

Step 2-3-2: extract_bio_features_v2() (NeuroKit2 + scipy)
  ┌────────────────────────────────────────────┐
  │ BVP (Blood Volume Pulse) 분석:             │
  │   PPG d1 채널 → 심박 추출                  │
  │   bvp_features (4,):                       │
  │     [0] mean_hr  = 72.5  (평균 심박수)     │
  │     [1] sdnn     = 45.2  (심박 변이도)     │
  │     [2] rmssd    = 38.1  (연속 심박 차이)   │
  │     [3] lf_hf    = 1.8   (교감/부교감 비)  │
  │                                            │
  │ EDA (Electrodermal Activity) 분석:         │
  │   eda_features (5,):                       │
  │     [0] mean_scl   = 2.3  (평균 피부전도)  │
  │     [1] std_scl    = 0.4  (변동성)         │
  │     [2] n_peaks    = 3    (SCR 피크 수)    │
  │     [3] amplitude  = 0.8  (피크 진폭)      │
  │     [4] auc        = 12.5 (곡선 아래 면적) │
  │                                            │
  │ HR + Temperature 분석:                     │
  │   hr_temp_features (6,):                   │
  │     [0] hr_mean    = 72.5  (평균 심박)     │
  │     [1] hr_std     = 8.3   (심박 표준편차) │
  │     [2] hr_range   = 25.0  (심박 범위)     │
  │     [3] temp_mean  = 36.5  (평균 체온)     │
  │     [4] temp_slope = -0.02 (체온 변화율)   │
  │     [5] temp_range = 0.3   (체온 범위)     │
  │                                            │
  │ Bio Quality:                               │
  │   bio_quality (3,) = 품질 지표             │
  └────────────────────────────────────────────┘

생체신호 처리 최종 출력 (→ Stage 3으로):
  bio_valid = True
  bvp_features      (4,)  → Token T7
  eda_features      (5,)  → Token T8
  hr_temp_features  (6,)  → Token T9
  bio_quality       (3,)  → Token T10
```

---

## Stage 3: Feature Dict 구성 → KMERFusion 입력

```
Step 3-1: Cross-Modal Feature 계산
  ┌────────────────────────────────────────────┐
  │ cross_modal (3,):                          │
  │   [0] face_audio_agree                     │
  │       = K-FER valence와 Audeering valence  │
  │         의 일치도 (cosine similarity)       │
  │                                            │
  │   [1] av_consistency                       │
  │       = arousal-valence 일관성             │
  │         (두 모달리티가 동의하는 정도)        │
  │                                            │
  │   [2] entropy_gap                          │
  │       = face entropy - audio entropy       │
  │         (어느 모달리티가 더 확신하는지)      │
  └────────────────────────────────────────────┘

Step 3-2: Validity Flags
  ┌────────────────────────────────────────────┐
  │ validity_flags (3,):                       │
  │   [0] face_valid  = 1.0 or 0.0             │
  │   [1] audio_valid = 1.0 or 0.0             │
  │   [2] bio_valid   = 1.0 or 0.0             │
  └────────────────────────────────────────────┘

Step 3-3: Valid Mask 생성 (build_valid_mask)
  ┌────────────────────────────────────────────┐
  │ valid_mask (1, 15) boolean:                │
  │                                            │
  │ FULL 모드 (모든 센서 정상):                 │
  │   [T T T T T T T T T T T T T T T]         │
  │    ─Face─ ─Audio─ ──Bio── ─Aux─ Meta CLS   │
  │                                            │
  │ CAM_ONLY 모드 (카메라만):                   │
  │   [T T T F F F F F F F T T T T T]         │
  │    ─Face─ ─Audio─ ──Bio── ─Aux─ Meta CLS   │
  │         ↑ valid    ↑ masked (attention 제외)│
  └────────────────────────────────────────────┘

Step 3-4: Tensor 변환
  ┌────────────────────────────────────────────┐
  │ 각 feature를 torch.tensor + batch dim:     │
  │                                            │
  │ features = {                               │
  │   "kfer_probs":      tensor (1, 7),        │
  │   "kfer_meta":       tensor (1, 2),        │
  │   "face_stats":      tensor (1, 3),        │
  │   "emo2vec_probs":   tensor (1, 9),        │
  │   "audeering_avd":   tensor (1, 3),        │
  │   "audio_quality":   tensor (1, 3),        │
  │   "bvp_features":    tensor (1, 4),        │
  │   "eda_features":    tensor (1, 5),        │
  │   "hr_temp_features":tensor (1, 6),        │
  │   "bio_quality":     tensor (1, 3),        │
  │   "perclos_ear":     tensor (1, 2),        │
  │   "facs_scores":     tensor (1, 6),        │
  │   "cross_modal":     tensor (1, 3),        │
  │   "validity_flags":  tensor (1, 3),        │
  │ }                                          │
  │ valid_mask = tensor (1, 15) bool           │
  │                                            │
  │ 모두 GPU (CUDA)로 이동                      │
  └────────────────────────────────────────────┘
```

---

## Stage 4: KMERFusion 신경망 (약 144K 파라미터)

```
Step 4-1: Token Formation (14개 Linear Projector)
  ┌──────────────────────────────────────────────────────────┐
  │ 각 feature vector → Linear(input_dim, 64) → 64d token   │
  │                                                          │
  │ Token │ Feature         │ Input │ Linear      │ Output   │
  │ ──────┼─────────────────┼───────┼─────────────┼──────────│
  │  T1   │ kfer_probs      │  (7,) │ Linear(7,64)│  (64,)   │
  │  T2   │ kfer_meta       │  (2,) │ Linear(2,64)│  (64,)   │
  │  T3   │ face_stats      │  (3,) │ Linear(3,64)│  (64,)   │
  │  T4   │ emo2vec_probs   │  (9,) │ Linear(9,64)│  (64,)   │
  │  T5   │ audeering_avd   │  (3,) │ Linear(3,64)│  (64,)   │
  │  T6   │ audio_quality   │  (3,) │ Linear(3,64)│  (64,)   │
  │  T7   │ bvp_features    │  (4,) │ Linear(4,64)│  (64,)   │
  │  T8   │ eda_features    │  (5,) │ Linear(5,64)│  (64,)   │
  │  T9   │ hr_temp_features│  (6,) │ Linear(6,64)│  (64,)   │
  │  T10  │ bio_quality     │  (3,) │ Linear(3,64)│  (64,)   │
  │  T11  │ perclos_ear     │  (2,) │ Linear(2,64)│  (64,)   │
  │  T12  │ facs_scores     │  (6,) │ Linear(6,64)│  (64,)   │
  │  T13  │ cross_modal     │  (3,) │ Linear(3,64)│  (64,)   │
  │  T14  │ validity_flags  │  (3,) │ Linear(3,64)│  (64,)   │
  │  T15  │ CLS (learnable) │ (64,) │ 학습 파라미터│  (64,)   │
  │                                                          │
  │ 결과: tokens (1, 15, 64)                                 │
  └──────────────────────────────────────────────────────────┘

Step 4-2: Modality Type Embedding 추가
  ┌──────────────────────────────────────────────────────────┐
  │ 5가지 모달리티 타입 (각 64d learnable embedding):        │
  │                                                          │
  │   Type 0 (Face):  T1, T2, T3        → + face_embed(64)  │
  │   Type 1 (Audio): T4, T5, T6        → + audio_embed(64) │
  │   Type 2 (Bio):   T7, T8, T9, T10   → + bio_embed(64)   │
  │   Type 3 (Aux):   T11, T12          → + aux_embed(64)   │
  │   Type 4 (Meta):  T13, T14, T15     → + meta_embed(64)  │
  │                                                          │
  │ tokens = tokens + type_embedding                         │
  │ 결과: tokens (1, 15, 64)                                 │
  └──────────────────────────────────────────────────────────┘

Step 4-3: Intra-Modal Pool-FFN (EfficientFormer 스타일)
  ┌──────────────────────────────────────────────────────────┐
  │ 같은 모달리티 토큰끼리 먼저 혼합 (지역적 상호작용):      │
  │                                                          │
  │ Face Pool-FFN:                                           │
  │   T1,T2,T3 (3개) → AvgPool → FFN → 업데이트된 T1,T2,T3  │
  │   "K-FER 확률 + 메타 + 통계를 얼굴 내부에서 먼저 혼합"    │
  │                                                          │
  │ Audio Pool-FFN:                                          │
  │   T4,T5,T6 (3개) → AvgPool → FFN → 업데이트된 T4,T5,T6  │
  │   "오디오 감정 + AVD + 품질을 오디오 내부에서 먼저 혼합"   │
  │                                                          │
  │ Bio Pool-FFN:                                            │
  │   T7,T8,T9,T10 (4개) → AvgPool → FFN → 업데이트         │
  │   "심박 + EDA + 체온 + 품질을 생체 내부에서 먼저 혼합"     │
  │                                                          │
  │ Aux/Meta 토큰: Pool-FFN 없이 통과                        │
  │                                                          │
  │ 결과: tokens (1, 15, 64) — 모달리티 내부 정보 혼합 완료   │
  └──────────────────────────────────────────────────────────┘

Step 4-4: Global Multi-Head Self-Attention (MHSA)
  ┌──────────────────────────────────────────────────────────┐
  │ 모든 모달리티 간 상호작용 (1 layer):                     │
  │                                                          │
  │   Pre-LayerNorm                                          │
  │        │                                                 │
  │        ▼                                                 │
  │   MultiHeadAttention(d_model=64, n_heads=4, head_dim=16) │
  │        │                                                 │
  │        │  Q, K, V = Linear(64→64) × 3                    │
  │        │  4 heads × 16d = 64d                            │
  │        │                                                 │
  │        │  key_padding_mask = ~valid_mask                  │
  │        │  → 센서 누락 토큰은 attention에서 제외            │
  │        │  → CAM_ONLY면 Audio/Bio 토큰 무시                │
  │        │                                                 │
  │        ▼                                                 │
  │   Residual + Dropout(0.1)                                │
  │        │                                                 │
  │        ▼                                                 │
  │   Pre-LayerNorm                                          │
  │        │                                                 │
  │        ▼                                                 │
  │   FFN: Linear(64→256) → GELU → Dropout → Linear(256→64) │
  │        │                                                 │
  │        ▼                                                 │
  │   Residual + Dropout(0.1)                                │
  │                                                          │
  │ 결과: tokens (1, 15, 64) — 모든 모달리티 정보 융합 완료   │
  └──────────────────────────────────────────────────────────┘

Step 4-5: CLS Token Pooling
  ┌──────────────────────────────────────────────────────────┐
  │ T15 (CLS 토큰) 추출                                     │
  │                                                          │
  │ tokens[:, -1, :]  →  fused_repr (1, 64)                  │
  │                                                          │
  │ CLS 토큰은 self-attention을 통해                         │
  │ 모든 14개 feature 토큰의 정보를 응축하고 있음              │
  └──────────────────────────────────────────────────────────┘

Step 4-6: Output Heads (3개)
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │ Arousal Head:                                            │
  │   fused_repr (64)                                        │
  │   → Linear(64, 32) → ReLU → Dropout(0.1)                │
  │   → Linear(32, 1) → Sigmoid                             │
  │   → arousal = 0.45  (범위 0~1)                           │
  │   의미: 각성도 (0=졸림/평온, 1=흥분/긴장)                 │
  │                                                          │
  │ Valence Head:                                            │
  │   fused_repr (64)                                        │
  │   → Linear(64, 32) → ReLU → Dropout(0.1)                │
  │   → Linear(32, 1) → Sigmoid                             │
  │   → valence = 0.62  (범위 0~1)                           │
  │   의미: 감정가 (0=부정, 1=긍정)                           │
  │                                                          │
  │ Drowsy Head:                                             │
  │   입력 = concat(fused_repr.detach(), perclos_ear)        │
  │        = (64 + 2) = 66d                                  │
  │   → Linear(66, 32) → ReLU → Dropout(0.1)                │
  │   → Linear(32, 3)                                        │
  │   → 3-class logits → argmax                              │
  │   → drowsy = 0  (0=정상, 1=졸림, 2=수면)                 │
  │                                                          │
  │   .detach() 이유: arousal gradient가                      │
  │   drowsy 판정에 역전파되는 것을 방지                      │
  │                                                          │
  │   perclos_ear 직접 입력: PERCLOS는 졸음의                 │
  │   가장 직접적 지표이므로 별도로 추가 입력                  │
  │                                                          │
  └──────────────────────────────────────────────────────────┘

KMERFusion 최종 출력:
  arousal = 0.45   (연속값 0~1)
  valence = 0.62   (연속값 0~1, None if not trained)
  drowsy  = 0      (이산값 0/1/2)
```

---

## Stage 5: CompoundEmotionMapper (규칙 기반, 학습 없음)

```
입력:
  kfer_top1_id = 2 (happy, K-FER softmax argmax)
  arousal      = 0.45
  is_drowsy    = False (drowsy ≥ 1이면 True)

Step 5-1: Drowsy 우선 체크
  is_drowsy == True → 즉시 (12, "drowsy") 반환, 아래 스킵

Step 5-2: Arousal 이산화
  arousal = 0.45 → "mid" (0.33 ≤ 0.45 < 0.66)

Step 5-3: 매핑 테이블 참조
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  K-FER 감정   │  low (<0.33)  │  mid          │  high  │
  │  ─────────────┼──────────────┼──────────────┼────────  │
  │  neutral (4)  │  1:calm      │  0:neutral   │ 0:neutral│
  │  happy   (2)  │  2:happy     │  3:pos_engaged│ 4:excited│
  │  sad     (5)  │  6:depressed │  5:sad       │  5:sad   │
  │  anxious (1)  │  7:anxious   │  7:anxious   │ 8:stressed│
  │  angry   (0)  │  9:angry     │  9:angry     │  9:angry │
  │  hurt    (3)  │ 10:hurt      │ 10:hurt      │ 10:hurt  │
  │  surprised(6) │ 11:surprised │ 11:surprised │11:surprised│
  │                                                         │
  │  이 예시: happy + mid → (3, "positive_engaged")         │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

최종 출력:
  compound_id    = 3
  compound_label = "positive_engaged"
```

---

## Stage 6: KMERInferencer.forward() 최종 반환

```
여기까지가 kmer_inferencer.py의 forward() 출력:

result = {
    "arousal":          0.45,              # KMERFusion arousal head
    "valence":          0.62,              # KMERFusion valence head
    "drowsy":           0,                 # KMERFusion drowsy head (0/1/2)
    "compound_id":      3,                 # CompoundEmotionMapper
    "compound_label":   "positive_engaged",# CompoundEmotionMapper
    "kfer_emotion":     "happy",           # K-FER top-1 label
    "kfer_confidence":  0.85,              # K-FER top-1 softmax prob
    "face_detected":    True,              # MediaPipe 얼굴 검출 여부
}
```

---

## Stage 7: Temporal Smoothing (E2E Pipeline)

```
위 result가 E2EPipeline.process_cycle()로 전달됨

MultimodalTemporalSmoother.smooth(result):

이산 값 → Majority Vote (슬라이딩 윈도우 다수결):
  ┌──────────────────────────────────────────────────────────┐
  │ kfer_emotion (window=7):                                 │
  │   최근 7프레임: [happy, happy, happy, neutral,           │
  │                  happy, happy, happy]                    │
  │   다수결: happy (6/7 = 85.7%)                            │
  │   → smoothed_kfer_emotion = "happy"                      │
  │   → smoothed_emotion_confidence = 0.857                  │
  │                                                          │
  │ drowsy (window=5):                                       │
  │   최근 5프레임: [0, 0, 0, 0, 0]                          │
  │   다수결: 0 (5/5)                                        │
  │   → smoothed_drowsy = 0                                  │
  │                                                          │
  │ compound_id (window=7):                                  │
  │   최근 7프레임: [3, 3, 3, 0, 3, 3, 3]                    │
  │   다수결: 3 (6/7)                                        │
  │   → smoothed_compound_id = 3                             │
  └──────────────────────────────────────────────────────────┘

연속 값 → EMA (지수이동평균, alpha=0.3):
  ┌──────────────────────────────────────────────────────────┐
  │ arousal:                                                 │
  │   new = 0.3 × current + 0.7 × previous                  │
  │   이전 smoothed = 0.50                                   │
  │   현재 raw      = 0.45                                   │
  │   → smoothed_arousal = 0.3 × 0.45 + 0.7 × 0.50 = 0.485 │
  │                                                          │
  │ valence:                                                 │
  │   이전 smoothed = 0.60                                   │
  │   현재 raw      = 0.62                                   │
  │   → smoothed_valence = 0.3 × 0.62 + 0.7 × 0.60 = 0.606 │
  │                                                          │
  │ 효과: 급격한 변동 완화                                    │
  │   raw:      0.8 → 0.3 → 0.7 → 0.2 (진동)               │
  │   smoothed: 0.8 → 0.65→ 0.67→ 0.53 (부드러움)           │
  └──────────────────────────────────────────────────────────┘

Smoothing 후 result에 추가되는 키:
  smoothed_kfer_emotion       = "happy"
  smoothed_emotion_confidence = 0.857
  smoothed_arousal            = 0.485
  smoothed_valence            = 0.606
  smoothed_drowsy             = 0
  smoothed_compound_id        = 3
```

---

## Stage 8: 10-Class 판정 (_compute_ten_class)

```
Smoothed result를 기반으로 최종 10-class 판정:

═══ Emotion 6개 ═══

  smoothed_kfer_emotion = "happy"
  → kfer_label_to_id["happy"] = 2
  → KFER_TO_PROTOCOL[2] = 4
  → PROTOCOL_NAMES[4] = "행복"

  결과: emotion_code = 4, emotion_name_ko = "행복"

  ┌─────────────────────────────────────────────┐
  │ K-FER 7-class    →    Protocol 6-class      │
  │                                             │
  │ angry(0)      ───→  Code 2 : 분노           │
  │ anxious(1)    ───→  Code 0 : 공포           │
  │ happy(2)      ───→  Code 4 : 행복    ◀ 이것 │
  │ hurt(3)       ──┐                           │
  │                 ├→  Code 3 : 슬픔/혐오       │
  │ sad(5)        ──┘                           │
  │ neutral(4)    ───→  Code 5 : 중립           │
  │ surprised(6)  ───→  Code 1 : 놀람           │
  └─────────────────────────────────────────────┘

═══ Stress ═══

  조건: emotion ∈ {angry(0), anxious(1)} AND arousal > 0.6

  이 예시:
    kfer_id = 2 (happy) → {angry, anxious}에 없음
    → stress = False

  만약 angry + arousal=0.85였다면:
    kfer_id = 0 (angry) → {angry, anxious}에 있음
    arousal = 0.85 > 0.6
    → stress = True

═══ Low Attention ═══

  조건: 0.2 ≤ PERCLOS < 0.4

  이 예시:
    PERCLOS = 0.15 → 0.15 < 0.2
    → low_attention = False

  만약 PERCLOS=0.30이었다면:
    0.2 ≤ 0.30 < 0.4
    → low_attention = True

═══ Drowsy ═══

  조건: PERCLOS ≥ 0.4 OR smoothed_drowsy ≥ 1

  이 예시:
    PERCLOS = 0.15 < 0.4
    smoothed_drowsy = 0 < 1
    → drowsy = False

  만약 PERCLOS=0.50이었다면:
    0.50 ≥ 0.4
    → drowsy = True

═══ Fatigue (FatigueTracker) ═══

  3가지 OR 조건 (하나라도 충족 → True):

  조건1: compound_label ∈ {"depressed","calm"} 30초 연속
    이 예시: "positive_engaged" → 해당 안 됨, 카운터 리셋
    (만약 "depressed"가 300프레임(30초×10Hz) 연속이면 True)

  조건2: arousal < 0.3 이 30초 연속
    이 예시: 0.485 > 0.3 → 해당 안 됨, 카운터 리셋
    (만약 arousal=0.2가 300프레임 연속이면 True)

  조건3: drowsy 상태가 10초 연속
    이 예시: drowsy=0 → 해당 안 됨, 카운터 리셋
    (만약 drowsy=1이 100프레임(10초) 연속이면 True)

  → fatigue = False

10-Class 최종 결과:
  ┌─────────────────────────────────────┐
  │  #  │ 항목          │ 값           │
  │ ────┼───────────────┼──────────────│
  │  1  │ 공포          │ -            │
  │  2  │ 놀람          │ -            │
  │  3  │ 분노          │ -            │
  │  4  │ 슬픔/혐오     │ -            │
  │  5  │ 행복          │ ◀ Code 4     │
  │  6  │ 중립          │ -            │
  │  7  │ Stress        │ False        │
  │  8  │ Low Attention │ False        │
  │  9  │ Drowsy        │ False        │
  │ 10  │ Fatigue       │ False        │
  └─────────────────────────────────────┘
```

---

## Stage 9: PacketEncoder → 8-Byte USB 패킷

```
입력:
  kfer_emotion_id      = 2      (happy)
  emotion_confidence   = 0.857  (smoothed confidence)
  arousal              = 0.485  (smoothed arousal)
  perclos              = 0.15
  fatigue              = False

Step 9-1: Emotion Code
  KFER_TO_PROTOCOL[2] = 4 (행복)

Step 9-2: Status Flags
  StressDetector:    kfer_id=2 ∉ {0,1}           → stress = 0
  AttentionDetector: 0.15 < 0.2                   → low_attn = 0
  DrowsyDetector:    0.15 < 0.4                   → drowsy = 0
  NegEmoDetector:    kfer_id=2 ∉ {0,1,3,5}        → neg_emo = 0

Step 9-3: Intensity Quantization
  emo_intensity  = quantize(0.857) = 6  (0~7, 3bit)
  state_intensity = quantize(0.485) = 3  (0~7, 3bit)

Step 9-4: Byte4 조립
  ┌──────────────────────────────────────┐
  │ Byte4 = EEEE SSLD                   │
  │                                      │
  │ E = emotion_code = 4 = 0100          │
  │ S = stress       = 0                 │
  │ L = low_attn     = 0                 │
  │ D = drowsy       = 0                 │
  │ - = end_flag     = 0                 │
  │                                      │
  │ Byte4 = 0100 0000 = 0x40             │
  └──────────────────────────────────────┘

Step 9-5: Byte5 조립
  ┌──────────────────────────────────────┐
  │ Byte5 = III JJJ NF                  │
  │                                      │
  │ I = emo_intensity   = 6 = 110        │
  │ J = state_intensity = 3 = 011        │
  │ N = neg_emo         = 0             │
  │ F = fatigue         = 0             │
  │                                      │
  │ Byte5 = 110 011 0 0 = 0xCC          │
  └──────────────────────────────────────┘

Step 9-6: CRC8 계산
  payload = [0x01, seq, 0x02, 0x40, 0xCC]
  CRC8 = compute_crc8(payload) = 0x??

Step 9-7: 최종 8-byte 패킷
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │ 0xAA │ 0x01 │ 0x00 │ 0x02 │ 0x40 │ 0xCC │ CRC  │ 0xFE │
  │ SOF  │ TYPE │ SEQ  │ LEN  │Byte4 │Byte5 │      │ EOF  │
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

만약 fatigue=True였다면:
  Byte5 bit0에 1 추가:
  0xCC = 1100 1100 → 0xCD = 1100 1101
  CRC 재계산
```

---

## Stage 10: Gateway 전송

```
GatewaySender.send(packet_8bytes)
       │
       ▼
  pyserial: /dev/ttyUSB0 @ 115200 baud
       │
       ▼
  USB Serial → 차량 ECU

  전송 실패 시:
    - 에러 로그만 남김 (크래시 안 함)
    - 5초마다 재연결 시도
    - NullGatewaySender: 아무것도 안 함 (테스트용)
```

---

## 전체 시간 흐름 (1 사이클 = 0.1초)

```
t=0.000s  센서 데이터 수집 (30fps 카메라, 48kHz 마이크, BLE 워치)
          ↓ 각 콜백이 공유 버퍼에 push

t=0.100s  inference_loop 사이클 시작 (10Hz)
          │
          ├─ ModelInputs.snapshot()                    ~0ms
          │
          ├─ _process_face()                           ~15ms
          │   ├─ MediaPipe FaceMesh                    ~8ms
          │   ├─ KFERExpert (ResNet forward)            ~5ms
          │   └─ FACSAuxExpert                          ~2ms
          │
          ├─ _process_audio()                          ~25ms
          │   ├─ librosa resample (48k→16k)             ~3ms
          │   ├─ Emotion2VecExpert (wav2vec2)           ~15ms
          │   └─ AudeeringExpert (wav2vec2)             ~7ms
          │
          ├─ _process_bio()                            ~5ms
          │   └─ extract_bio_features_v2 (NeuroKit2)    ~5ms
          │
          ├─ _build_feature_dict() + KMERFusion         ~3ms
          │   ├─ Token formation (14 Linear)            ~1ms
          │   ├─ Pool-FFN + MHSA                        ~1ms
          │   └─ Output heads                           ~1ms
          │
          ├─ CompoundEmotionMapper                      ~0ms
          │
          ├─ TemporalSmoother                           ~0ms
          │
          ├─ 10-Class 판정                               ~0ms
          │
          ├─ FatigueTracker.update()                    ~0ms
          │
          ├─ PacketEncoder.encode()                     ~0ms
          │
          └─ GatewaySender.send()                       ~1ms
                                                   총 ~50ms
          남은 50ms: sleep으로 대기

t=0.200s  다음 사이클 시작
```
