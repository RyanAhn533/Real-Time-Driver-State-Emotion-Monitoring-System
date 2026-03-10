# K-MER Sensing System 전체 아키텍처 및 코드 흐름
> 각 코드의 기능, 데이터 흐름, 라벨 산출 방식 정리

---

## 1. 전체 데이터 흐름 (한눈에 보기)

```
 ┌─────────────────────────── 센서 계층 ───────────────────────────┐
 │                                                                 │
 │  RealSense D435        RODE Wireless GO II      ADI Study Watch │
 │  (1280×720@30fps)      (48kHz mono)             (PPG/EDA/Temp)  │
 │       │                      │                     │  │  │      │
 │  run_realsense()        run_rode()            run_watch()       │
 │       │                      │                  │    │    │      │
 │  on_frame(ts,bgr)    on_audio(ts,mono,sr)  on_ppg on_eda on_temp│
 │       │                      │                  │    │    │      │
 │  Latest.set()         Ring1D.push()       BioQueues.push_*()    │
 │       └──────────┐          │          ┌────────┘               │
 │                  ▼          ▼          ▼                         │
 │               ModelInputs (공유 버퍼)                             │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                        10Hz snapshot
                               │
 ┌─────────────────────────────▼───────────────────────────────────┐
 │                      추론 계층 (KMERInferencer)                   │
 │                                                                 │
 │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐        │
 │  │ _process_face│  │_process_audio│  │  _process_bio  │        │
 │  │              │  │              │  │                │        │
 │  │ FaceMesh     │  │ resample     │  │ NeuroKit2      │        │
 │  │ → KFERExpert │  │ → Emotion2Vec│  │ → BVP(4)       │        │
 │  │ → FACSAux    │  │ → Audeering  │  │ → EDA(5)       │        │
 │  │              │  │              │  │ → HR+Temp(6)   │        │
 │  │ 출력:        │  │ 출력:        │  │ → Quality(3)   │        │
 │  │ kfer_probs(7)│  │ emo2vec(9)   │  │                │        │
 │  │ perclos(2)   │  │ avd(3)       │  │ 출력:          │        │
 │  │ facs(6)      │  │ quality(3)   │  │ bio_feats(18)  │        │
 │  └──────┬───────┘  └──────┬───────┘  └───────┬────────┘        │
 │         └─────────────────┼───────────────────┘                 │
 │                           ▼                                     │
 │              _build_feature_dict()                               │
 │              → 14 tokens (각 64d) + CLS token                   │
 │              → valid_mask (어떤 센서가 살아있는지)                  │
 │                           │                                     │
 │                           ▼                                     │
 │                ┌─────────────────────┐                          │
 │                │    KMERFusion       │                          │
 │                │  (15 tokens × 64d)  │                          │
 │                │   ~144K params      │                          │
 │                │                     │                          │
 │                │ Pool-FFN → MHSA     │                          │
 │                │ → CLS pooling       │                          │
 │                └─────────┬───────────┘                          │
 │                   ┌──────┼──────┐                               │
 │                   ▼      ▼      ▼                               │
 │               arousal valence drowsy                             │
 │               (0~1)   (0~1)  (0/1/2)                            │
 │                   │      │      │                               │
 │                   ▼      ▼      ▼                               │
 │            CompoundEmotionMapper                                │
 │            → compound_id (0~12)                                 │
 │            → compound_label                                     │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
 ┌─────────────────────────────▼───────────────────────────────────┐
 │                   후처리 계층 (E2E Pipeline)                      │
 │                                                                 │
 │  TemporalSmoother ──→ 10-Class 판정 ──→ PacketEncoder           │
 │   - EMA(arousal)       - Emotion 6개      - 8-byte 패킷         │
 │   - MajVote(emotion)   - Stress            - CRC8               │
 │   - MajVote(drowsy)    - Low Attention     - Fatigue bit 추가    │
 │                        - Drowsy                                 │
 │  FatigueTracker ───→   - Fatigue                                │
 │   - 30초 지속 판정                                                │
 │                                                                 │
 │  GatewaySender ──→ /dev/ttyUSB0 ──→ 차량 ECU                    │
 └─────────────────────────────────────────────────────────────────┘
```

---

## 2. 파일별 기능 정리

### 센서 드라이버 (원본 코드, 수정 없이 import)

| 파일 | 기능 | 입력 | 출력 |
|------|------|------|------|
| `realsense.py` | RealSense 카메라 BGR 프레임 캡처 | 카메라 시리얼, fps | `on_frame(ts, frame_bgr)` 콜백 |
| `rode.py` | RODE 마이크 오디오 캡처 | sounddevice 자동 검색 | `on_audio_chunk(ts, mono, sr)` 콜백 |
| `watch.py` | ADI Study Watch BLE 생체신호 | VID/PID/MAC | `on_ppg/on_eda/on_temp` 콜백 |

### 핵심 추론 (원본 코드, 수정 없이 import)

| 파일 | 기능 | 입력 | 출력 |
|------|------|------|------|
| `sensing_main.py` | 버퍼 클래스 + 스레드 관리 | - | Latest, Ring1D, BioQueues, ModelInputs |
| `kmer_inferencer.py` | 멀티모달 추론 엔진 | frame, audio, ppg, eda, temp | arousal, valence, drowsy, compound, kfer_emotion |
| `kmer_fusion.py` | 퓨전 신경망 (144K params) | 14 feature tokens | arousal, valence, drowsy |
| `compound_emotion.py` | 규칙 기반 복합감정 매핑 | kfer_top1 × arousal | 13-class compound label |
| `packet_encoder.py` | 8-byte USB 패킷 생성 | emotion_id, arousal, perclos | 8-byte binary packet |

### 새로 구축한 파이프라인

| 파일 | 기능 | 입력 | 출력 |
|------|------|------|------|
| `config/__init__.py` | YAML → dataclass 설정 로더 | `sensing_config.yaml` | SensingConfig 객체 |
| `core/logger.py` | 구조적 로깅 (콘솔+파일) | 로거 이름 | named logger |
| `core/buffers.py` | 원본 버퍼 클래스 re-export | - | Latest, Ring1D 등 |
| `pipeline/temporal_smoother.py` | 출력 안정화 | raw 추론 결과 | smoothed 결과 |
| `pipeline/fatigue_tracker.py` | 피로 누적 감지 | compound, arousal, drowsy | fatigue True/False |
| `pipeline/gateway_sender.py` | USB 시리얼 전송 | 8-byte packet | 차량 ECU로 전송 |
| `pipeline/e2e_pipeline.py` | **전체 오케스트레이터** | 센서 데이터 | 10-class + 패킷 |
| `sensing_main_shadow.py` | Shadow 모드 진입점 | - | 기존+새 파이프라인 병렬 |

---

## 3. 추론 단계별 상세

### Step 1: 얼굴 처리 (`_process_face`)

```
BGR 프레임 (1280×720)
    │
    ▼
MediaPipe FaceMesh (468 랜드마크)
    │
    ├─ 얼굴 bbox 추출 (20% padding)
    ├─ face_chw (3, 224, 224) RGB crop
    └─ au_coords (8, 2) : 8개 AU 영역 좌표
       [이마, 왼눈, 오른눈, 코, 왼볼, 오른볼, 입, 턱]
    │
    ├──→ KFERExpert.extract(face_chw, au_coords)
    │    → kfer_probs (7,) : 7가지 감정 확률
    │    → kfer_meta  (2,) : quality, entropy
    │    → face_stats (3,) : max/mean/std confidence
    │
    └──→ FACSAuxExpert.extract(face_chw, au_coords)
         → perclos_ear (2,) : PERCLOS, mean EAR
         → facs_scores (6,) : 6개 FACS AU 강도
```

**K-FER 7-class 라벨:**
| Index | 라벨 | 설명 |
|-------|------|------|
| 0 | angry | 분노 |
| 1 | anxious | 불안/공포 |
| 2 | happy | 행복 |
| 3 | hurt | 상처/혐오 |
| 4 | neutral | 중립 |
| 5 | sad | 슬픔 |
| 6 | surprised | 놀람 |

### Step 2: 오디오 처리 (`_process_audio`)

```
오디오 버퍼 (48kHz, 2초분)
    │
    ▼
librosa resample (48kHz → 16kHz)
    │
    ├─ audio_quality (3,) : RMS, Zero Crossing Rate, SNR
    │
    ├──→ Emotion2VecExpert.extract(audio_16k)
    │    → emo2vec_probs (9,) : 9-class 오디오 감정 확률
    │      [angry, disgusted, fearful, happy, neutral,
    │       other, sad, surprised, unknown]
    │
    └──→ AudeeringExpert.extract(audio_16k)
         → audeering_avd (3,) : arousal, valence, dominance
```

### Step 3: 생체신호 처리 (`_process_bio`)

```
PPG [(ts, d1, d2), ...]    EDA [(ts, real), ...]    Temp [(ts, skin_c), ...]
    │                           │                         │
    └───────────────────────────┼─────────────────────────┘
                                │
                                ▼
                    extract_bio_features_v2()
                    (NeuroKit2 + scipy 기반)
                                │
    ┌───────────────────────────┼───────────────────────────────┐
    │                           │                               │
    ▼                           ▼                               ▼
bvp_features (4)         eda_features (5)           hr_temp_features (6)
 - mean_hr                - mean_scl                 - hr_mean
 - sdnn                   - std_scl                  - hr_std
 - rmssd                  - n_peaks                  - hr_range
 - lf_hf                  - amplitude                - temp_mean
                          - auc                      - temp_slope
                                                     - temp_range

    + bio_quality (3) : 품질 지표
```

### Step 4: 토큰 구성 → KMERFusion

```
14개 Feature 토큰 (각각 → Linear → 64d):

 Token │ 이름           │ 원본차원│ 출처       │ 모달리티
 ──────┼────────────────┼────────┼───────────┼──────────
  T1   │ kfer_probs     │   7    │ KFERExpert│ Face
  T2   │ kfer_meta      │   2    │ KFERExpert│ Face
  T3   │ face_stats     │   3    │ KFERExpert│ Face
  T4   │ emo2vec_probs  │   9    │ Emotion2Vec│ Audio
  T5   │ audeering_avd  │   3    │ Audeering │ Audio
  T6   │ audio_quality  │   3    │ 직접계산   │ Audio
  T7   │ bvp_features   │   4    │ BioExpert │ Bio
  T8   │ eda_features   │   5    │ BioExpert │ Bio
  T9   │ hr_temp_feats  │   6    │ BioExpert │ Bio
  T10  │ bio_quality    │   3    │ BioExpert │ Bio
  T11  │ perclos_ear    │   2    │ FACSAux   │ Aux
  T12  │ facs_scores    │   6    │ FACSAux   │ Aux
  T13  │ cross_modal    │   3    │ 교차계산   │ Meta
  T14  │ validity_flags │   3    │ 센서상태   │ Meta
  T15  │ CLS            │  64    │ learnable │ -

    + Modality Type Embedding (5종: face/audio/bio/aux/meta)
    + valid_mask (1, 15) : 센서 누락 시 해당 토큰 masking
                │
                ▼
         ┌──────────────────────────────────┐
         │         KMERFusion 신경망          │
         │                                  │
         │  1. Intra-Modal Pool-FFN         │
         │     Face(T1-T3), Audio(T4-T6),   │
         │     Bio(T7-T10) 각 그룹 내 혼합    │
         │                                  │
         │  2. Global MHSA (4 heads)        │
         │     valid_mask로 누락 토큰 무시     │
         │     모든 모달리티 간 상호작용        │
         │                                  │
         │  3. CLS Token Pooling            │
         │     T15에 전체 정보 응축 (64d)      │
         │                                  │
         │  4. Output Heads                 │
         │     arousal: Linear→Sigmoid (0~1)│
         │     valence: Linear→Sigmoid (0~1)│
         │     drowsy:  Linear→3-class      │
         │       (perclos도 입력에 추가)      │
         └──────────────────────────────────┘
```

### Step 5: CompoundEmotionMapper (13-class 복합감정)

```
입력: kfer_top1_id, arousal, is_drowsy

if is_drowsy → (12, "drowsy") 즉시 반환

arousal 이산화:
  low  = arousal < 0.33
  mid  = 0.33 ≤ arousal < 0.66
  high = arousal ≥ 0.66

매핑 테이블:
 K-FER 감정  │  low arousal   │  mid arousal     │  high arousal
 ────────────┼────────────────┼──────────────────┼───────────────
 neutral(4)  │ (1) calm       │ (0) neutral      │ (0) neutral
 happy(2)    │ (2) happy      │ (3) pos_engaged  │ (4) excited
 sad(5)      │ (6) depressed  │ (5) sad          │ (5) sad
 anxious(1)  │ (7) anxious    │ (7) anxious      │ (8) stressed
 angry(0)    │ (9) angry      │ (9) angry        │ (9) angry
 hurt(3)     │ (10) hurt      │ (10) hurt        │ (10) hurt
 surprised(6)│ (11) surprised │ (11) surprised   │ (11) surprised
```

---

## 4. 10-Class 판정 로직 상세

### Emotion 6개 (K-FER → Protocol 매핑)

```
K-FER 7-class                Protocol 6-class (sad+hurt 병합)
─────────────                ────────────────────────────────
angry(0)      ──→  Code 2 : 분노
anxious(1)    ──→  Code 0 : 공포
happy(2)      ──→  Code 4 : 행복
hurt(3)       ──┐
               ├→  Code 3 : 슬픔/혐오  (병합)
sad(5)        ──┘
neutral(4)    ──→  Code 5 : 중립
surprised(6)  ──→  Code 1 : 놀람
```

### Driver State 4개

```
┌──────────────────────────────────────────────────────────────┐
│ #7 Stress                                                    │
│                                                              │
│  조건: emotion ∈ {angry, anxious} AND arousal > 0.6          │
│  입력: smoothed_kfer_emotion + smoothed_arousal              │
│  센서 없을 때: emotion만으로 판정 (arousal=None → True)        │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ #8 Low Attention                                             │
│                                                              │
│  조건: 0.2 ≤ PERCLOS < 0.4                                   │
│  입력: FACSAuxExpert → perclos_ear[0]                        │
│  센서 없을 때: False                                          │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ #9 Drowsy                                                    │
│                                                              │
│  조건: PERCLOS ≥ 0.4 OR drowsy_head ≥ 1                     │
│  입력: FACSAux PERCLOS + KMERFusion drowsy head              │
│  센서 없을 때: False                                          │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ #10 Fatigue (FatigueTracker)                                 │
│                                                              │
│  3가지 OR 조건 (하나라도 충족 시 True):                        │
│                                                              │
│  조건1: compound_label ∈ {depressed, calm} 이 30초 연속      │
│         → 저각성 감정이 오래 지속 = 만성 피로                   │
│                                                              │
│  조건2: arousal < 0.3 이 30초 연속                            │
│         → 생체적 저각성 상태 지속                               │
│                                                              │
│  조건3: drowsy 상태가 10초 연속                                │
│         → 졸음이 장시간 지속 = 피로                             │
│                                                              │
│  Bio 센서 없을 때: 조건2 불가 → 조건1,3만으로 판정              │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Temporal Smoothing (출력 안정화)

```
매 프레임 (10Hz) KMERInferencer 출력이 흔들릴 수 있음

  raw output:    happy → happy → angry → happy → happy → happy
  smoothed:      happy → happy → happy → happy → happy → happy
                                  ↑ 1프레임 노이즈 제거

방식:
  이산 값 (emotion, drowsy, compound) → Majority Vote (슬라이딩 윈도우)
    - emotion: window=7 (최근 7프레임 중 다수결)
    - drowsy:  window=5
    - compound: window=7

  연속 값 (arousal, valence) → EMA (지수이동평균)
    - new_value = 0.3 × raw + 0.7 × previous
    - 급격한 변동 완화
```

---

## 6. 8-Byte USB 패킷 구조

```
 Byte0   Byte1   Byte2   Byte3   Byte4        Byte5        Byte6   Byte7
┌──────┬──────┬──────┬──────┬────────────┬────────────┬──────┬──────┐
│ 0xAA │ 0x01 │ SEQ  │ 0x02 │ EEEE SSLD  │ III JJJ NF │ CRC  │ 0xFE │
│ SOF  │ TYPE │ 순번  │ LEN  │            │            │      │ EOF  │
└──────┴──────┴──────┴──────┴────────────┴────────────┴──────┴──────┘

Byte4 비트맵:
  [7:4] EEEE = Emotion Code (0~5)
  [3]   S    = Stress flag
  [2]   L    = Low Attention flag
  [1]   D    = Drowsy flag
  [0]   -    = End flag (reserved)

Byte5 비트맵:
  [7:5] III  = Emotion Intensity (0~7, confidence 3bit 양자화)
  [4:2] JJJ  = State Intensity (0~7, arousal 3bit 양자화)
  [1]   N    = Negative Emotion flag
  [0]   F    = Fatigue flag (신규 추가)

예시: happy(code=4), conf=0.9, arousal=0.5, no flags
  Byte4 = 0100 0000 = 0x40
  Byte5 = 110 100 0 0 = 0xC8
  → AA 01 00 02 40 C8 [CRC] FE
```

---

## 7. Degraded Mode (센서 장애 대응)

```
 모드       │ 가용 센서      │ 동작하는 Expert                │ 가능한 출력
 ──────────┼───────────────┼──────────────────────────────┼────────────────
 FULL      │ Cam+Mic+Bio   │ K-FER, FACS, Emo2Vec,        │ 10-class 전체
           │               │ Audeering, BioExpert         │
 ──────────┼───────────────┼──────────────────────────────┼────────────────
 CAM+MIC   │ Cam+Mic       │ K-FER, FACS, Emo2Vec,        │ Emotion 6개
           │               │ Audeering                    │ + Stress(emo only)
           │               │ Bio tokens masked            │ + LowAttn + Drowsy
 ──────────┼───────────────┼──────────────────────────────┼────────────────
 CAM+BIO   │ Cam+Bio       │ K-FER, FACS, BioExpert       │ Emotion 6개
           │               │ Audio tokens masked          │ + Stress + LowAttn
           │               │                              │ + Drowsy + Fatigue
 ──────────┼───────────────┼──────────────────────────────┼────────────────
 CAM_ONLY  │ Cam           │ K-FER, FACS                  │ Emotion 6개
           │               │ Audio+Bio tokens masked      │ + LowAttn + Drowsy
 ──────────┼───────────────┼──────────────────────────────┼────────────────
 NO_CAM    │ 없음           │ 추론 불가                     │ neutral + 모든
           │               │                              │ state = False

 KMERFusion의 valid_mask가 이 구조를 지원:
   - 누락된 센서의 토큰을 attention에서 제외
   - 나머지 토큰만으로 추론 수행
```

---

## 8. Shadow Mode 동작 구조

```
sensing_main_shadow.py 실행 시:

  센서 3개 → ModelInputs (공유 버퍼)
                  │
     ┌────────────┴────────────┐
     │                         │
     ▼                         ▼
 inference_loop()         shadow_inference_loop()
 (원본, 기존 코드)         (새 E2E Pipeline)
     │                         │
     ▼                         ▼
 print() 출력              구조적 로그 출력
 (기존 동작 그대로)         (10-class + 패킷 + 모드)

 → 두 출력을 비교하여 새 파이프라인 검증
 → 검증 완료 후 shadow_mode.enabled=false → E2E만 사용
```

---

## 9. 실행 방법

```bash
# 1. 로직 검증 (하드웨어 불필요)
cd /home/ajy/Jetson_thor/sensing
python test_e2e_verification.py          # 42개 테스트 통과 확인

# 2. 센서 개별 테스트
python test_sensors.py --test camera     # 카메라만
python test_sensors.py --test mic        # 마이크만
python test_sensors.py --test watch      # 워치만

# 3. Shadow 모드 (기존+새 파이프라인 병렬)
python sensing_main_shadow.py

# 4. Shadow only (새 파이프라인만)
python sensing_main_shadow.py --shadow_only

# 5. Gateway 비활성화 (시리얼 포트 없을 때)
python sensing_main_shadow.py --gateway_off
```

---

## 10. Config로 제어되는 모든 값

```yaml
# config/sensing_config.yaml 하나로 모든 하드코딩 제거

hardware:
  realsense_main:
    serial: "021222070391"       # 카메라 시리얼 → realsense.py
  rode:
    sample_rate: 48000           # 마이크 SR → rode.py
  watch:
    mac: "F1-18-1C-93-7C-42"    # 워치 MAC → watch.py

inference:
  hz: 10                        # 추론 루프 주기
  device: "cuda"                # GPU 사용

thresholds:
  perclos_drowsy: 0.4           # Drowsy 판정 기준
  arousal_stress: 0.6           # Stress 판정 기준
  fatigue_duration_sec: 30      # Fatigue 지속시간

gateway:
  port: "/dev/ttyUSB0"          # USB 시리얼 포트
  enabled: true                 # 전송 on/off

temporal_smoothing:
  emotion_window: 7             # 감정 안정화 윈도우
  ema_alpha: 0.3                # arousal EMA 계수
```
