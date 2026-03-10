# K-MER 센싱 시스템 완전 가이드

> **이 문서의 목적**: 이 프로젝트를 **처음 보는 사람**이 "뭐가 뭔지, 어떻게 돌아가는지" 전부 이해할 수 있도록 작성한 문서입니다.

---

## 1. 이 시스템이 뭐야?

**한 줄 요약**: 차량 운전자의 **얼굴(카메라) + 목소리(마이크) + 생체신호(손목시계)**를 실시간으로 읽어서, **지금 운전자가 어떤 감정/상태인지 10가지로 판단**하고, 그 결과를 **차량 ECU에 USB로 전송**하는 시스템.

### 판단하는 10가지 항목

| # | 항목 | 한국어 | 어떻게 판단하나 |
|---|------|--------|----------------|
| 1 | Fear (공포) | 공포 | 얼굴 표정 분석 |
| 2 | Surprise (놀람) | 놀람 | 얼굴 표정 분석 |
| 3 | Anger (분노) | 분노 | 얼굴 표정 분석 |
| 4 | Sadness/Disgust (슬픔/혐오) | 슬픔/혐오 | 얼굴 표정 분석 |
| 5 | Happy (행복) | 행복 | 얼굴 표정 분석 |
| 6 | Neutral (중립) | 중립 | 얼굴 표정 분석 |
| 7 | **Stress** (스트레스) | 스트레스 | 부정감정 + 높은 각성도(arousal) |
| 8 | **Low Attention** (주의력 저하) | 주의력 저하 | 눈 감긴 비율(PERCLOS) 20~40% |
| 9 | **Drowsy** (졸음) | 졸음 | 눈 감긴 비율(PERCLOS) 40% 이상 |
| 10 | **Fatigue** (피로) | 피로 | 우울/침착 상태 30초 지속 등 |

### 사용하는 센서 3개

| 센서 | 장비명 | 뭘 측정하나 |
|------|-------|-----------|
| **카메라** | Intel RealSense D435 | 운전자 얼굴 영상 (1280×720, 30fps) |
| **마이크** | RODE Wireless GO II | 운전자 목소리 (48kHz) |
| **손목시계** | ADI Study Watch | PPG(맥박), EDA(피부전도), 체온 |

---

## 2. 프로젝트 폴더 구조 (뭐가 어디에 있나)

```
Jetson_thor/                          ← 프로젝트 루트
│
├── sensing/                          ← ★ 실시간 센서 수집 + 추론 (이 시스템의 메인)
│   ├── config/                       ← 설정 파일
│   ├── core/                         ← 공통 유틸 (로그, 버퍼)
│   ├── pipeline/                     ← E2E 파이프라인 (핵심!)
│   ├── docs/                         ← 문서들
│   ├── raw_sensing_code/             ← 팀원(석희)의 원본 코드 (보존용)
│   ├── sensing_main_shadow.py        ← ★ 실행 파일 (이걸 실행하면 됨)
│   └── test_e2e_verification.py      ← 테스트
│
├── emotion_system/                   ← 감정 인식 AI 모델 (학습용)
│   ├── models/                       ← 딥러닝 모델 아키텍처
│   ├── training/                     ← 학습 코드
│   └── result/best.pth              ← ★ K-FER 학습된 체크포인트
│
└── multimodal_dms/                   ← 멀티모달 퓨전 + 게이트웨이
    ├── experts/                      ← 각 센서별 전문가 모델
    ├── fusion/                       ← KMERFusion (퓨전 모델)
    ├── gateway/                      ← 8-byte 패킷 인코더
    └── results_kmer/best_model.pth   ← ★ KMERFusion 학습된 체크포인트
```

---

## 3. 전체 동작 흐름 (큰 그림)

```
실행: python sensing_main_shadow.py
        │
        ▼
┌──────────────────────────────────────────────────────┐
│                     sensing_main_shadow.py           │
│                                                      │
│  1. config/sensing_config.yaml 설정 로드             │
│  2. 로그 시스템 초기화                                │
│  3. 공유 버퍼(ModelInputs) 생성                      │
│  4. 센서 스레드 4개 시작:                             │
│     - RS_MAIN  : 카메라 → 프레임 버퍼에 저장          │
│     - RODE     : 마이크 → 오디오 버퍼에 저장          │
│     - WATCH    : 시계  → 생체 버퍼에 저장             │
│  5. 추론 스레드 2개 시작:                             │
│     - INFER_ORIG  : 기존 방식 (비교용)               │
│     - INFER_SHADOW: ★ 새 E2E 파이프라인              │
│  6. Ctrl+C 누를 때까지 계속 실행                      │
└──────────────────────────────────────────────────────┘
        │
        │  0.1초(10Hz)마다 반복 실행
        ▼
┌──────────────────────────────────────────────────────┐
│              E2EPipeline.process_cycle()              │
│                                                      │
│  Step 1: 센서 상태 확인 (FULL/CAM+MIC/CAM_ONLY/...)  │
│  Step 2: KMERInferencer.forward() 호출               │
│          → 얼굴/음성/생체 분석 → 감정/각성도/졸음 예측│
│  Step 3: Temporal Smoothing (떨림 방지)              │
│  Step 4: 10가지 항목 판정                            │
│  Step 5: Fatigue(피로) 판정                          │
│  Step 6: 8-byte 패킷 생성                           │
│  Step 7: USB로 차량에 전송                           │
└──────────────────────────────────────────────────────┘
        │
        ▼
   [차량 ECU가 패킷 수신 → 운전자 상태에 따라 경고/조치]
```

---

## 4. 모든 코드 파일 설명 (파일별)

### 4.1 실행 파일

#### `sensing/sensing_main_shadow.py` — ★ 이걸 실행하면 됨

```
역할: 전체 시스템의 진입점 (main 함수)
위치: sensing/sensing_main_shadow.py
실행: python sensing_main_shadow.py
```

**하는 일:**
1. `config/sensing_config.yaml`에서 설정을 읽음
2. 로그 시스템 초기화
3. 센서 데이터를 담을 공유 버퍼(`ModelInputs`) 생성
4. 센서 스레드 4개를 띄움 (카메라, 마이크, 시계, 서브카메라)
5. 기존 추론 루프(`inference_loop`) 실행 (비교용, `--shadow_only`로 끌 수 있음)
6. 새 E2E 추론 루프(`shadow_inference_loop`) 실행
7. `Ctrl+C`로 종료 시 모든 스레드 정리

**핵심 코드 흐름:**
```python
# 공유 버퍼 생성 (모든 센서가 여기에 데이터를 넣음)
mi = ModelInputs(audio_sr=48000, audio_sec=2.0)

# 카메라가 프레임을 찍으면 → 버퍼에 저장
def on_frame_main(ts_ms, frame_bgr):
    mi.frame_main.set(ts_ms, frame_bgr)

# 마이크가 소리를 잡으면 → 버퍼에 저장
def on_audio_chunk(ts_ms, mono, sr):
    mi.audio.push(ts_ms, mono)

# E2E 파이프라인이 0.1초마다 버퍼에서 데이터를 꺼내서 분석
pipeline = E2EPipeline(cfg)
shadow_inference_loop(shutdown, mi, pipeline, hz=10)
```

**옵션:**
| 옵션 | 설명 |
|------|------|
| `--shadow_only` | 새 파이프라인만 실행 (기존 것 안 돌림) |
| `--no_audio_experts` | 오디오 AI 모델 비활성화 (메모리 절약) |
| `--gateway_off` | 차량 USB 전송 끔 |
| `--device cpu` | CPU 모드 (GPU 없을 때) |
| `--hz 5` | 추론 속도 변경 (기본 10Hz) |

---

### 4.2 설정 파일

#### `sensing/config/sensing_config.yaml` — 모든 설정 한 곳에

```
역할: 하드웨어 시리얼번호, AI 모델 경로, 임계값 등 모든 설정
위치: sensing/config/sensing_config.yaml
```

**주요 섹션:**
```yaml
hardware:          # 카메라 시리얼, 마이크 키워드, 시계 MAC주소 등
inference:         # 추론 속도(10Hz), GPU/CPU, 체크포인트 경로
thresholds:        # 졸음 판정 기준(0.4), 스트레스 기준(0.6) 등
gateway:           # USB 포트(/dev/ttyUSB0), 통신속도(115200)
temporal_smoothing: # 떨림 방지 윈도우 크기
shadow_mode:       # Shadow mode ON/OFF
logging:           # 로그 레벨, 로그 파일 크기
```

**왜 필요한가:**
- 원래 코드는 시리얼 번호, MAC 주소 등이 코드에 하드코딩되어 있었음
- 하드웨어 교체 시 이 YAML 파일만 수정하면 됨

#### `sensing/config/__init__.py` — 설정 파일 로더

```
역할: YAML 파일을 읽어서 Python dataclass로 변환
사용법: from config import load_config; cfg = load_config()
```

---

### 4.3 핵심 파이프라인 코드 (`sensing/pipeline/`)

#### `sensing/pipeline/e2e_pipeline.py` — ★ 핵심 중의 핵심

```
역할: 센서 데이터 → AI 분석 → 10가지 판정 → 패킷 생성 → 전송
      전체 프로세스를 하나로 묶는 오케스트레이터
위치: sensing/pipeline/e2e_pipeline.py
```

**클래스: `E2EPipeline`**

| 메서드 | 하는 일 |
|--------|---------|
| `__init__(config)` | Smoother, FatigueTracker, GatewaySender 초기화 |
| `_lazy_init()` | 첫 호출 시 KMERInferencer + PacketEncoder 로드 (무거우니까 지연 로드) |
| `process_cycle(frame, audio, ppg, eda, temp)` | ★ 0.1초마다 호출되는 메인 함수 |
| `_compute_ten_class(smoothed)` | K-FER 결과 → 10가지 항목으로 변환 |
| `_encode_packet(ten_class, smoothed)` | 판정 결과 → 8바이트 패킷 |
| `_add_fatigue_bit(packet)` | 패킷에 피로 비트 추가 |
| `_make_fallback_result()` | 카메라 없을 때 기본값 반환 |

**`process_cycle()` 내부 동작:**
```
입력: frame(카메라), audio(마이크), ppg/eda/temp(시계)
  │
  ├─ Step 1: detect_mode() → 어떤 센서가 살아있나? (FULL/CAM_ONLY/NO_CAM 등)
  │
  ├─ Step 2: KMERInferencer.forward(frame, audio, ppg, eda, temp)
  │           → arousal(각성도), valence(쾌불쾌), drowsy(졸음), emotion(감정)
  │
  ├─ Step 3: TemporalSmoother.smooth(result)
  │           → 프레임 간 떨림 방지 (7프레임 다수결, EMA 필터)
  │
  ├─ Step 4: _compute_ten_class(smoothed)
  │           → emotion_code(0~5), stress(T/F), low_attention(T/F), drowsy(T/F)
  │
  ├─ Step 5: FatigueTracker.update(compound, arousal, drowsy)
  │           → fatigue(T/F)
  │
  ├─ Step 6: PacketEncoder.encode() → 8바이트 패킷 생성
  │
  └─ Step 7: GatewaySender.send(packet) → USB로 차량에 전송

출력: {emotion_code, stress, low_attention, drowsy, fatigue, packet, ...}
```

**`DegradedMode` (센서 고장 대응):**
```
FULL     = 카메라 + 마이크 + 시계 → 10가지 전부 판정 가능
CAM+MIC  = 카메라 + 마이크         → 생체 빠짐, arousal 없이 감정만으로 판정
CAM+BIO  = 카메라 + 시계          → 오디오 빠짐, 오디오 토큰 마스킹
CAM_ONLY = 카메라만              → 감정 + 졸음만 판정
NO_CAM   = 카메라 없음            → 전체 판정 불가, 중립(neutral) 반환
```

#### `sensing/pipeline/temporal_smoother.py` — 떨림 방지

```
역할: AI 모델 출력이 프레임마다 흔들리는 걸 안정화
위치: sensing/pipeline/temporal_smoother.py
```

**왜 필요한가:**
- AI 모델은 같은 사람 얼굴이라도 프레임마다 "happy → neutral → happy" 왔다갔다할 수 있음
- 이걸 7프레임 동안의 다수결(majority vote)로 안정화

**클래스 3개:**
| 클래스 | 용도 | 방식 |
|--------|------|------|
| `MajorityVoteSmoother` | 감정/졸음 같은 **이산값** | 최근 7개 중 가장 많은 것 선택 |
| `EMASmoother` | arousal/valence 같은 **연속값** | 지수이동평균 (alpha=0.3) |
| `MultimodalTemporalSmoother` | 위 둘을 합쳐서 한번에 | emotion, drowsy, compound → 다수결<br>arousal, valence → EMA |

**예시:**
```
raw emotion: [happy, happy, neutral, happy, happy, neutral, happy]
smoothed:     happy (7개 중 5개가 happy → happy 선택)

raw arousal: [0.7, 0.3, 0.8, ...]
smoothed:     0.3*0.8 + 0.7*이전값 (급격한 변화 억제)
```

#### `sensing/pipeline/fatigue_tracker.py` — 피로 감지

```
역할: 운전자가 피로한 상태인지 판정
위치: sensing/pipeline/fatigue_tracker.py
```

**피로 = 다음 3개 조건 중 하나라도 충족:**
```
조건 1: compound_label이 "depressed" 또는 "calm"으로 30초 이상 연속
        → 오랫동안 기운이 없는 상태

조건 2: arousal(각성도)이 0.3 미만으로 30초 이상 연속
        → 오랫동안 각성이 낮은 상태

조건 3: drowsy(졸음)가 10초 이상 연속
        → 졸음이 지속되는 상태
```

**센서 고장 시:**
- 시계 없음 → 조건2 판정 불가 → 조건1, 3만으로 판정
- 카메라 없음 → 조건1, 3 판정 불가 → 조건2만으로 판정

#### `sensing/pipeline/gateway_sender.py` — 차량 USB 전송

```
역할: 8-byte 패킷을 USB 시리얼(/dev/ttyUSB0)로 차량에 전송
위치: sensing/pipeline/gateway_sender.py
```

**클래스 2개:**
| 클래스 | 용도 |
|--------|------|
| `GatewaySender` | 실제 USB 전송 (pyserial 사용) |
| `NullGatewaySender` | 게이트웨이 OFF 시 아무것도 안 함 (로그만) |

**특징:**
- 연결 실패해도 크래시 안 함 → 로그만 남기고 계속 동작
- 일정 간격(5초)으로 재연결 시도
- `--gateway_off` 옵션으로 끌 수 있음

---

### 4.4 공통 유틸 (`sensing/core/`)

#### `sensing/core/buffers.py` — 센서 데이터 버퍼

```
역할: 센서 스레드 → 추론 스레드 간 데이터를 안전하게 전달하는 버퍼
위치: sensing/core/buffers.py
```

**원본 코드(sensing_main.py)의 버퍼 클래스를 그대로 가져다 씀:**

| 클래스 | 용도 | 어디서 씀 |
|--------|------|----------|
| `Latest` | 최신 값 1개만 보관 (카메라 프레임) | 카메라 → 추론 루프 |
| `Ring1D` | 원형 버퍼 (오디오 샘플) | 마이크 → 추론 루프 |
| `BioQueues` | PPG/EDA/체온 큐 (deque) | 시계 → 추론 루프 |
| `ModelInputs` | 위 3개를 하나로 묶음 | 모든 센서 → 추론 루프 |

**왜 이런 구조인가:**
```
[카메라 스레드] 30fps로 프레임 생성 ──┐
[마이크 스레드] 48kHz로 오디오 수집 ──┤ → ModelInputs (공유 버퍼)
[시계 스레드] PPG/EDA/체온 수집 ──────┘         │
                                               │ 0.1초마다 읽기
                                               ▼
                                    [추론 루프 스레드]
```
- 센서는 제각각 다른 속도로 데이터를 생산
- 추론 루프는 0.1초(10Hz)마다 "지금 있는 최신 데이터"를 가져감
- `Latest`는 항상 최신 1개만 유지 (카메라는 30fps인데 추론은 10Hz → 나머지 20개는 버림)
- `Ring1D`는 원형 버퍼 (2초 분량의 오디오를 유지)
- `BioQueues`는 deque로 생체신호 큐 관리

#### `sensing/core/logger.py` — 로그 시스템

```
역할: print() 대신 구조화된 로그 사용
위치: sensing/core/logger.py
```

**사용법:**
```python
from core.logger import get_logger
logger = get_logger("kmer.pipeline.e2e")
logger.info("현재 모드: %s", mode)
logger.error("센서 장애: %s", e)
```

**기능:**
- 콘솔: INFO 레벨 이상 출력
- 파일: DEBUG 레벨까지 전부 저장 (10MB 단위 로테이션, 5개 보관)
- `"kmer.pipeline.e2e"` 같은 계층적 이름으로 어디서 나온 로그인지 구분

---

### 4.5 원본 센서 드라이버 코드 (`raw_sensing_code/.../sensing/`)

> 이 파일들은 팀원(석희)이 작성한 원본 코드. 수정하지 않고 그대로 import해서 사용.

#### `realsense.py` — 카메라 드라이버

```
역할: Intel RealSense D435 카메라에서 실시간 영상 획득
함수: run_realsense(shutdown_event, device_serial, on_frame, ...)
```

**동작:**
1. `device_serial`로 특정 카메라 지정 (우리 카메라: `"021222070391"`)
2. 1280×720 해상도, 30fps, BGR 컬러 스트림 설정
3. 무한 루프로 프레임 폴링
4. 프레임 올 때마다 `on_frame(timestamp, image)` 콜백 호출
5. `shutdown_event` 설정되면 종료

#### `rode.py` — 마이크 드라이버

```
역할: RODE Wireless GO II 무선 마이크에서 실시간 오디오 획득
함수: run_rode(shutdown_event, on_audio_chunk, ...)
```

**동작:**
1. `"Wireless GO II"`, `"RØDE"` 등 키워드로 오디오 장치 자동 검색
2. 48kHz, 8192 샘플 블록 단위로 수집
3. 스테레오 → 모노 변환 (양쪽 채널 평균)
4. `on_audio_chunk(timestamp, mono_float32, sr=48000)` 콜백 호출
5. 연결 끊기면 5번까지 재시도

#### `watch.py` — 생체신호 시계 드라이버

```
역할: ADI Study Watch에서 PPG(맥박)/EDA(피부전도)/체온 획득
함수: run_watch(shutdown_event, on_ppg, on_eda, on_temp)
```

**동작:**
1. USB BLE 동글 검색 (VID=0x0456, PID=0x2CFE)
2. BLE로 시계 연결 (MAC: `"F1-18-1C-93-7C-42"`)
3. 3가지 센서 스트림 시작:
   - ADPD → PPG(맥박) → `on_ppg(ts, d1, d2)` 콜백
   - EDA → 피부전도 → `on_eda(ts, real)` 콜백
   - TEMP → 체온 → `on_temp(ts, skin_c)` 콜백

**특이사항:**
- BLE 연결이 불안정해서 `_patched_ble_open()` 이라는 몽키패치가 있음
- 시리얼 번호 읽기 타임아웃 → VID/PID로 매칭하는 우회 로직

#### `sensing_main.py` (원본) — 원래 메인 파일

```
역할: 원래 시스템의 메인 (지금은 버퍼 클래스만 가져다 씀)
제공하는 것: Latest, Ring1D, BioQueues, ModelInputs 클래스 + inference_loop 함수
```

---

### 4.6 AI 추론 엔진

#### `raw_sensing_code/.../sensing/kmer_inferencer.py` — ★ AI 추론의 핵심

```
역할: 카메라+마이크+시계 데이터를 받아서 AI 모델로 감정/각성도/졸음 예측
클래스: KMERInferencer
```

**이 파일이 하는 모든 것:**

```
입력: frame(카메라 이미지), audio(오디오), ppg/eda/temp(생체)
  │
  ├─── _process_face(frame) ──────────────────────────────────┐
  │    1. MediaPipe FaceMesh로 얼굴 검출 (468개 랜드마크)        │
  │    2. 얼굴 영역 잘라내기 (bbox + 20% 패딩 → 224×224)        │
  │    3. AU 좌표 추출 (8개 얼굴 근육 영역)                      │
  │    4. K-FER 모델로 7가지 감정 확률 예측                      │
  │    5. FACS 전문가로 PERCLOS(눈감김비율), EAR 계산            │
  │                                                           │
  │    출력: kfer_probs(7개 감정확률), perclos, facs_scores     │
  │                                                           │
  ├─── _process_audio(audio) ─────────────────────────────────┤
  │    1. 48kHz → 16kHz 리샘플링                                │
  │    2. emotion2vec 모델로 9가지 감정 확률 예측                 │
  │    3. audeering 모델로 arousal/valence/dominance 예측       │
  │    4. 오디오 품질 계산 (RMS, 영점교차율, SNR)                 │
  │                                                           │
  │    출력: emo2vec_probs(9), audeering_avd(3), quality(3)    │
  │                                                           │
  ├─── _process_bio(ppg, eda, temp) ──────────────────────────┤
  │    1. PPG → HRV 특징 (심박수, SDNN, RMSSD, LF/HF)         │
  │    2. EDA → 피부전도 특징 (평균, 분산, 피크수, AUC)          │
  │    3. TEMP → 체온 특징 (평균, 기울기, 범위)                  │
  │                                                           │
  │    출력: bvp_features(4), eda_features(5), hr_temp(6)      │
  │                                                           │
  ├─── _build_feature_dict() ─────────────────────────────────┤
  │    위 3개 결과를 14개 토큰으로 정리                          │
  │    + 유효성 마스크(어떤 센서가 살아있나) 생성                 │
  │                                                           │
  └─── KMERFusion.forward(features, valid_mask) ──────────────┘
       14개 토큰을 64차원으로 투영 → 퓨전 → 예측

출력: {
  arousal: 0.65,           # 각성도 (0~1, 높으면 흥분)
  valence: 0.72,           # 쾌불쾌 (0~1, 높으면 기분 좋음)
  drowsy: 0,               # 졸음 (0=정상, 1=졸림, 2=수면)
  compound_label: "happy",  # 13가지 복합감정 중 하나
  kfer_emotion: "happy",   # K-FER 7가지 감정 중 하나
  face_detected: True,     # 얼굴 검출 성공 여부
}
```

#### `raw_sensing_code/.../sensing/fer_inferencer.py` — 단독 표정 인식

```
역할: 카메라만으로 감정 예측 (멀티모달 아님, 디버그/테스트용)
클래스: FERInferencer
특징: KMERFusion 없이 K-FER 모델만 사용
```

---

### 4.7 AI 모델 코드 (multimodal_dms/experts/)

> 이 파일들은 KMERInferencer 내부에서 import되어 사용됨

#### `multimodal_dms/experts/kfer_expert.py` — 얼굴 감정 전문가

```
역할: 얼굴 이미지 → 7가지 감정 확률
입력: 얼굴 크롭 이미지 (224×224)
출력: probs=[0.01, 0.02, 0.85, 0.01, 0.05, 0.03, 0.03]  (7개 감정)
모델: MobileViTv2 + AU RoI Cross-Attention
```

**7가지 감정 (K-FER):**
```
Index 0: angry    (분노)     →  Protocol Code 2
Index 1: anxious  (불안/공포) →  Protocol Code 0
Index 2: happy    (행복)     →  Protocol Code 4
Index 3: hurt     (상처)     →  Protocol Code 3  ← sad와 합쳐짐
Index 4: neutral  (중립)     →  Protocol Code 5
Index 5: sad      (슬픔)     →  Protocol Code 3  ← hurt와 합쳐짐
Index 6: surprised(놀람)     →  Protocol Code 1
```

#### `multimodal_dms/experts/audio_expert.py` — 음성 감정 전문가

```
역할: 음성 → 감정 + arousal/valence
두 개 모델 사용:
  1. emotion2vec: 음성 → 9가지 감정 확률 (1024차원 임베딩)
  2. audeering:   음성 → arousal/valence/dominance (3개 연속값)
입력: 16kHz 모노 오디오 (약 2초)
```

#### `multimodal_dms/experts/bio_expert.py` — 생체신호 전문가

```
역할: PPG/EDA/체온 → 15개 수치 특징
모델 없음 (수작업 특징 추출, 딥러닝 아님)
```

**추출하는 15개 특징:**
```
PPG(맥박) → 4개: 평균 심박수, SDNN, RMSSD, LF/HF 비율
EDA(피부) → 5개: 평균, 표준편차, 피크수, 피크 평균진폭, AUC
체온       → 3개: 평균, 기울기, 범위
HR(심박)  → 3개: 평균, 표준편차, 범위
```

#### `multimodal_dms/experts/facs_aux.py` — 얼굴 보조 전문가

```
역할: 얼굴에서 눈 감김(PERCLOS)과 표정 근육(FACS) 추출
모델: MediaPipe FaceMesh (468개 랜드마크)
```

**출력:**
```
PERCLOS: 최근 프레임 중 눈 감긴 비율 (0~1)
  - 0.0~0.2: 정상
  - 0.2~0.4: 주의력 저하 (Low Attention)
  - 0.4이상:  졸음 (Drowsy)

EAR (Eye Aspect Ratio): 눈 높이/너비 비율
  - 0.21 미만: 눈 감김으로 판정

FACS scores (6개): 입벌림, 좌/우 눈썹, 좌/우 입꼬리, 코주름
```

---

### 4.8 퓨전 모델 (multimodal_dms/fusion/)

#### `multimodal_dms/fusion/kmer_fusion.py` — ★ 멀티모달 퓨전

```
역할: 얼굴 + 음성 + 생체 특징을 하나로 합쳐서 최종 예측
클래스: KMERFusion
파라미터: ~144,000개 (매우 가벼움, Jetson에서 실시간 동작 가능)
```

**퓨전 과정 (6단계):**
```
Stage 1: Token Formation (토큰화)
  14개 expert 출력을 각각 64차원으로 투영 + 1개 CLS 토큰 추가
  → 15 tokens × 64d

Stage 2: Intra-Modal Pool-FFN (모달리티 내부 정리)
  같은 종류 토큰끼리 묶어서 지역 패턴 학습
  Face 토큰(T1~T6), Audio 토큰(T7~T9), Bio 토큰(T10~T12) 각각

Stage 3: Global Cross-Modal MHSA (모달리티 간 연결)
  15개 토큰이 서로 attention (4-head)
  → "표정은 화나는데 목소리는 차분하면?" 같은 cross-modal 관계 학습
  → 센서 고장 토큰은 valid_mask로 마스킹 (무시)

Stage 4: CLS Pooling
  CLS 토큰(0번)이 모든 정보를 모아서 64차원 벡터 1개로 요약

Stage 5: Temporal Context (Bi-GRU, 선택적)
  여러 프레임에 걸친 시간 맥락 반영 (현재 미사용)

Stage 6: Output Heads (출력)
  arousal: 64d → 1d → sigmoid → [0,1]
  valence: 64d → 1d → sigmoid → [0,1]
  drowsy:  64d → 3d (alert / drowsy / sleeping)
```

**`valid_mask` (센서 고장 대응):**
```python
# 15개 토큰 각각에 대해 True(사용)/False(무시) 설정
# 예: 마이크 고장 → 오디오 토큰 3개를 False로 설정
valid_mask = [T,T,T,T,T,T,  F,F,F,  T,T,T,  T,T,  T]
#            ─Face 6개──  ─Audio─  ─Bio 3─  ─기타─ CLS
```

#### `multimodal_dms/fusion/compound_emotion.py` — 복합 감정 매퍼

```
역할: 기본감정(7개) + 각성도 → 복합감정(13개)으로 확장
클래스: CompoundEmotionMapper
방식: 규칙 기반 (AI 모델 아님)
```

**매핑 규칙:**
```
happy + 낮은 각성 → "happy"
happy + 중간 각성 → "positive_engaged" (긍정적 몰입)
happy + 높은 각성 → "excited" (흥분)
neutral + 낮은 각성 → "calm" (침착)
anxious + 높은 각성 → "stressed" (스트레스)
sad + 낮은 각성 → "depressed" (우울)
drowsy 상태 → "drowsy" (졸음)
... 총 13가지
```

---

### 4.9 게이트웨이 (multimodal_dms/gateway/)

#### `multimodal_dms/gateway/packet_encoder.py` — 8바이트 패킷

```
역할: AI 판정 결과를 차량이 이해할 수 있는 8바이트 이진 데이터로 변환
클래스: PacketEncoder, StressDetector, AttentionDetector, DrowsyDetector
```

**8바이트 구조:**
```
Byte 0: 0xAA (시작 마커 - "패킷 시작이야")
Byte 1: 0x01 (타입 - "감정/상태 패킷이야")
Byte 2: 0~255 (순번 - "몇 번째 패킷이야")
Byte 3: 0x02 (길이 - "내용은 2바이트야")
Byte 4: [감정코드 4bit][스트레스 1bit][주의력저하 1bit][졸음 1bit][종료 1bit]
Byte 5: [감정강도 3bit][상태강도 3bit][부정감정 1bit][예약 1bit]
Byte 6: CRC8 (체크섬 - "데이터가 안 깨졌는지 확인용")
Byte 7: 0xFE (끝 마커 - "패킷 끝이야")
```

**예시:**
```
운전자가 화나고(angry) 스트레스 받는 상태:
  감정코드 = 2 (분노)
  스트레스 = 1 (True)
  주의력저하 = 0 (False)
  졸음 = 0 (False)

  Byte4 = 0010_1_0_0_0 = 0x28

→ 패킷: AA 01 07 02 28 62 ?? FE
```

**K-FER(7개) → Protocol(6개) 변환 테이블:**
```
K-FER angry(0)    → Protocol 2 (분노)
K-FER anxious(1)  → Protocol 0 (공포)
K-FER happy(2)    → Protocol 4 (행복)
K-FER hurt(3)     → Protocol 3 (슬픔/혐오)  ← sad와 합쳐짐!
K-FER neutral(4)  → Protocol 5 (중립)
K-FER sad(5)      → Protocol 3 (슬픔/혐오)  ← hurt와 합쳐짐!
K-FER surprised(6)→ Protocol 1 (놀람)
```
> 왜 합쳐지나: K-FER은 7개 감정이지만 차량 프로토콜은 6개만 지원. sad(슬픔)와 hurt(상처)는 비슷하니까 하나(코드3)로 합침.

---

### 4.10 테스트 파일

#### `sensing/test_e2e_verification.py` — E2E 검증 (42개 테스트)

```
역할: 실제 센서 없이도 파이프라인 로직이 맞는지 검증
실행: python test_e2e_verification.py
결과: 42개 테스트 전부 통과해야 정상
```

**테스트 항목:**
```
TEST 1:  Config 로드/접근
TEST 2:  Logger 초기화
TEST 3:  버퍼 (Latest, Ring1D, BioQueues)
TEST 4:  Temporal Smoother (다수결, EMA)
TEST 5:  Fatigue Tracker (3가지 조건)
TEST 6:  Gateway Sender (연결/전송)
TEST 7:  PacketEncoder 매핑 (K-FER → Protocol)
TEST 8:  Degraded Mode 판별
TEST 9:  E2EPipeline 통합 시나리오 7개
TEST 10: 10-class 출력 매핑 확인
```

---

## 5. 모달리티별 전체 정리 (풀 경로 포함)

> 각 센서가 어떤 코드를 거쳐서 어떤 출력을 내는지, **실제 파일 풀 경로**와 함께 정리

---

### 5.1 Face (카메라)

#### 센서 수집

| 항목 | 내용 |
|------|------|
| **센서** | Intel RealSense D435 (시리얼: `021222070391`) |
| **해상도** | 1280×720, 30fps, BGR |
| **드라이버 코드** | `/home/ajy/Jetson_thor/sensing/raw_sensing_code/Real-Time-Driver-State-Emotion-Monitoring-System-ysh_sensing_260305/sensing/realsense.py` |
| **드라이버 함수** | `run_realsense(shutdown_event, device_serial, on_frame, ...)` |
| **버퍼** | `Latest` 클래스 (최신 프레임 1개만 보관) |
| **버퍼 정의** | `/home/ajy/Jetson_thor/sensing/raw_sensing_code/Real-Time-Driver-State-Emotion-Monitoring-System-ysh_sensing_260305/sensing/sensing_main.py` → `Latest` 클래스 |

#### 전처리 (얼굴 검출 + AU 추출)

| 항목 | 내용 |
|------|------|
| **코드** | `/home/ajy/Jetson_thor/sensing/raw_sensing_code/Real-Time-Driver-State-Emotion-Monitoring-System-ysh_sensing_260305/sensing/kmer_inferencer.py` |
| **함수** | `KMERInferencer._process_face(frame_bgr)` |
| **내부 동작** | MediaPipe FaceMesh → 468개 랜드마크 → bbox 추출(20% 패딩) → 224×224 크롭 → AU 좌표 8개 영역 추출 |
| **AU 영역 정의** | 같은 파일 내 `_AU_REGIONS` 상수 (이마, 왼눈, 오른눈, 코, 왼볼, 오른볼, 입, 턱) |

#### Expert 1: KFERExpert (얼굴 감정 7개)

| 항목 | 내용 |
|------|------|
| **코드** | `/home/ajy/Jetson_thor/multimodal_dms/experts/kfer_expert.py` |
| **클래스** | `KFERExpert` (line 75) |
| **내부 모델** | `AUFERModel` — 정의: `/home/ajy/Jetson_thor/emotion_system/models/fer_model.py` (line 35) |
| **백본** | MobileViTv2 — 정의: `/home/ajy/Jetson_thor/emotion_system/models/backbones/mobilevit_v3.py` → `MobileViTBackbone` |
| **AU 퓨전** | Cross-Attention — 정의: `/home/ajy/Jetson_thor/emotion_system/models/fusion/cross_attention.py` → `CrossAttentionFusion` |
| **체크포인트** | `/home/ajy/Jetson_thor/emotion_system/result/best.pth` |

```
입력: 얼굴 크롭 (3, 224, 224) + AU 좌표 (8, 2)
      │
      ▼
MobileViTv2 백본 (5M params, 1회 forward)
      │ feature_map [B, 384, h, w]
      ▼
AU RoI Extract → 8개 AU 토큰 추출
      │
      ▼
CrossAttentionFusion (CLS가 AU 토큰에 attention)
      │
      ▼
FER Head → 7-class softmax

출력:
  kfer_probs (7,)  : [angry 0.05, anxious 0.02, happy 0.85, hurt 0.01, neutral 0.05, sad 0.01, surprised 0.01]
  kfer_meta  (2,)  : [quality=0.92, entropy=0.45]
  face_stats (3,)  : [max_conf=0.85, mean_conf=0.14, std_conf=0.28]
```

#### Expert 2: FACSAuxExpert (눈/표정근육)

| 항목 | 내용 |
|------|------|
| **코드** | `/home/ajy/Jetson_thor/multimodal_dms/experts/facs_aux.py` |
| **클래스** | `FACSAuxExpert` (line 122) |
| **PERCLOS 계산** | `/home/ajy/Jetson_thor/emotion_system/models/drowsiness/perclos.py` → `compute_ear()`, `compute_perclos()` |
| **EAR 임계값** | 0.21 미만 → 눈 감김 (설정: `sensing_config.yaml` → `thresholds.ear_closed`) |

```
입력: MediaPipe 랜드마크 (468개)
      │
      ├─ EAR (Eye Aspect Ratio) 계산 → 눈 높이/너비
      ├─ PERCLOS 계산 → 최근 N 프레임 중 눈 감긴 비율
      └─ FACS geometric scores → 6개 표정근육 강도

출력:
  perclos_ear  (2,)  : [PERCLOS=0.15, mean_EAR=0.28]
  facs_scores  (6,)  : [이마, 좌눈썹, 우눈썹, 코주름, 좌입꼬리, 우입꼬리]
```

#### Face가 KMERFusion에 제공하는 토큰 (5개)

| 토큰 | 이름 | 차원 | 코드 출처 |
|------|------|------|----------|
| T1 | kfer_probs | 7 → 64 | `kfer_expert.py` → `KFERExpert` |
| T2 | kfer_meta | 2 → 64 | `kfer_expert.py` → `KFERExpert` |
| T3 | face_stats | 3 → 64 | `kmer_inferencer.py` → `_process_face()` |
| T11 | perclos_ear | 2 → 64 | `facs_aux.py` → `FACSAuxExpert` |
| T12 | facs_scores | 6 → 64 | `facs_aux.py` → `FACSAuxExpert` |

#### Face로 직접 판정하는 항목

| 판정 항목 | 판정 로직 | 코드 위치 |
|-----------|----------|----------|
| **Emotion 6개** | kfer_probs argmax → KFER_TO_PROTOCOL 매핑 | `/home/ajy/Jetson_thor/sensing/pipeline/e2e_pipeline.py` → `_compute_ten_class()` |
| **Low Attention** | PERCLOS ∈ [0.2, 0.4) | `/home/ajy/Jetson_thor/sensing/pipeline/e2e_pipeline.py` → `_compute_ten_class()` |
| **Drowsy** | PERCLOS ≥ 0.4 (NHTSA 표준) | `/home/ajy/Jetson_thor/sensing/pipeline/e2e_pipeline.py` → `_compute_ten_class()` |

---

### 5.2 Audio (마이크)

#### 센서 수집

| 항목 | 내용 |
|------|------|
| **센서** | RODE Wireless GO II 무선 마이크 |
| **스펙** | 48kHz, 8192 샘플/블록 |
| **드라이버 코드** | `/home/ajy/Jetson_thor/sensing/raw_sensing_code/Real-Time-Driver-State-Emotion-Monitoring-System-ysh_sensing_260305/sensing/rode.py` |
| **드라이버 함수** | `run_rode(shutdown_event, on_audio_chunk, ...)` / `pick_input_device()` |
| **버퍼** | `Ring1D` 클래스 (원형 버퍼, 2초 = 96,000 샘플) |
| **버퍼 정의** | `/home/ajy/Jetson_thor/sensing/raw_sensing_code/Real-Time-Driver-State-Emotion-Monitoring-System-ysh_sensing_260305/sensing/sensing_main.py` → `Ring1D` 클래스 |

#### 전처리 (리샘플링)

| 항목 | 내용 |
|------|------|
| **코드** | `/home/ajy/Jetson_thor/sensing/raw_sensing_code/Real-Time-Driver-State-Emotion-Monitoring-System-ysh_sensing_260305/sensing/kmer_inferencer.py` |
| **함수** | `_resample_audio(audio, src_sr=48000, dst_sr=16000)` (librosa 사용) |
| **변환** | 48kHz → 16kHz (96,000 → 32,000 샘플) |

#### Expert 3: Emotion2VecExpert (음성 감정 9개)

| 항목 | 내용 |
|------|------|
| **코드** | `/home/ajy/Jetson_thor/multimodal_dms/experts/audio_expert.py` |
| **클래스** | `Emotion2VecExpert` (line 20) |
| **기반 모델** | FunASR `emotion2vec_plus_large` (frozen, 1024d) |
| **의존 라이브러리** | `funasr` |

```
입력: 16kHz 모노 오디오 (32,000 샘플 ≈ 2초)
      │
      ▼
emotion2vec_plus_large (frozen encoder)
      │
      ▼
출력:
  emo2vec_probs (9,) : [angry, disgusted, fearful, happy, neutral,
                        other, sad, surprised, unknown]
```

#### Expert 4: AudeeringExpert (음성 arousal/valence)

| 항목 | 내용 |
|------|------|
| **코드** | `/home/ajy/Jetson_thor/multimodal_dms/experts/audio_expert.py` |
| **클래스** | `AudeeringExpert` (line 100) |
| **기반 모델** | audeering wav2vec2 (arousal/valence/dominance 3-output) |
| **의존 라이브러리** | `transformers` |

```
입력: 16kHz 모노 오디오
      │
      ▼
wav2vec2 (audeering pretrained)
      │
      ▼
출력:
  audeering_avd (3,) : [arousal=0.65, valence=0.72, dominance=0.45]
```

#### 오디오 품질 (수작업 계산)

| 항목 | 내용 |
|------|------|
| **코드** | `/home/ajy/Jetson_thor/sensing/raw_sensing_code/Real-Time-Driver-State-Emotion-Monitoring-System-ysh_sensing_260305/sensing/kmer_inferencer.py` |
| **함수** | `_compute_audio_quality(audio)` |

```
출력:
  audio_quality (3,) : [RMS_norm, ZCR(영점교차율), SNR_est]
  audio_valid = True if RMS > 0.01
```

#### Audio가 KMERFusion에 제공하는 토큰 (3개)

| 토큰 | 이름 | 차원 | 코드 출처 |
|------|------|------|----------|
| T4 | emo2vec_probs | 9 → 64 | `audio_expert.py` → `Emotion2VecExpert` |
| T5 | audeering_avd | 3 → 64 | `audio_expert.py` → `AudeeringExpert` |
| T6 | audio_quality | 3 → 64 | `kmer_inferencer.py` → `_compute_audio_quality()` |

#### Audio가 기여하는 판정 항목

| 판정 항목 | 기여 방식 |
|-----------|----------|
| **Stress** | audeering → arousal 예측에 영향 → arousal > 0.6이면 Stress 판정에 간접 기여 |
| **Fatigue** | arousal 예측 정확도 향상 → arousal < 0.3이 30초 지속 시 Fatigue |

---

### 5.3 Bio (손목시계)

#### 센서 수집

| 항목 | 내용 |
|------|------|
| **센서** | ADI Study Watch (BLE 무선) |
| **MAC** | `F1-18-1C-93-7C-42` |
| **BLE 동글** | VID=0x0456, PID=0x2CFE |
| **드라이버 코드** | `/home/ajy/Jetson_thor/sensing/raw_sensing_code/Real-Time-Driver-State-Emotion-Monitoring-System-ysh_sensing_260305/sensing/watch.py` |
| **드라이버 함수** | `run_watch(shutdown_event, on_ppg, on_eda, on_temp)` / `find_dongle()` |
| **버퍼** | `BioQueues` 클래스 (3개 deque: PPG, EDA, TEMP) |
| **버퍼 정의** | `/home/ajy/Jetson_thor/sensing/raw_sensing_code/Real-Time-Driver-State-Emotion-Monitoring-System-ysh_sensing_260305/sensing/sensing_main.py` → `BioQueues` 클래스 |

#### 3가지 센서 스트림

| 스트림 | 콜백 | 데이터 형식 |
|--------|------|-----------|
| **PPG** (맥박/광용적맥파) | `on_ppg(ts, d1, d2)` | 2채널 PPG 신호 |
| **EDA** (전기피부반응) | `on_eda(ts, real)` | 피부전도 실수값 |
| **TEMP** (체온) | `on_temp(ts, skin_c)` | 피부 온도 (°C) |

#### Expert 5: BioExpert (생체 특징 15개)

| 항목 | 내용 |
|------|------|
| **코드** | `/home/ajy/Jetson_thor/multimodal_dms/experts/bio_expert.py` |
| **함수** | `extract_bio_features_v2()` (line 124) |
| **의존 라이브러리** | `neurokit2`, `scipy` |
| **모델** | 없음 (수작업 특징 추출, 딥러닝 아님) |

```
입력: PPG/EDA/TEMP 시계열 데이터
      │
      ├─ PPG → NeuroKit2 HRV 분석
      │   → bvp_features (4,) : [mean_hr=72.5, sdnn=45.2, rmssd=38.1, lf_hf=1.8]
      │
      ├─ EDA → NeuroKit2 SCR 분석
      │   → eda_features (5,) : [mean_scl=2.1, std_scl=0.5, n_peaks=3, amplitude=0.8, auc=15.2]
      │
      ├─ HR+TEMP → 직접 통계 계산
      │   → hr_temp_features (6,) : [hr_mean, hr_std, hr_range, temp_mean, temp_slope, temp_range]
      │
      └─ 품질 계산
          → bio_quality (3,)
```

#### Bio가 KMERFusion에 제공하는 토큰 (4개)

| 토큰 | 이름 | 차원 | 코드 출처 |
|------|------|------|----------|
| T7 | bvp_features | 4 → 64 | `bio_expert.py` → `extract_bio_features_v2()` |
| T8 | eda_features | 5 → 64 | `bio_expert.py` → `extract_bio_features_v2()` |
| T9 | hr_temp_features | 6 → 64 | `bio_expert.py` → `extract_bio_features_v2()` |
| T10 | bio_quality | 3 → 64 | `bio_expert.py` → `extract_bio_features_v2()` |

#### Bio가 기여하는 판정 항목

| 판정 항목 | 기여 방식 |
|-----------|----------|
| **Stress** | HRV LF/HF + EDA peaks → arousal 예측 정확도 향상 → Stress 판정 보조 |
| **Fatigue** | arousal < 0.3이 30초 지속 조건 판정에 직접 기여 |

---

### 5.4 Meta 토큰 (2개)

| 토큰 | 이름 | 차원 | 코드 출처 | 설명 |
|------|------|------|----------|------|
| T13 | cross_modal | 3 → 64 | `kmer_inferencer.py` → `_compute_cross_modal()` | 모달리티 간 일치도 (face-audio 일치, AV 일관성, entropy 격차) |
| T14 | validity_flags | 3 → 64 | `kmer_inferencer.py` → `_build_feature_dict()` | [face 유효, audio 유효, bio 유효] |

---

### 5.5 퓨전 (모든 토큰 합치기)

#### KMERFusion 모델

| 항목 | 내용 |
|------|------|
| **코드** | `/home/ajy/Jetson_thor/multimodal_dms/fusion/kmer_fusion.py` |
| **클래스** | `KMERFusion(nn.Module)` (line 74) |
| **파라미터** | ~144,000개 (매우 가벼움) |
| **체크포인트** | `/home/ajy/Jetson_thor/multimodal_dms/results_kmer/best_model.pth` |
| **학습 코드** | `/home/ajy/Jetson_thor/multimodal_dms/train_kmer.py` |

```
Face 토큰 5개 (T1~T3, T11~T12)  ─┐
Audio 토큰 3개 (T4~T6)           ─┤→ 14개 토큰 + CLS = 15개 × 64d
Bio 토큰 4개 (T7~T10)            ─┤
Meta 토큰 2개 (T13~T14)          ─┘
      │
      ▼ Stage 1: Token Formation (각 토큰을 Linear로 64d에 투영)
      │
      ▼ Stage 2: Intra-Modal Pool-FFN (같은 모달리티 토큰끼리 지역 패턴)
      │   Face(T1~T3,T11~T12) / Audio(T4~T6) / Bio(T7~T10) 각각
      │
      ▼ Stage 3: Global Cross-Modal MHSA (4-head, 모든 토큰 간 attention)
      │   센서 고장 토큰은 valid_mask로 마스킹(무시)
      │
      ▼ Stage 4: CLS Pooling (CLS 토큰이 전체 요약 → 64d 벡터 1개)
      │
      ▼ Stage 5: Bi-GRU (시간 맥락, 현재 미사용)
      │
      ▼ Stage 6: Output Heads
      │
      ├─ arousal head: 64d → Linear(64,1) → sigmoid → [0, 1]
      ├─ valence head: 64d → Linear(64,1) → sigmoid → [0, 1]
      └─ drowsy head:  64d → Linear(64,3) → softmax → [alert, drowsy, sleeping]
```

#### CompoundEmotionMapper (복합 감정)

| 항목 | 내용 |
|------|------|
| **코드** | `/home/ajy/Jetson_thor/multimodal_dms/fusion/compound_emotion.py` |
| **클래스** | `CompoundEmotionMapper` (line 83) |
| **방식** | 규칙 기반 (AI 모델 아님) |

```
K-FER 감정 + arousal 레벨 → 13가지 복합 감정

예시:
  happy  + low arousal  → "happy"
  happy  + mid arousal  → "positive_engaged"
  happy  + high arousal → "excited"
  neutral + low arousal → "calm"
  anxious + high arousal → "stressed"
  sad    + low arousal  → "depressed"
  drowsy 상태            → "drowsy"
```

---

### 5.6 후처리 파이프라인 (퓨전 후 → 최종 판정)

#### Temporal Smoother (떨림 방지)

| 항목 | 내용 |
|------|------|
| **코드** | `/home/ajy/Jetson_thor/sensing/pipeline/temporal_smoother.py` |
| **클래스** | `MultimodalTemporalSmoother` / `MajorityVoteSmoother` / `EMASmoother` |

```
KMERFusion 원본 출력 → Temporal Smoother
  │
  ├─ kfer_emotion → MajorityVoteSmoother(window=7)  → smoothed_kfer_emotion
  ├─ drowsy       → MajorityVoteSmoother(window=5)  → smoothed_drowsy
  ├─ compound     → MajorityVoteSmoother(window=7)  → smoothed_compound
  ├─ arousal      → EMASmoother(alpha=0.3)           → smoothed_arousal
  └─ valence      → EMASmoother(alpha=0.3)           → smoothed_valence
```

#### 10-class 판정

| 항목 | 내용 |
|------|------|
| **코드** | `/home/ajy/Jetson_thor/sensing/pipeline/e2e_pipeline.py` |
| **함수** | `E2EPipeline._compute_ten_class(smoothed)` |

```
smoothed 결과 → 10가지 판정

Emotion 6개:
  smoothed_kfer_emotion → KFER_TO_PROTOCOL 매핑 → emotion_code (0~5)
  매핑 테이블 정의: /home/ajy/Jetson_thor/multimodal_dms/gateway/packet_encoder.py
    KFER_TO_PROTOCOL = {0:2, 1:0, 2:4, 3:3, 4:5, 5:3, 6:1}

Driver State 4개:
  stress        = (kfer ∈ {angry,anxious}) AND (arousal > 0.6)
  low_attention = (0.2 ≤ PERCLOS < 0.4)
  drowsy        = (PERCLOS ≥ 0.4) OR (drowsy_level ≥ 1)
  fatigue       = FatigueTracker 결과
```

#### Fatigue Tracker (피로 감지)

| 항목 | 내용 |
|------|------|
| **코드** | `/home/ajy/Jetson_thor/sensing/pipeline/fatigue_tracker.py` |
| **클래스** | `FatigueTracker` (line 36) |

```
3개 조건 중 하나라도 충족 → fatigue = True

조건1: compound_label ∈ {"depressed", "calm"} 이 30초 이상 연속
조건2: arousal < 0.3 이 30초 이상 연속
조건3: drowsy 가 10초 이상 연속
```

#### PacketEncoder (8바이트 패킷)

| 항목 | 내용 |
|------|------|
| **코드** | `/home/ajy/Jetson_thor/multimodal_dms/gateway/packet_encoder.py` |
| **클래스** | `PacketEncoder` (line 149) |
| **보조 클래스** | `StressDetector` (line 86), `AttentionDetector` (line 105), `DrowsyDetector` (line 120), `NegativeEmotionDetector` (line 134) |

```
10-class 판정 결과 → 8바이트 이진 패킷

Byte 0: 0xAA (시작)
Byte 1: 0x01 (타입)
Byte 2: 순번 (0~255)
Byte 3: 0x02 (길이)
Byte 4: [감정코드 4bit][스트레스 1bit][주의력저하 1bit][졸음 1bit][종료 1bit]
Byte 5: [감정강도 3bit][상태강도 3bit][부정감정 1bit][예약 1bit]
Byte 6: CRC8 (체크섬)
Byte 7: 0xFE (끝)
```

#### Gateway Sender (USB 전송)

| 항목 | 내용 |
|------|------|
| **코드** | `/home/ajy/Jetson_thor/sensing/pipeline/gateway_sender.py` |
| **클래스** | `GatewaySender` (line 33) / `NullGatewaySender` |
| **포트** | `/dev/ttyUSB0` (설정: `sensing_config.yaml` → `gateway.port`) |
| **속도** | 115200 baud |

---

### 5.7 모달리티 × 판정항목 기여 매트릭스

```
                    Face 토큰 5개
                         │
                    Audio 토큰 3개  ──→  KMERFusion ──→ arousal ──→ Stress, Fatigue
                         │                              valence
                    Bio 토큰 4개                        drowsy ──→ Drowsy, Fatigue
                         │
                    Meta 토큰 2개
                    (cross_modal + validity)
```

| 판정 항목 | Face | Audio | Bio | 퓨전 후 | 판정 코드 위치 |
|-----------|------|-------|-----|---------|---------------|
| **Emotion 6개** | ◎ | | | | `e2e_pipeline.py` → `_compute_ten_class()` |
| **Stress** | ○ | ○ | ○ | ◎ | `e2e_pipeline.py` → `_compute_ten_class()` |
| **Low Attention** | ◎ | | | | `e2e_pipeline.py` → `_compute_ten_class()` |
| **Drowsy** | ◎ | | | ○ | `e2e_pipeline.py` → `_compute_ten_class()` |
| **Fatigue** | | | ○ | ◎ | `fatigue_tracker.py` → `FatigueTracker.update()` |

```
◎ = 주 입력 (이 모달리티가 직접 판정)
○ = 보조 입력 (퓨전을 통해 간접 기여)
```

---

### 5.8 전체 데이터 흐름 한눈에 보기 (풀 경로)

```
[센서 수집] ──────────────────────────────────────────────────────────────────
│
├─ RealSense D435 ─────→ realsense.py ──→ Latest 버퍼 ────┐
│   (30fps, 1280×720)     run_realsense()   frame_main     │
│                                                          │
├─ RODE Wireless GO II ─→ rode.py ──────→ Ring1D 버퍼 ─────┤ ModelInputs
│   (48kHz mono)          run_rode()       audio           │ (공유)
│                                                          │
└─ ADI Study Watch ─────→ watch.py ─────→ BioQueues 버퍼 ──┘
    (BLE: PPG/EDA/TEMP)   run_watch()      bio
│
│  0.1초(10Hz)마다 버퍼에서 최신 데이터 가져옴
│
[AI 추론] ────────────────────────────────────────────────────────────────────
│
│  kmer_inferencer.py → KMERInferencer.forward()
│
├─ _process_face(frame) ────────────────────────────────────────────────────
│   │ MediaPipe FaceMesh → 얼굴 크롭 224×224 + AU 좌표 8개
│   ├─ kfer_expert.py → KFERExpert     → kfer_probs(7), meta(2), stats(3)
│   └─ facs_aux.py    → FACSAuxExpert  → perclos_ear(2), facs_scores(6)
│
├─ _process_audio(audio) ───────────────────────────────────────────────────
│   │ librosa 48kHz→16kHz 리샘플링
│   ├─ audio_expert.py → Emotion2VecExpert → emo2vec_probs(9)
│   ├─ audio_expert.py → AudeeringExpert   → audeering_avd(3)
│   └─ 직접 계산                            → audio_quality(3)
│
├─ _process_bio(ppg, eda, temp) ────────────────────────────────────────────
│   └─ bio_expert.py → extract_bio_features_v2()
│      → bvp(4) + eda(5) + hr_temp(6) + bio_quality(3)
│
├─ _build_feature_dict() → 14개 토큰 + valid_mask
│
└─ kmer_fusion.py → KMERFusion.forward()
   → arousal, valence, drowsy
│
│  compound_emotion.py → CompoundEmotionMapper → compound_label (13가지)
│
[후처리] ─────────────────────────────────────────────────────────────────────
│
│  e2e_pipeline.py → E2EPipeline.process_cycle()
│
├─ temporal_smoother.py → 다수결(7프레임) + EMA(alpha=0.3) → 안정화
├─ e2e_pipeline.py      → _compute_ten_class() → 6감정 + Stress/LowAttn/Drowsy
├─ fatigue_tracker.py   → FatigueTracker.update() → Fatigue
├─ packet_encoder.py    → PacketEncoder.encode() → 8바이트 패킷
└─ gateway_sender.py    → GatewaySender.send()   → USB 시리얼 전송
│
▼
[차량 ECU: 운전자 상태에 따라 경고/조치]
```

---

## 6. 14개 토큰 전체 정리

KMERFusion에 들어가는 14개 토큰 + 1개 CLS 토큰:

| 토큰 | 이름 | 차원 | 어디서 오나 | 무슨 정보 |
|------|------|------|-----------|----------|
| T0 | CLS | 64 | 학습 파라미터 | 전체 요약용 (BERT의 [CLS]과 같음) |
| T1 | kfer_probs | 7 | KFERExpert | 7개 감정 확률 |
| T2 | kfer_meta | 2 | KFERExpert | 품질(quality) + 불확실성(entropy) |
| T3 | face_stats | 3 | KFERExpert | 확률 통계 (max, mean, std) |
| T4 | emo2vec_probs | 9 | Emotion2Vec | 9개 음성감정 확률 |
| T5 | audeering_avd | 3 | Audeering | arousal, valence, dominance |
| T6 | audio_quality | 3 | 수작업 계산 | RMS, 영점교차율, SNR |
| T7 | bvp_features | 4 | BioExpert | 심박 HRV 특징 |
| T8 | eda_features | 5 | BioExpert | 피부전도 특징 |
| T9 | hr_temp_features | 6 | BioExpert | 심박+체온 특징 |
| T10 | bio_quality | 3 | BioExpert | 생체신호 품질 |
| T11 | perclos_ear | 2 | FACSAux | 눈감김비율 + 눈열림정도 |
| T12 | facs_scores | 6 | FACSAux | 6개 표정근육 점수 |
| T13 | cross_modal | 3 | 수작업 계산 | 모달리티 간 일치도 |
| T14 | validity_flags | 3 | 자동 | [face유효, audio유효, bio유효] |

각 토큰은 64차원으로 투영됨: `(원래차원) → Linear(원래차원, 64) → (64)`

---

## 7. 각 판정 항목의 구체적 로직

### 7.1 감정 6개 (Emotion Code 0~5)

```python
# KFERExpert가 7개 감정 확률을 예측
kfer_probs = [0.01, 0.02, 0.85, 0.01, 0.05, 0.03, 0.03]
#             angry  anx   happy  hurt  neut  sad   surp

# 가장 높은 확률의 감정 선택
kfer_top1 = 2  # → happy

# 7개 → 6개로 변환 (sad와 hurt 합침)
KFER_TO_PROTOCOL = {0:2, 1:0, 2:4, 3:3, 4:5, 5:3, 6:1}
emotion_code = KFER_TO_PROTOCOL[2]  # → 4 (행복)
```

### 7.2 Stress (스트레스)

```python
# 조건: 부정감정(angry 또는 anxious) + 높은 각성도
is_negative = kfer_top1 in {0, 1}  # angry(0) 또는 anxious(1)
high_arousal = arousal > 0.6       # 각성도 60% 초과

stress = is_negative and high_arousal
# 생체신호(시계) 없으면 → arousal 없으니까 부정감정만으로 판정
```

### 7.3 Low Attention (주의력 저하)

```python
# PERCLOS = 최근 프레임에서 눈 감긴 비율
# EAR(Eye Aspect Ratio) < 0.21이면 "눈 감김"
low_attention = (0.2 <= perclos < 0.4)
# 20~40%: 눈을 좀 많이 감긴 하지만 졸음까진 아님
```

### 7.4 Drowsy (졸음)

```python
drowsy = perclos >= 0.4
# 40% 이상: 눈을 거의 감고 있음 → 졸음 (NHTSA 표준)
```

### 7.5 Fatigue (피로)

```python
# 3가지 중 하나라도 해당하면 피로
fatigue = (
    depressed_or_calm_30sec  # 우울/침착 상태가 30초 이상 지속
    or low_arousal_30sec     # 각성도 0.3 미만이 30초 이상 지속
    or drowsy_10sec          # 졸음 상태가 10초 이상 지속
)
```

---

## 8. Shadow Mode란?

```
기존 방식: 센서 → inference_loop()  → print() 출력
새 방식:   센서 → E2EPipeline       → 10-class + 패킷 + USB 전송

Shadow Mode = 둘 다 동시에 실행해서 결과 비교
```

**왜 Shadow Mode를 쓰나:**
- 새 파이프라인이 기존 것과 결과가 같은지 확인하기 위해
- 갑자기 교체하면 뭔가 잘못될 수 있으니까
- 검증 완료 후 기존 것을 제거하고 새 것만 사용

**설정:**
```yaml
shadow_mode:
  enabled: true    # Shadow mode ON
  log_comparison: true  # 비교 로그 출력
```

---

## 9. Jetson에서 실행하려면

### 필요한 것
```
하드웨어:
  - NVIDIA Jetson Orin (GPU)
  - Intel RealSense D435 (USB 카메라)
  - RODE Wireless GO II (무선 마이크)
  - ADI Study Watch + BLE 동글 (손목시계)

소프트웨어 (pip):
  - torch, torchvision     (AI 프레임워크)
  - pyrealsense2            (카메라 드라이버)
  - sounddevice             (마이크 드라이버)
  - mediapipe               (얼굴 검출)
  - librosa                 (오디오 리샘플링)
  - funasr                  (emotion2vec 모델)
  - transformers            (audeering 모델)
  - neurokit2               (생체신호 처리)
  - pyserial                (USB 시리얼 통신)
  - scipy, numpy, opencv-python

모델 체크포인트:
  - emotion_system/result/best.pth           (K-FER)
  - multimodal_dms/results_kmer/best_model.pth (KMERFusion)
```

### 실행 방법
```bash
cd Jetson_thor/sensing

# 기본 실행 (Shadow mode)
python sensing_main_shadow.py

# 새 파이프라인만 (기존 비교 안 함)
python sensing_main_shadow.py --shadow_only

# 오디오 모델 끄기 (메모리 부족 시)
python sensing_main_shadow.py --no_audio_experts

# 게이트웨이 끄기 (USB 연결 안 됐을 때)
python sensing_main_shadow.py --gateway_off

# CPU 모드 (GPU 없을 때)
python sensing_main_shadow.py --device cpu
```

### 테스트 (센서 없이 로직 검증)
```bash
cd Jetson_thor/sensing
python test_e2e_verification.py
# → 42개 테스트 전부 통과해야 정상
```

---

## 10. 파일 의존성 맵 (뭐가 뭘 import하나)

```
sensing_main_shadow.py
  ├── config/__init__.py          → load_config()
  ├── core/logger.py              → setup_logging(), get_logger()
  ├── core/buffers.py             → Latest, Ring1D, BioQueues, ModelInputs
  │   └── (원본) sensing_main.py  → 실제 클래스 정의
  ├── realsense.py                → run_realsense()
  ├── rode.py                     → run_rode()
  ├── watch.py                    → run_watch()
  └── pipeline/e2e_pipeline.py    → E2EPipeline
        ├── pipeline/temporal_smoother.py → MultimodalTemporalSmoother
        ├── pipeline/fatigue_tracker.py   → FatigueTracker
        ├── pipeline/gateway_sender.py    → GatewaySender
        └── (lazy import)
            ├── kmer_inferencer.py        → KMERInferencer
            │   ├── experts/kfer_expert.py     → KFERExpert
            │   ├── experts/facs_aux.py        → FACSAuxExpert
            │   ├── experts/audio_expert.py    → Emotion2VecExpert, AudeeringExpert
            │   ├── experts/bio_expert.py      → extract_bio_features_v2()
            │   ├── fusion/kmer_fusion.py      → KMERFusion
            │   └── fusion/compound_emotion.py → CompoundEmotionMapper
            └── gateway/packet_encoder.py → PacketEncoder, KFER_TO_PROTOCOL
```

---

## 11. 용어 정리

| 용어 | 뜻 |
|------|-----|
| **K-FER** | Korean Facial Expression Recognition. 한국인 표정 7가지 분류 모델 |
| **K-MER** | Korean Multimodal Emotion Recognition. 얼굴+음성+생체 멀티모달 모델 |
| **KMERFusion** | K-MER의 퓨전 모델. 14개 토큰을 합쳐서 arousal/valence/drowsy 예측 |
| **PERCLOS** | Percentage of Eyelid Closure. 눈 감긴 비율 (졸음 판정 표준) |
| **EAR** | Eye Aspect Ratio. 눈 높이/너비 비율 (0.21 미만이면 눈 감김) |
| **arousal** | 각성도 (0~1). 높으면 흥분, 낮으면 차분/졸림 |
| **valence** | 쾌불쾌도 (0~1). 높으면 기분 좋음, 낮으면 기분 나쁨 |
| **FACS** | Facial Action Coding System. 표정 근육 코딩 시스템 |
| **AU** | Action Unit. 얼굴 근육 움직임 단위 (이마, 눈, 코, 입 등 8영역) |
| **compound emotion** | 복합 감정. 기본감정+각성도 조합 (예: happy+high→excited) |
| **emotion2vec** | FunASR의 음성 감정 인식 모델 (1024차원) |
| **audeering** | wav2vec2 기반 음성 arousal/valence 예측 모델 |
| **MobileViTv2** | 경량 비전 트랜스포머 (5M params, Jetson에서 돌아감) |
| **MHSA** | Multi-Head Self-Attention. 트랜스포머의 핵심 구조 |
| **Pool-FFN** | EfficientFormer 스타일 지역 토큰 믹서 |
| **CLS 토큰** | Classification 토큰. 전체 정보를 요약하는 특수 토큰 |
| **EMA** | Exponential Moving Average. 지수이동평균 (급변 방지) |
| **Degraded Mode** | 일부 센서 고장 시 나머지로만 동작하는 모드 |
| **Shadow Mode** | 기존/새 파이프라인을 동시 실행하여 비교하는 모드 |
| **Gateway** | 차량 ECU와 통신하는 USB 시리얼 인터페이스 |
| **ECU** | Electronic Control Unit. 차량 전자제어장치 |
| **Bi-GRU** | Bidirectional GRU. 양방향 시계열 모델 (시간 맥락 반영) |
| **PPG** | Photoplethysmography. 광용적맥파 (맥박 측정) |
| **EDA** | Electrodermal Activity. 피부전도반응 (긴장/스트레스 지표) |
| **HRV** | Heart Rate Variability. 심박변이도 (자율신경 활성 지표) |
| **BLE** | Bluetooth Low Energy. 저전력 블루투스 (시계 통신) |
| **CRC8** | Cyclic Redundancy Check 8-bit. 데이터 무결성 검증 |

---

## 12. FAQ

**Q: 카메라가 고장나면 어떻게 되나?**
A: `NO_CAM` 모드로 전환. 모든 감정을 "중립(neutral)"로 보고하고, 모든 상태 플래그를 False로 설정. 시스템은 크래시하지 않음.

**Q: 마이크가 고장나면?**
A: `CAM+BIO` 또는 `CAM_ONLY` 모드로 전환. 오디오 토큰을 마스킹(무시)하고 카메라+생체신호만으로 판정.

**Q: 시계가 고장나면?**
A: `CAM+MIC` 또는 `CAM_ONLY` 모드로 전환. 생체 토큰을 마스킹하고 arousal을 감정만으로 추정.

**Q: 전체 추론 한 사이클에 얼마나 걸리나?**
A: Jetson Orin에서 약 50ms (20fps 가능하지만 10Hz로 운영).

**Q: sad와 hurt는 왜 합치나?**
A: 차량 프로토콜이 6개 감정만 지원. sad(슬픔)과 hurt(상처)는 운전 맥락에서 비슷하므로 "슬픔/혐오(코드3)"로 합침.

**Q: 모델 체크포인트는 어디서 나오나?**
A: `emotion_system/scripts/train.py`로 K-FER 학습 → `best.pth` 생성. `multimodal_dms/train_kmer.py`로 KMERFusion 학습 → `best_model.pth` 생성.

**Q: Shadow mode 검증이 끝나면?**
A: `sensing_main_shadow.py` → `sensing_main.py`로 교체. `shadow_mode.enabled: false`로 변경. 기존 `inference_loop` 제거.
