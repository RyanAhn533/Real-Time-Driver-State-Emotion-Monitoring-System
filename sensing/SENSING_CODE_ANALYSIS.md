# K-MER 센싱 코드 분석 보고서

> 분석 대상: `sensing/raw_sensing_code/Real-Time-Driver-State-Emotion-Monitoring-System-ysh_sensing_260305/`
> 분석 일자: 2026-03-09
> 목적: 현재 센싱 코드 구조 파악, 문제점 식별, 개선 방향 제시

---

## 1. 현재 코드 구조

### 1.1 파일 목록 (8개 파일, ~1,800 lines)

| 파일 | Lines | 역할 |
|------|-------|------|
| `sensing_main.py` | 240 | 메인 오케스트레이터 (5개 스레드 + 추론 루프) |
| `kmer_inferencer.py` | 605 | K-MER 멀티모달 퓨전 추론기 (핵심) |
| `realsense.py` | 51 | Intel RealSense D435 카메라 드라이버 |
| `rode.py` | 101 | RODE Wireless GO II 마이크 드라이버 |
| `watch.py` | 187 | ADI Study Watch BLE 생체신호 드라이버 |
| `fer_inferencer.py` | 209 | K-FER 단독 추론기 (얼굴 감정) |
| `inference.py` | 283 | 구형 FER 추론 + OpenCV 시각화 |
| `test_sensors.py` | 133 | 센서 연결 테스트 유틸리티 |

### 1.2 데이터 흐름도

```
┌──────────────── 센서 스레드 (4개, daemon) ────────────────┐
│                                                           │
│  RealSense D435 Main     RODE Wireless GO II   ADI Watch │
│  (30fps, 1280x720)       (48kHz, mono)         (BLE)    │
│  serial: 021222070391     blocksize: 8192       PPG/EDA  │
│         │                       │               /TEMP    │
│  on_frame_main()         on_audio_chunk()       │        │
│         │                       │          on_ppg/eda/   │
│         │                       │          on_temp()     │
└─────────┼───────────────────────┼───────────────┼────────┘
          │                       │               │
    ┌─────▼───────────────────────▼───────────────▼─────┐
    │          ModelInputs (Thread-Safe Buffers)         │
    │  frame_main: Latest         (최신 1프레임)        │
    │  audio:      Ring1D(96000)  (2초 @ 48kHz)         │
    │  bio:        BioQueues(512) (PPG/EDA/TEMP deque)  │
    └───────────────────────┬───────────────────────────┘
                            │
    ┌───────────────────────▼───────────────────────────┐
    │          inference_loop (10Hz, 100ms 주기)        │
    │                                                    │
    │  KMERInferencer.forward(frame, audio, ppg,eda,temp)│
    │    ├── _process_face() → K-FER 7class + FACS       │
    │    ├── _process_audio() → emotion2vec + audeering  │
    │    ├── _process_bio() → BVP/EDA/HR_TEMP features   │
    │    ├── _build_feature_dict() → 14 tokens + mask    │
    │    └── KMERFusion → arousal, valence, drowsy       │
    │         └── CompoundEmotionMapper → 13-class       │
    │                                                    │
    │  출력: print() (1초마다 콘솔 출력)                 │
    └────────────────────────────────────────────────────┘
```

### 1.3 버퍼 클래스 (sensing_main.py)

| 클래스 | 용도 | Thread-Safety |
|--------|------|---------------|
| `Latest` | 최신 프레임 1장 보관 (ts, val) | `threading.Lock` |
| `Ring1D` | 오디오 순환 버퍼 (96K float32) | `threading.Lock` |
| `BioQueues` | PPG/EDA/TEMP 각각 deque(512) | `threading.Lock` |
| `ModelInputs` | 위 3개를 묶는 컨테이너 | 개별 lock |

### 1.4 KMERInferencer 토큰 구조 (14 tokens + 1 CLS = 15 tokens)

| Token | Feature | Dim | Source |
|-------|---------|-----|--------|
| T1 | kfer_probs | 7 | K-FER 7-class 확률 |
| T2 | kfer_meta | 2 | quality + entropy |
| T3 | face_stats | 3 | max/mean/std confidence |
| T4 | emo2vec_probs | 9 | emotion2vec 9-class |
| T5 | audeering_avd | 3 | arousal/valence/dominance |
| T6 | audio_quality | 3 | RMS/ZCR/SNR |
| T7 | bvp_features | 4 | mean_hr/sdnn/rmssd/lf_hf |
| T8 | eda_features | 5 | mean_scl/std_scl/n_peaks/amp/auc |
| T9 | hr_temp_features | 6 | hr stats + temp stats |
| T10 | bio_quality | 3 | validity flags |
| T11 | perclos_ear | 2 | PERCLOS + EAR |
| T12 | facs_scores | 6 | geometric AU features |
| T13 | cross_modal | 3 | face-audio consistency |
| T14 | validity_flags | 3 | face/audio/bio valid |

---

## 2. 문제점 분석

### 2.1 Critical (P0) - 즉시 해결 필요

#### C1. 하드코딩된 하드웨어 주소
- `realsense.py`: 카메라 시리얼 `"021222070391"`, `"405622073483"` 하드코딩
- `watch.py`: VID=0x0456, PID=0x2CFE, MAC=`"F1-18-1C-93-7C-42"` 하드코딩
- `rode.py`: SAMPLE_RATE=48000, BLOCKSIZE=8192 상수 고정
- **문제**: 하드웨어 교체 시 소스코드 수정 필요
- **개선**: YAML config 파일로 추출

#### C2. 센서 장애 시 전체 시스템 크래시
- `watch.py`: `find_dongle()` 실패 → `RuntimeError` → 전체 크래시
- `rode.py`: `pick_input_device()` 실패 → `RuntimeError` → 전체 크래시
- `realsense.py`: `pipe.start(cfg)` 실패 → return (스레드만 종료, graceful)
- **문제**: 마이크 또는 워치 하나 장애 시 카메라도 같이 죽음
- **개선**: Degraded mode (cam-only, cam+mic, cam+bio, full)

#### C3. demo_pipeline.py가 FER-only
- `DemoPipeline`은 `FERInferencer`만 사용 (K-FER 7-class)
- KMERFusion 미사용 → 멀티모달 퓨전 결과 게이트웨이에 전달 불가
- bio/audio hooks는 placeholder (`register_bio_processor`, `register_audio_processor`)
- **문제**: 실증 시연에서 멀티모달 결과 전송 불가
- **개선**: KMERFusion 기반 E2E 파이프라인 구축

#### C4. 게이트웨이 USB 전송 미구현
- `PacketEncoder`는 8-byte 패킷 생성 가능 (완성)
- 하지만 실제 USB serial 전송 코드 없음 (pyserial 코드 미작성)
- **문제**: 패킷은 만들지만 차량 ECU에 전달 불가
- **개선**: `GatewaySender` 클래스 (pyserial 기반) 추가

### 2.2 Performance (P1) - 성능 최적화

#### P1. 카메라 30fps → 추론 10Hz (프레임 낭비)
- RealSense: 초당 30 프레임 생성
- inference_loop: 초당 10회 실행
- **결과**: 매 초 20 프레임 버려짐 (66.7% 낭비)
- **현 상태**: `Latest` 버퍼가 최신 프레임만 유지하므로 기능적으로는 문제 없음
- **참고**: 이는 의도된 설계 (카메라 FPS와 추론 FPS 분리). 프레임 낭비는 아키텍처적 트레이드오프

#### P2. 매 추론 사이클마다 오디오 리샘플링
- `_process_audio()`: 매 호출 시 `librosa.resample(48000→16000)`
- 2초 오디오 = 96K samples → 32K samples 리샘플링
- **비용**: ~5-10ms per cycle (Jetson Orin 기준)
- **개선 가능**: 리샘플링 캐시 (동일 오디오 구간 재처리 방지)

#### P3. TensorRT 미적용
- K-FER (MobileViTv2) + KMERFusion 모두 PyTorch 직접 실행
- **참고**: 현재 모델 크기가 작아 (K-FER ~5M, KMERFusion ~144K) 급하진 않음
- **향후**: TensorRT 변환 시 ~2-3x 추론 속도 향상 가능

### 2.3 Reliability (P1) - 안정성

#### R1. RODE 재연결 로직 부족
- 스트림 close 시 재연결 1회 시도 후 예외 발생 시 1회 더 시도 (총 2회)
- 그 후 실패하면 스레드 종료
- **문제**: 장시간 운행 중 마이크 일시 연결 끊김 → 영구 손실
- **개선**: 지수 백오프 (2s, 4s, 8s, max 30s) 무한 재시도

#### R2. 워치독/헬스 모니터링 없음
- 센서 데이터 수신 중단 감지 메커니즘 없음
- 추론 루프 hang 감지 없음
- **문제**: 센서 침묵 장애 (silent failure) 감지 불가
- **개선**: 마지막 데이터 수신 시간 추적 + 타임아웃 경고

#### R3. print-only 로깅
- 모든 상태/에러 출력이 `print()` 사용
- 파일 로깅 없음, 로그 레벨 구분 없음
- **문제**: 장시간 운행 후 디버깅 불가 (콘솔 버퍼 유실)
- **개선**: Python `logging` 모듈 (콘솔 INFO + 파일 DEBUG)

#### R4. Graceful Shutdown 미흡
- 모든 스레드가 `daemon=True` → 메인 프로세스 종료 시 즉시 kill
- Watch SDK 정리 (`unsubscribe_stream`, `stop_sensor`) 실행 보장 안 됨
- **문제**: BLE 연결 정리 안 되면 다음 실행 시 연결 실패 가능
- **개선**: `shutdown.set()` → 각 스레드 정리 → timeout join

### 2.4 Architecture (P2) - 구조 개선

#### A1. 서브카메라 스레드 불필요 실행
- `RS_SUB` 스레드 생성되지만 `on_frame=None` → 프레임 처리 안 함
- USB 대역폭 + CPU 자원 소모
- **개선**: config에서 `enabled: false`로 비활성화

#### A2. 멀티모달 Temporal Smoothing 없음
- `demo_pipeline.py`에 `TemporalSmoother` (7-frame majority vote) 있지만 FER-only
- `KMERInferencer.forward()` 결과에 temporal smoothing 미적용
- **문제**: arousal/valence/compound 출력이 프레임 간 급변동 가능
- **개선**: EMA (연속값) + majority vote (이산값) 이중 smoothing

#### A3. Fatigue 판정 로직 없음
- Drowsy (PERCLOS ≥ 0.4) + Low Attention (PERCLOS [0.2, 0.4)) 만 존재
- Fatigue (만성 피로/저각성 지속) 판정 로직 없음
- **개선**: depressed/calm 상태 지속 시간 기반 Fatigue 판정기

#### A4. 10-class 출력 체계 미정의
- K-FER 7-class → Protocol 6 emotion은 정의됨 (KFER_TO_PROTOCOL)
- Stress/Low Attention/Drowsy status flag는 정의됨
- Fatigue 미정의, 전체 10-class 통합 명세 없음
- **개선**: OUTPUT_STATE_SPEC.md로 명세 정리

---

## 3. 개선 제안 (우선순위별)

### P0 (즉시)
1. **Config 시스템**: 하드코딩된 값 → YAML config 추출
2. **Degraded Mode**: 센서 장애 시 가용 모드 정의 + graceful fallback
3. **E2E Pipeline**: KMERFusion 기반 파이프라인 (demo_pipeline 대체)
4. **Gateway Sender**: pyserial USB 전송 코드

### P1 (1주일 내)
5. **구조적 로깅**: Python logging 모듈 도입
6. **Temporal Smoothing**: EMA + majority vote
7. **재연결 강화**: 지수 백오프 무한 재시도
8. **Fatigue 판정기**: 지속적 저각성/우울 상태 감지

### P2 (향후)
9. **서브카메라 비활성화**: config로 토글
10. **오디오 리샘플 캐시**: 동일 구간 재처리 방지
11. **TensorRT 변환**: 추론 속도 최적화
12. **헬스 대시보드**: 센서 상태 실시간 모니터링

---

## 4. 현재 vs 개선 아키텍처 비교

### 현재 (As-Is)
```
sensing_main.py
  ├── 5 daemon threads (하드코딩, 재연결 부족)
  ├── inference_loop() → KMERInferencer.forward()
  ├── 출력: print() (콘솔만)
  └── 게이트웨이: 없음

demo_pipeline.py
  ├── FER-only (KMERFusion 미사용)
  ├── TemporalSmoother (7-frame)
  ├── PacketEncoder (패킷 생성만)
  └── 게이트웨이: 없음
```

### 개선 (To-Be)
```
sensing_main_shadow.py
  ├── Config 기반 센서 스레드 (YAML, degraded mode)
  ├── 기존 inference_loop() [보존, 비교용]
  ├── shadow E2E Pipeline [병렬 실행]
  │   ├── KMERInferencer.forward() (원본 그대로)
  │   ├── MultimodalTemporalSmoother (EMA + majority vote)
  │   ├── FatigueTracker (지속 상태 판정)
  │   ├── 10-class 매핑
  │   ├── PacketEncoder.encode() (기존 재사용)
  │   └── GatewaySender (pyserial USB 전송)
  ├── 출력: logging (콘솔 + 파일)
  └── 검증: old vs new 로그 비교
```

---

## 5. 코드 품질 평가

### 5.1 잘 된 부분
- **KMERInferencer**: 14-token 특성 구성이 체계적, valid_mask로 missing modality 처리
- **버퍼 설계**: `Latest`/`Ring1D`/`BioQueues` thread-safe 설계 적절
- **BLE 패치**: `watch.py`의 `_patched_ble_open`이 Jetson cdc_acm 문제 해결
- **PacketEncoder**: CRC8, 상태 플래그, 인코딩/디코딩 양방향 완성

### 5.2 개선 필요
- **에러 처리**: try-except 후 pass 또는 print만 → 무시되는 에러 다수
- **타입 힌트**: 일부 함수에만 적용, 일관성 부족
- **테스트 코드**: `test_sensors.py` 외 단위 테스트 없음
- **문서화**: 코드 내 docstring은 있으나 시스템 레벨 문서 부족

---

## 6. 결론

현재 센싱 코드는 **기능적으로는 동작**하지만, 실증 배포를 위해서는 다음이 필수:

1. **Config 외부화** → 하드웨어 교체 대응
2. **Degraded mode** → 센서 장애 시 연속 운행
3. **E2E Pipeline** → KMERFusion 결과를 차량 ECU까지 전달
4. **Shadow mode 검증** → 기존 동작 보존 확인 후 교체

**핵심 원칙**: 기존 코드의 실행 경로를 먼저 보존하고, wrapper 방식으로 개선한다.
