# K-MER 파이프라인 ↔ 코드 매핑
> 각 Stage가 어떤 파일의 어떤 함수/클래스에서 실행되는지 정리

---

## 전체 파이프라인 흐름과 코드 위치

```
Stage 0  센서 수집       → realsense.py, rode.py, watch.py
Stage 1  버퍼 관리       → sensing_main.py (Latest, Ring1D, BioQueues, ModelInputs)
Stage 2  Expert 추출     → kmer_inferencer.py → experts/*.py
Stage 3  토큰 구성       → kmer_inferencer.py (_build_feature_dict)
Stage 4  KMERFusion      → kmer_fusion.py (KMERFusion.forward)
Stage 5  복합감정 매핑    → compound_emotion.py (CompoundEmotionMapper)
Stage 6  Smoothing       → pipeline/temporal_smoother.py
Stage 7  10-Class 판정   → pipeline/e2e_pipeline.py (_compute_ten_class)
Stage 8  Fatigue 판정    → pipeline/fatigue_tracker.py
Stage 9  패킷 생성       → gateway/packet_encoder.py (PacketEncoder)
Stage 10 전송            → pipeline/gateway_sender.py (GatewaySender)
```

---

## Stage 0: 센서 데이터 수집

### 카메라

```
파일: sensing/realsense.py (원본 코드 그대로 import)
     raw_sensing_code/.../sensing/realsense.py

함수: run_realsense(
        shutdown_event,
        device_serial="021222070391",
        is_main_cam=True,
        on_frame=callback,        ← 프레임마다 호출되는 콜백
        width=1280, height=720, fps=30
      )

실행 위치: sensing_main_shadow.py에서 스레드로 실행
  threading.Thread(target=run_realsense, kwargs={...}, name="RS_MAIN")

콜백 동작:
  def on_frame_main(ts_ms, frame_bgr):
      mi.frame_main.set(ts_ms, frame_bgr)
      # Latest 버퍼에 최신 프레임 1장 저장
      # 이전 프레임은 덮어씀 (30fps 중 20fps 자연 드롭)

설정값 출처: config/sensing_config.yaml → hardware.realsense_main
```

### 마이크

```
파일: sensing/rode.py (원본 코드 그대로 import)
     raw_sensing_code/.../sensing/rode.py

함수: run_rode(
        shutdown_event,
        on_audio_chunk=callback,  ← 오디오 블록마다 호출되는 콜백
        sample_rate=48000,
        blocksize=8192,
        latency=0.5
      )

내부 동작:
  1. pick_input_device(keywords=["Wireless GO II", "RØDE", ...])
     → sounddevice 디바이스 목록에서 RODE 마이크 자동 검색
  2. sd.InputStream(callback=_audio_callback, ...)
     → 8192 samples씩 콜백 호출 (약 0.17초 간격)
  3. _audio_callback 내부:
     → 스테레오면 모노로 변환: mono = np.mean(data, axis=1)
     → on_audio_chunk(ts_ms, mono, sr) 호출

콜백 동작:
  def on_audio_chunk(ts_ms, mono, sr):
      mi.audio.push(ts_ms, mono)
      # Ring1D 버퍼에 push (최대 96,000 samples = 2초)

설정값 출처: config/sensing_config.yaml → hardware.rode
```

### 워치

```
파일: sensing/watch.py (원본 코드 그대로 import)
     raw_sensing_code/.../sensing/watch.py

함수: run_watch(
        shutdown_event,
        on_ppg=callback,    ← PPG 데이터마다 호출
        on_eda=callback,    ← EDA 데이터마다 호출
        on_temp=callback    ← 온도 데이터마다 호출
      )

내부 동작:
  1. _patched_ble_open()  ← BLE 시리얼 리눅스 호환 monkey-patch
  2. hid.enumerate(vid=0x0456, pid=0x2CFE) → BLE 동글 검색
  3. sdk.connect(mac="F1-18-1C-93-7C-42") → 워치 BLE 연결
  4. ADPD 설정 → PPG 스트리밍 시작
  5. EDA 설정 → EDA 스트리밍 시작
  6. Temperature 설정 → 온도 스트리밍 시작
  7. 각 콜백이 데이터 수신 시 호출됨

콜백 동작:
  def on_ppg(ts_ms, d1, d2):
      mi.bio.push_ppg(ts_ms, d1, d2)
  def on_eda(ts_ms, real):
      mi.bio.push_eda(ts_ms, real)
  def on_temp(ts_ms, skin_c):
      mi.bio.push_temp(ts_ms, skin_c)
  # BioQueues에 push (deque, 최대 512개)

설정값 출처: config/sensing_config.yaml → hardware.watch
```

---

## Stage 1: 공유 버퍼

```
파일: raw_sensing_code/.../sensing/sensing_main.py
     → sensing/core/buffers.py에서 re-export

클래스들:

  class Latest:
    .set(ts, value)     ← 센서 콜백이 호출 (lock 보호)
    .get() → (ts, value) ← 추론 루프가 호출

  class Ring1D:
    .push(ts, x)        ← 오디오 콜백이 호출 (lock 보호)
    .snapshot() → (ts, buf) ← 추론 루프가 호출
    내부: maxlen=96000, circular buffer, filled flag

  class BioQueues:
    .push_ppg(ts, d1, d2)  ← 워치 콜백이 호출
    .push_eda(ts, real)
    .push_temp(ts, skin_c)
    .snapshot() → (ppg_list, eda_list, temp_list)
    내부: deque(maxlen=512) × 3

  class ModelInputs:
    .frame_main = Latest()            ← 카메라
    .audio = Ring1D(maxlen=96000)      ← 마이크
    .bio = BioQueues(maxlen=512)       ← 워치

추론 루프 진입:
  파일: sensing_main_shadow.py → shadow_inference_loop()
  또는: pipeline/e2e_pipeline.py → E2EPipeline.process_cycle()

  매 0.1초마다 (10Hz):
    ts, frame = mi.frame_main.get()
    _, audio_buf = mi.audio.snapshot()
    ppg, eda, temp = mi.bio.snapshot()
    result = pipeline.process_cycle(frame, audio_buf, ppg, eda, temp)
```

---

## Stage 2: Expert Feature 추출

```
파일: raw_sensing_code/.../sensing/kmer_inferencer.py
클래스: KMERInferencer
메서드: forward(frame_bgr, audio_1d, ppg, eda, temp)

내부적으로 3개 처리 함수를 순차 호출:
```

### 2-1. 얼굴 처리

```
메서드: KMERInferencer._process_face(frame_bgr)

  Step A: 얼굴 검출
    클래스: KMERInferencer._FaceDetector (내부 클래스)
    라이브러리: mediapipe.solutions.face_mesh
    메서드: .detect(frame_bgr)
    코드 위치: kmer_inferencer.py 내 _FaceDetector 클래스

    입력: BGR (720, 1280, 3)
    처리: 짧은변 800px 리사이즈 → FaceMesh → 468 랜드마크
          → bbox (20% padding) → 224×224 crop → RGB 정규화
          → AU 좌표 8개 계산 (이마,왼눈,오른눈,코,왼볼,오른볼,입,턱)
    출력: face_chw (3,224,224), au_coords (8,2), bbox

  Step B: K-FER 감정 인식
    파일: raw_sensing_code/.../sensing/experts/kfer_expert.py
    클래스: KFERExpert
    메서드: .extract(face_chw, au_coords)
    모델: AUFERModel (체크포인트: emotion_system/result/best.pth)

    입력: face_chw (3,224,224), au_coords (8,2)
    출력: kfer_probs (7,)    → [angry, anxious, happy, hurt, neutral, sad, surprised]
          kfer_meta (2,)     → [quality=max(probs), entropy]
          face_stats (3,)    → [max_conf, mean_conf, std_conf]

  Step C: PERCLOS + FACS
    파일: raw_sensing_code/.../sensing/experts/facs_aux.py
    클래스: FACSAuxExpert
    메서드: .extract(face_chw, au_coords)

    입력: face_chw (3,224,224), au_coords (8,2)
    출력: perclos_ear (2,)   → [PERCLOS, mean_EAR]
          facs_scores (6,)   → 6개 얼굴 영역 AU 강도
```

### 2-2. 오디오 처리

```
메서드: KMERInferencer._process_audio(audio_1d)

  Step A: Resample
    라이브러리: librosa.resample
    48kHz → 16kHz (96,000 → 32,000 samples)

  Step B: Audio Quality 계산
    코드 위치: kmer_inferencer.py 내 _process_audio 메서드
    직접 계산: RMS, Zero Crossing Rate, SNR
    출력: audio_quality (3,)

  Step C: 오디오 감정 인식
    파일: raw_sensing_code/.../sensing/experts/audio_expert.py
    클래스: Emotion2VecExpert
    메서드: .extract(audio_16k)
    모델: FunASR emotion2vec

    입력: audio_16k (32,000 samples)
    출력: emo2vec_probs (9,) → 9-class 오디오 감정 확률

  Step D: Arousal/Valence/Dominance
    파일: raw_sensing_code/.../sensing/experts/audio_expert.py
    클래스: AudeeringExpert
    메서드: .extract(audio_16k)
    모델: wav2vec2 기반 audeering

    입력: audio_16k (32,000 samples)
    출력: audeering_avd (3,) → [arousal, valence, dominance]
```

### 2-3. 생체신호 처리

```
메서드: KMERInferencer._process_bio(ppg, eda, temp)

  Step A: NPZ 변환
    메서드: _build_bio_npz()
    BioQueues 형식 → numpy array 변환

  Step B: Feature 추출
    파일: raw_sensing_code/.../sensing/experts/bio_expert.py
    함수: extract_bio_features_v2(bio_npz)
    라이브러리: neurokit2, scipy

    입력: PPG/EDA/Temp numpy arrays
    출력: bvp_features (4,)      → [mean_hr, sdnn, rmssd, lf_hf]
          eda_features (5,)      → [mean_scl, std_scl, n_peaks, amplitude, auc]
          hr_temp_features (6,)  → [hr_mean, hr_std, hr_range, temp_mean, temp_slope, temp_range]
          bio_quality (3,)       → 품질 지표
```

---

## Stage 3: 토큰 구성

```
파일: raw_sensing_code/.../sensing/kmer_inferencer.py
메서드: KMERInferencer._build_feature_dict(face_out, audio_out, bio_out)

  Step A: Cross-Modal Feature 계산
    cross_modal (3,) = [face_audio_agree, av_consistency, entropy_gap]
    코드 위치: _build_feature_dict 내부

  Step B: Validity Flags
    validity_flags (3,) = [face_valid, audio_valid, bio_valid]

  Step C: Valid Mask 생성
    파일: multimodal_dms/fusion/kmer_fusion.py
    함수: build_valid_mask(face_valid, audio_valid, bio_valid)
    출력: valid_mask (1, 15) boolean tensor

  Step D: Tensor 변환 + GPU 이동
    모든 feature → torch.tensor → .cuda()
    features dict + valid_mask → KMERFusion 입력 준비 완료

최종 features dict (14개):
  "kfer_probs"       (1,7)  │ "emo2vec_probs"    (1,9)  │ "bvp_features"     (1,4)
  "kfer_meta"        (1,2)  │ "audeering_avd"    (1,3)  │ "eda_features"     (1,5)
  "face_stats"       (1,3)  │ "audio_quality"    (1,3)  │ "hr_temp_features" (1,6)
  "perclos_ear"      (1,2)  │                           │ "bio_quality"      (1,3)
  "facs_scores"      (1,6)  │                           │
                            │ "cross_modal"      (1,3)  │ "validity_flags"   (1,3)
```

---

## Stage 4: KMERFusion 신경망

```
파일: multimodal_dms/fusion/kmer_fusion.py
클래스: KMERFusion
체크포인트: multimodal_dms/results_kmer/best_model.pth
메서드: forward(features, valid_mask)

  Layer 1: Token Formation
    코드: self.projectors (nn.ModuleDict, 14개 Linear)
    각 feature → Linear(input_dim, 64) → 64d token
    + self.cls_token (nn.Parameter, learnable 64d)
    → tokens (1, 15, 64)

  Layer 2: Modality Type Embedding
    코드: self.modality_embed (nn.Embedding(5, 64))
    Face(0), Audio(1), Bio(2), Aux(3), Meta(4)
    tokens += modality_embed
    → tokens (1, 15, 64)

  Layer 3: Intra-Modal Pool-FFN
    코드: self.face_pool_ffn, self.audio_pool_ffn, self.bio_pool_ffn
    같은 모달리티 토큰끼리 AvgPool → FFN → 업데이트
    Face: T1-T3 (3개), Audio: T4-T6 (3개), Bio: T7-T10 (4개)
    → tokens (1, 15, 64)

  Layer 4: Global MHSA
    코드: self.mhsa_norm (LayerNorm) + self.mhsa (MultiheadAttention)
          + self.ffn_norm (LayerNorm) + self.ffn (Sequential)
    MultiheadAttention(d_model=64, nhead=4)
    key_padding_mask = ~valid_mask (누락 센서 토큰 무시)
    + FFN: Linear(64→256) → GELU → Dropout → Linear(256→64)
    → tokens (1, 15, 64)

  Layer 5: CLS Pooling
    코드: cls_out = tokens[:, -1, :]
    → fused_repr (1, 64)

  Layer 6: Output Heads
    코드: self.arousal_head, self.valence_head, self.drowsy_head
    arousal: Linear(64→32)→ReLU→Dropout→Linear(32→1)→Sigmoid → (0~1)
    valence: Linear(64→32)→ReLU→Dropout→Linear(32→1)→Sigmoid → (0~1)
    drowsy:  concat(fused.detach(), perclos_ear) → Linear(66→32)→ReLU→Dropout→Linear(32→3) → argmax → (0/1/2)

출력:
  arousal = 0.45
  valence = 0.62
  drowsy_logits = [-1.2, 0.3, -0.8] → argmax → drowsy = 0
```

---

## Stage 5: CompoundEmotionMapper

```
파일: multimodal_dms/fusion/compound_emotion.py
클래스: CompoundEmotionMapper
메서드: map_single(kfer_top1_id, arousal_pred, is_drowsy)

  입력:
    kfer_top1_id = argmax(kfer_probs) = 2 (happy)
    arousal_pred = 0.45
    is_drowsy = (drowsy >= 1) = False

  내부 로직:
    1. is_drowsy 체크 → True면 즉시 (12, "drowsy") 반환
    2. arousal 이산화: 0.45 → "mid" (0.33~0.66)
    3. 규칙 테이블 참조: happy + mid → (3, "positive_engaged")

  코드 위치: map_single 메서드 내 self._rules dict

출력:
  compound_id = 3
  compound_label = "positive_engaged"
```

---

## Stage 6: forward() 반환 → E2E Pipeline 진입

```
반환 위치: kmer_inferencer.py → KMERInferencer.forward() 마지막

  result = {
      "arousal": 0.45, "valence": 0.62, "drowsy": 0,
      "compound_id": 3, "compound_label": "positive_engaged",
      "kfer_emotion": "happy", "kfer_confidence": 0.85,
      "face_detected": True,
  }

수신 위치: pipeline/e2e_pipeline.py → E2EPipeline.process_cycle()
  Line: raw_result = self._inferencer.forward(frame, audio, ppg, eda, temp)
```

---

## Stage 7: Temporal Smoothing

```
파일: sensing/pipeline/temporal_smoother.py
클래스: MultimodalTemporalSmoother
메서드: smooth(raw_result)
호출 위치: e2e_pipeline.py → process_cycle() 내부
  Line: smoothed = self.smoother.smooth(raw_result)

내부 동작:

  self.emotion_smoother (MajorityVoteSmoother, window=7):
    .push("happy") → 최근 7개 중 다수결 → "happy"
    result에 추가: smoothed_kfer_emotion = "happy"
                   smoothed_emotion_confidence = 0.857

  self.arousal_smoother (EMASmoother, alpha=0.3):
    .push(0.45) → 0.3×0.45 + 0.7×prev → 0.485
    result에 추가: smoothed_arousal = 0.485

  self.valence_smoother (EMASmoother, alpha=0.3):
    .push(0.62) → 0.3×0.62 + 0.7×prev → 0.606
    result에 추가: smoothed_valence = 0.606

  self.drowsy_smoother (MajorityVoteSmoother, window=5):
    .push(0) → 최근 5개 중 다수결 → 0
    result에 추가: smoothed_drowsy = 0

  self.compound_smoother (MajorityVoteSmoother, window=7):
    .push(3) → 최근 7개 중 다수결 → 3
    result에 추가: smoothed_compound_id = 3
```

---

## Stage 8: 10-Class 판정

```
파일: sensing/pipeline/e2e_pipeline.py
클래스: E2EPipeline
메서드: _compute_ten_class(smoothed)
호출 위치: process_cycle() 내부
  Line: ten_class = self._compute_ten_class(smoothed)

═══ Emotion 6개 (lines 297~306) ═══

  코드:
    kfer_emotion = smoothed["smoothed_kfer_emotion"]        # "happy"
    kfer_id = kfer_label_to_id[kfer_emotion]                # 2
    emotion_code = self._kfer_to_protocol[kfer_id]          # 4
    emotion_name = self._protocol_names[emotion_code]       # "행복"

  매핑 테이블 (packet_encoder.py에서 import):
    KFER_TO_PROTOCOL = {0:2, 1:0, 2:4, 3:3, 4:5, 5:3, 6:1}
    PROTOCOL_NAMES = {0:"공포", 1:"놀람", 2:"분노", 3:"슬픔/혐오", 4:"행복", 5:"중립"}

═══ Stress (lines 308~317) ═══

  코드:
    arousal = smoothed["smoothed_arousal"]                  # 0.485
    is_negative = kfer_id in {0, 1}                         # False (happy=2)
    if is_negative and arousal > config.thresholds.arousal_stress:  # 0.6
        stress = True
    else:
        stress = False

═══ Low Attention (lines 319~331) ═══

  코드:
    perclos = raw_face_out.get("perclos_ear", [0.0])[0]    # 0.15
    low_attention = (0.2 <= perclos < 0.4)                  # False

  설정값: config.thresholds.perclos_low_attn = 0.2
          config.thresholds.perclos_drowsy = 0.4

═══ Drowsy (lines 333~336) ═══

  코드:
    drowsy_val = smoothed["smoothed_drowsy"]                # 0
    drowsy = (drowsy_val >= 1) or (perclos >= 0.4)          # False

═══ Fatigue (별도 Stage) ═══

  fatigue = False  ← placeholder, Stage 9에서 설정
```

---

## Stage 9: Fatigue 판정

```
파일: sensing/pipeline/fatigue_tracker.py
클래스: FatigueTracker
메서드: update(compound_label, arousal, drowsy)
호출 위치: e2e_pipeline.py → process_cycle() 내부
  Line: fatigue = self.fatigue_tracker.update(
            compound_label=smoothed.get("compound_label"),
            arousal=smoothed.get("smoothed_arousal"),
            drowsy=smoothed.get("smoothed_drowsy"),
        )
        ten_class["fatigue"] = fatigue

내부 카운터 3개:

  compound_counter (조건1):
    if compound_label in {"depressed", "calm"}:
        compound_counter += 1
    else:
        compound_counter = 0
    if compound_counter >= duration_sec × hz:    # 30 × 10 = 300
        fatigue = True

  arousal_counter (조건2):
    if arousal is not None and arousal < arousal_thresh:  # 0.3
        arousal_counter += 1
    elif arousal is None:
        pass  # 동결 (증가도 리셋도 안 함)
    else:
        arousal_counter = 0
    if arousal_counter >= duration_sec × hz:     # 300
        fatigue = True

  drowsy_counter (조건3):
    if drowsy >= 1:
        drowsy_counter += 1
    else:
        drowsy_counter = 0
    if drowsy_counter >= drowsy_sec × hz:        # 10 × 10 = 100
        fatigue = True

설정값 출처: config/sensing_config.yaml → thresholds
  fatigue_duration_sec: 30
  fatigue_arousal: 0.3
  fatigue_drowsy_sec: 10
```

---

## Stage 10: 패킷 인코딩

```
파일: multimodal_dms/gateway/packet_encoder.py
클래스: PacketEncoder
메서드: encode(kfer_emotion_id, emotion_confidence, arousal, perclos)
호출 위치: e2e_pipeline.py → _encode_packet()
  Line: packet = self._encoder.encode(
            kfer_emotion_id=kfer_id,
            emotion_confidence=confidence,
            arousal=arousal,
            perclos=perclos,
        )

내부 동작:

  1. Emotion Code: KFER_TO_PROTOCOL[kfer_id]           → 4bit (Byte4 상위)
  2. StressDetector.detect(kfer_id, arousal)            → 1bit (Byte4 bit3)
  3. AttentionDetector.detect(perclos)                  → 1bit (Byte4 bit2)
  4. DrowsyDetector.detect(perclos)                     → 1bit (Byte4 bit1)
  5. NegativeEmotionDetector.detect(kfer_id)            → 1bit (Byte5 bit1)
  6. quantize_intensity(confidence)                     → 3bit (Byte5 상위)
  7. quantize_intensity(arousal)                        → 3bit (Byte5 중간)
  8. compute_crc8(Byte1~Byte5)                          → Byte6

Fatigue bit 추가 (e2e_pipeline.py → _add_fatigue_bit):
  if fatigue:
      packet[5] |= 0x01       # Byte5 bit0 = 1
      packet[6] = compute_crc8(packet[1:6])  # CRC 재계산

최종 패킷:
  [0xAA, 0x01, SEQ, 0x02, Byte4, Byte5, CRC, 0xFE]
```

---

## Stage 11: Gateway 전송

```
파일: sensing/pipeline/gateway_sender.py
클래스: GatewaySender
메서드: send(packet)
호출 위치: e2e_pipeline.py → process_cycle() 내부
  Line: self.sender.send(packet_info["packet"])

내부 동작:
  self._serial.write(packet)    # pyserial
  self._serial.flush()

  실패 시:
    self._connected = False
    5초마다 self.connect() 재시도
    크래시 없음 (에러 로그만)

NullGatewaySender:
  config.gateway.enabled = false일 때 사용
  send()가 아무것도 안 함 (테스트/개발용)

설정값 출처: config/sensing_config.yaml → gateway
  port: "/dev/ttyUSB0"
  baudrate: 115200
```

---

## 오케스트레이터: process_cycle() 전체 흐름

```
파일: sensing/pipeline/e2e_pipeline.py
클래스: E2EPipeline
메서드: process_cycle(frame, audio, ppg, eda, temp)

  Line ~227: self._lazy_init()
             → 첫 호출 시 KMERInferencer + PacketEncoder 로드

  Line ~231: mode = detect_mode(frame, audio, ppg, eda, temp)
             → FULL / CAM+MIC / CAM+BIO / CAM_ONLY / NO_CAM

  Line ~240: if mode == NO_CAM: return fallback
             → emotion=neutral, 모든 state=False

  Line ~244: raw_result = self._inferencer.forward(...)    ← Stage 2~6
             → arousal, valence, drowsy, compound, kfer_emotion

  Line ~257: smoothed = self.smoother.smooth(raw_result)   ← Stage 7
             → smoothed_arousal, smoothed_kfer_emotion, ...

  Line ~260: ten_class = self._compute_ten_class(smoothed) ← Stage 8
             → emotion_code, stress, low_attention, drowsy

  Line ~263: fatigue = self.fatigue_tracker.update(...)     ← Stage 9
             → fatigue True/False

  Line ~271: packet_info = self._encode_packet(...)        ← Stage 10
             → 8-byte packet + fatigue bit

  Line ~275: self.sender.send(packet)                      ← Stage 11
             → USB serial 전송

  Line ~278: result = {} (모든 결과 합성)
             result.update(raw_result)      # KMERFusion 원본
             result.update(smoothed)        # Smoothed 값
             result.update(ten_class)       # 10-class 판정
             result.update(packet_info)     # 패킷 정보
             result["mode"] = mode
             return result
```

---

## 진입점 파일

```
파일: sensing/sensing_main_shadow.py
역할: 전체 시스템 실행 진입점

  1. config 로드: load_config()
  2. ModelInputs 생성
  3. 센서 스레드 3개 시작 (realsense, rode, watch)
  4. 기존 inference_loop 스레드 시작 (원본 동작 보존)
  5. shadow_inference_loop 스레드 시작 (E2E Pipeline)
  6. Ctrl+C → shutdown.set() → 모든 스레드 종료

  shadow_inference_loop:
    pipeline = E2EPipeline(config)
    while not shutdown.is_set():
        frame, audio, ppg, eda, temp = snapshot(mi)
        result = pipeline.process_cycle(frame, audio, ppg, eda, temp)
        _log_shadow_result(result)  # 구조적 로그 출력
        sleep(1/hz)

파일: sensing/test_e2e_verification.py
역할: 하드웨어 없이 소프트웨어 로직 검증 (42개 테스트)

파일: sensing/test_sensors.py
역할: 개별 센서 연결 테스트
```

---

## 설정 파일

```
파일: sensing/config/sensing_config.yaml
역할: 모든 하드코딩 값을 한 곳에서 관리

로더: sensing/config/__init__.py → load_config()
     → SensingConfig dataclass 반환

사용처:
  realsense.py  ← hardware.realsense_main.serial/width/height/fps
  rode.py       ← hardware.rode.sample_rate/blocksize/keywords
  watch.py      ← hardware.watch.vid/pid/mac
  kmer_infer    ← inference.kfer_ckpt/kmer_ckpt/device/hz
  e2e_pipeline  ← thresholds.arousal_stress/perclos_drowsy/...
  fatigue       ← thresholds.fatigue_duration_sec/fatigue_arousal
  gateway       ← gateway.port/baudrate/enabled
  smoother      ← temporal_smoothing.emotion_window/ema_alpha
```
