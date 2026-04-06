# K-MER 10-Class 출력/상태 판정 로직 명세서

> 버전: 1.0
> 작성일: 2026-03-09

---

## 1. 개요

K-MER 시스템의 최종 출력은 **10개 항목**으로 구성된다:
- **감정 (Emotion)**: 6개 — Fear, Surprise, Anger, Unpleasant, Happy, Neutral
- **운전자 상태 (Driver State)**: 4개 — Stress, Low Attention, Drowsy, Fatigue

각 항목에 대해 **입력 source**, **계산 방식**, **threshold**, **fallback 규칙**, **sensor missing 시 대체 로직**을 정의한다.

---

## 2. 감정 판정 (Emotion, 6개)

### 공통 사항
- **입력 Source**: K-FER 모델 (AUFERModel, MobileViTv2 backbone)
- **계산 방식**: 7-class softmax → top-1 argmax → `KFER_TO_PROTOCOL` 매핑
- **Temporal Smoothing**: majority vote (window=7)
- **Sensor Missing**: 카메라 없으면 전부 Neutral (code=6)

### 2.1 Fear (공포) — Protocol Code 0

| 항목 | 값 |
|------|---|
| **입력** | K-FER class index 1 (anxious) |
| **매핑** | `KFER_TO_PROTOCOL[1] = 0` |
| **판정** | K-FER softmax top-1 == anxious |
| **Confidence** | softmax[1] (해당 클래스 확률) |
| **Smoothing** | 7-frame majority vote |
| **Fallback** | top-1이 아니면 해당 없음 |
| **Cam missing** | Neutral (code=6) |

### 2.2 Surprise (놀람) — Protocol Code 1

| 항목 | 값 |
|------|---|
| **입력** | K-FER class index 6 (surprised) |
| **매핑** | `KFER_TO_PROTOCOL[6] = 1` |
| **판정** | K-FER softmax top-1 == surprised |
| **Cam missing** | Neutral (code=6) |

### 2.3 Anger (분노) — Protocol Code 2

| 항목 | 값 |
|------|---|
| **입력** | K-FER class index 0 (angry) |
| **매핑** | `KFER_TO_PROTOCOL[0] = 2` |
| **판정** | K-FER softmax top-1 == angry |
| **Cam missing** | Neutral (code=6) |

### 2.4 Unpleasant (불쾌) — Protocol Code 3 또는 5

| 항목 | 값 |
|------|---|
| **입력** | K-FER class index 5 (sad) 또는 3 (hurt) |
| **매핑** | `KFER_TO_PROTOCOL[5] = 3` (슬픔), `KFER_TO_PROTOCOL[3] = 5` (혐오) |
| **판정** | K-FER softmax top-1 == sad 또는 hurt |
| **통합 규칙** | sad와 hurt 모두 "Unpleasant"로 표시하되, protocol code는 원래 값 유지 |
| **Cam missing** | Neutral (code=6) |

### 2.5 Happy (행복) — Protocol Code 4

| 항목 | 값 |
|------|---|
| **입력** | K-FER class index 2 (happy) |
| **매핑** | `KFER_TO_PROTOCOL[2] = 4` |
| **판정** | K-FER softmax top-1 == happy |
| **Cam missing** | Neutral (code=6) |

### 2.6 Neutral (중립) — Protocol Code 6

| 항목 | 값 |
|------|---|
| **입력** | K-FER class index 4 (neutral) |
| **매핑** | `KFER_TO_PROTOCOL[4] = 6` |
| **판정** | K-FER softmax top-1 == neutral, 또는 모든 fallback의 기본값 |
| **Cam missing** | Neutral (항상 가능) |

---

## 3. 운전자 상태 판정 (Driver State, 4개)

### 3.1 Stress — Byte4 bit3

| 항목 | 값 |
|------|---|
| **입력 Source** | arousal (KMERFusion 출력) + K-FER emotion |
| **계산 방식** | `StressDetector.detect()` |
| **조건** | arousal > 0.6 AND emotion ∈ {angry(0), anxious(1)} |
| **Threshold** | `arousal_threshold = 0.6` |
| **Fallback** | False |
| **Bio missing** | arousal = None → emotion만으로 판정 (negative emotion이면 True) |
| **Cam missing** | emotion 판정 불가 → False |
| **Smoothing** | arousal에 EMA(alpha=0.3) 적용 후 판정 |

**Stress 판정 로직 (의사 코드)**:
```python
def detect_stress(kfer_emotion_id, arousal):
    is_negative = kfer_emotion_id in {0, 1}  # angry, anxious
    if not is_negative:
        return False
    if arousal is None:
        return True  # bio없이 emotion만으로 stress 판정
    return arousal > 0.6
```

### 3.2 Low Attention — Byte4 bit2

| 항목 | 값 |
|------|---|
| **입력 Source** | PERCLOS (FACSAuxExpert 출력) |
| **계산 방식** | `AttentionDetector.detect()` |
| **조건** | 0.2 ≤ PERCLOS < 0.4 |
| **Threshold** | `perclos_low = 0.2`, `perclos_high = 0.4` |
| **Fallback** | False |
| **Cam missing** | PERCLOS 계산 불가 → False |
| **Smoothing** | PERCLOS 자체가 EAR의 시간 평균이므로 추가 smoothing 불필요 |

### 3.3 Drowsy — Byte4 bit1

| 항목 | 값 |
|------|---|
| **입력 Source** | PERCLOS (FACSAuxExpert) + KMERFusion drowsy head |
| **계산 방식** | `DrowsyDetector.detect()` 또는 KMERFusion drowsy head argmax |
| **조건 A (PERCLOS)** | PERCLOS ≥ 0.4 |
| **조건 B (모델)** | KMERFusion drowsy_head argmax ≥ 1 |
| **최종 판정** | 조건 A OR 조건 B |
| **Threshold** | `perclos_drowsy = 0.4` |
| **Fallback** | False |
| **Cam missing** | PERCLOS 불가 + drowsy head도 face 필요 → False |
| **Smoothing** | majority vote (window=5) |

### 3.4 Fatigue — Byte5 bit0 (신규)

| 항목 | 값 |
|------|---|
| **입력 Source** | compound_label (CompoundEmotionMapper) + arousal (KMERFusion) + drowsy |
| **계산 방식** | `FatigueTracker.update()` |
| **조건 1** | compound_label ∈ {"depressed", "calm"} 이 **30초 이상** 연속 |
| **조건 2** | arousal < 0.3 이 **30초 이상** 연속 |
| **조건 3** | drowsy 상태가 **10초 이상** 연속 |
| **최종 판정** | 조건 1 OR 조건 2 OR 조건 3 |
| **Threshold** | `fatigue_arousal = 0.3`, `fatigue_duration_sec = 30`, `fatigue_drowsy_sec = 10` |
| **Fallback** | False |
| **Bio missing** | 조건 2 불가 → 조건 1, 3만으로 판정 |
| **Cam missing** | 조건 1 (compound 불가), 조건 3 (drowsy 불가) → False |
| **Smoothing** | 시간 기반 (duration 자체가 smoothing) |

**Fatigue 판정 로직 (의사 코드)**:
```python
class FatigueTracker:
    def __init__(self, duration_sec=30, arousal_thresh=0.3, drowsy_sec=10, hz=10):
        self.duration_frames = int(duration_sec * hz)   # 300 frames
        self.drowsy_frames = int(drowsy_sec * hz)       # 100 frames
        self.arousal_thresh = arousal_thresh
        self.compound_counter = 0
        self.arousal_counter = 0
        self.drowsy_counter = 0

    def update(self, compound_label, arousal, drowsy):
        # 조건 1: depressed/calm 연속
        if compound_label in ("depressed", "calm"):
            self.compound_counter += 1
        else:
            self.compound_counter = 0

        # 조건 2: 저각성 연속
        if arousal is not None and arousal < self.arousal_thresh:
            self.arousal_counter += 1
        else:
            self.arousal_counter = 0

        # 조건 3: drowsy 연속
        if drowsy >= 1:
            self.drowsy_counter += 1
        else:
            self.drowsy_counter = 0

        return (self.compound_counter >= self.duration_frames or
                self.arousal_counter >= self.duration_frames or
                self.drowsy_counter >= self.drowsy_frames)
```

---

## 4. Degraded Mode별 출력 가용성

| Mode | Emotion 6 | Stress | Low Attn | Drowsy | Fatigue |
|------|-----------|--------|----------|--------|---------|
| **FULL** (Cam+Mic+Bio) | O | O (arousal+emotion) | O | O | O (3조건 모두) |
| **CAM+MIC** (Bio 없음) | O | △ (emotion only) | O | O | △ (조건1,3만) |
| **CAM+BIO** (Mic 없음) | O | O | O | O | O |
| **CAM_ONLY** | O | △ (emotion only) | O | O | △ (조건1,3만) |
| **NO_CAM** | X (neutral) | X | X | X | X |

- O: 정상 동작
- △: 일부 조건 불가, 가용 조건만으로 판정
- X: 판정 불가, fallback 값 사용

---

## 5. Packet Byte 매핑

```
Byte4 (상위 → 하위):
  [3:0] Emotion Code (4bit): 0=Fear, 1=Surprise, 2=Anger, 3=Sadness, 4=Happy, 5=Disgust, 6=Neutral
  [3]   Stress flag
  [2]   Low Attention flag
  [1]   Drowsy flag
  [0]   END_FLAG

Byte5 (상위 → 하위):
  [7:5] Emotion Intensity (3bit, 0~7)
  [4:2] State Intensity (3bit, 0~7)
  [1]   Reserved
  [0]   Fatigue flag (신규)
```

---

## 6. Temporal Smoothing 정리

| 출력 | 타입 | Smoothing 방식 | Window/Alpha |
|------|------|---------------|-------------|
| emotion_code | 이산 (0~6) | majority vote | window=7 |
| arousal | 연속 [0,1] | EMA | alpha=0.3 |
| valence | 연속 [0,1] | EMA | alpha=0.3 |
| drowsy | 이산 (0,1,2) | majority vote | window=5 |
| compound_label | 이산 (0~12) | majority vote | window=7 |
| stress | bool | emotion smoothing 후 판정 | - |
| low_attention | bool | PERCLOS (자체 시간 평균) | - |
| fatigue | bool | duration counter | 30초/10초 |
