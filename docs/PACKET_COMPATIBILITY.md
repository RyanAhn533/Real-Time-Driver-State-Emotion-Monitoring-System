# PacketEncoder 호환성 분석 문서

> 버전: 1.0
> 작성일: 2026-03-09
> 대상: `multimodal_dms/gateway/packet_encoder.py`

---

## 1. 기존 8-Byte 패킷 구조

```
Byte0: SOF = 0xAA (Start Of Frame)
Byte1: TYPE = 1 (감정/상태 패킷)
Byte2: SEQ = 0~255 (순번, 자동 증가)
Byte3: LEN = 2 (Payload 길이: Byte4 + Byte5)
Byte4: [Emotion Code (4bit)] [Stress (1)] [LowAttn (1)] [Drowsy (1)] [END_FLAG (1)]
Byte5: [Emotion Intensity (3bit)] [State Intensity (3bit)] [Reserved (2bit)]
Byte6: CRC8 (Byte1~Byte5 기반, polynomial=0x07)
Byte7: EOF = 0xFE (End Of Frame)
```

### Byte4 비트 레이아웃
```
bit7 bit6 bit5 bit4 | bit3    | bit2      | bit1   | bit0
─────────────────── | ─────── | ───────── | ────── | ──────
 Emotion Code (4b)  | Stress  | Low Attn  | Drowsy | END_FLAG
```

### Byte5 비트 레이아웃
```
bit7 bit6 bit5 | bit4 bit3 bit2 | bit1     | bit0
────────────── | ────────────── | ──────── | ────────
Emo Intensity  | State Intens.  | Reserved | Reserved
   (3bit)      |    (3bit)      |          |
```

---

## 2. 기존 Emotion Code 매핑

### K-FER 7-class → Protocol Code

| K-FER Index | K-FER Label | Protocol Code | Protocol Name |
|-------------|-------------|---------------|---------------|
| 0 | angry | 2 | 분노 |
| 1 | anxious | 0 | 공포 |
| 2 | happy | 4 | 행복 |
| 3 | hurt | 5 | 혐오 |
| 4 | neutral | 6 | 중립 |
| 5 | sad | 3 | 슬픔 |
| 6 | surprised | 1 | 놀람 |

### Protocol Code 범위
- 사용 중: 0~6 (7개)
- 4bit 최대: 0~15 (16개)
- **여유: 7~15 (9개 코드 사용 가능)**

---

## 3. 기존 Status Flag 판정 로직

### StressDetector (Byte4 bit3)
```python
NEGATIVE_EMOTIONS = {0, 1}  # angry, anxious
arousal_threshold = 0.6

def detect(kfer_emotion_id, arousal):
    if kfer_emotion_id in NEGATIVE_EMOTIONS:
        if arousal is None or arousal > arousal_threshold:
            return True
    return False
```

### AttentionDetector (Byte4 bit2)
```python
perclos_low = 0.2
perclos_high = 0.4

def detect(perclos):
    return perclos_low <= perclos < perclos_high
```

### DrowsyDetector (Byte4 bit1)
```python
threshold = 0.4

def detect(perclos):
    return perclos >= threshold
```

---

## 4. 10-Class 확장 호환성 분석

### 4.1 Emotion 6개: 완전 호환

| 10-class | Protocol Code | 기존 매핑 | 호환성 |
|----------|---------------|----------|--------|
| Fear | 0 | anxious → 0 | **호환** |
| Surprise | 1 | surprised → 1 | **호환** |
| Anger | 2 | angry → 2 | **호환** |
| Unpleasant (sad) | 3 | sad → 3 | **호환** |
| Happy | 4 | happy → 4 | **호환** |
| Unpleasant (hurt) | 5 | hurt → 5 | **호환** |
| Neutral | 6 | neutral → 6 | **호환** |

**결론**: Emotion 코드는 기존 `KFER_TO_PROTOCOL` 매핑과 100% 호환. 변경 불필요.

### 4.2 Stress / Low Attention / Drowsy: 완전 호환

- Byte4 bit3 (Stress): 기존 `StressDetector` 그대로 사용
- Byte4 bit2 (Low Attention): 기존 `AttentionDetector` 그대로 사용
- Byte4 bit1 (Drowsy): 기존 `DrowsyDetector` 그대로 사용

**결론**: 3개 status flag는 기존 로직과 100% 호환. 변경 불필요.

### 4.3 Fatigue: 신규 비트 필요

기존 패킷에 Fatigue 전용 비트가 없음. 3가지 옵션 분석:

#### 옵션 A: Byte5 Reserved bit0 사용 (권장)

```
Byte5 변경:
  기존: [Emo Intensity (3)] [State Intensity (3)] [Reserved (2)]
  변경: [Emo Intensity (3)] [State Intensity (3)] [Reserved (1)] [Fatigue (1)]
```

| 항목 | 평가 |
|------|------|
| **장점** | 패킷 크기 변경 없음 (8 bytes 유지) |
| **장점** | Byte4 완전 호환 (기존 decoder 영향 없음) |
| **장점** | CRC8 범위 변경 없음 (Byte1~5) |
| **단점** | Reserved 1bit 소모 |
| **ECU 호환** | 기존 decoder가 Byte5 bit0을 무시하면 무충돌 |

#### 옵션 B: State Intensity로 인코딩

```
State Intensity (3bit, 0~7):
  0~5: 기존 (arousal 강도)
  6: Fatigue 경고
  7: Fatigue 위험
```

| 항목 | 평가 |
|------|------|
| **장점** | Reserved 비트 소모 없음 |
| **단점** | State Intensity의 의미 변경 → 기존 decoder 호환 문제 |
| **ECU 호환** | State Intensity 해석이 달라져서 호환 안 될 수 있음 |

#### 옵션 C: 별도 패킷 TYPE=2 추가

```
새 패킷 TYPE=2 (Driver State 전용):
  Byte4: [Stress(1)] [LowAttn(1)] [Drowsy(1)] [Fatigue(1)] [Reserved(4)]
  Byte5: [각 상태별 intensity]
```

| 항목 | 평가 |
|------|------|
| **장점** | 기존 TYPE=1 패킷 완전 보존 |
| **장점** | 향후 확장 여유 있음 |
| **단점** | 패킷 전송량 2배 (10Hz × 2 packets = 20 packets/s) |
| **ECU 호환** | 기존 decoder가 TYPE=2를 무시하면 무충돌 |

### 4.4 권장 방안: 옵션 A

**이유**:
1. 패킷 크기 유지 (8 bytes) → 대역폭 변경 없음
2. Byte4 완전 보존 → 기존 ECU decoder 영향 최소
3. CRC8 계산 범위 동일 → 검증 로직 변경 없음
4. Reserved bit은 향후 확장을 위해 예약된 것이므로 사용이 적절

---

## 5. PacketEncoder 수정 사항

### 5.1 encode() 함수 변경

```python
# 기존
def encode(self, kfer_emotion_id, emotion_confidence=0.5,
           arousal=None, perclos=0.0, end_flag=False):

# 변경 (fatigue 파라미터 추가)
def encode(self, kfer_emotion_id, emotion_confidence=0.5,
           arousal=None, perclos=0.0, end_flag=False,
           fatigue=False):  # 신규
```

### 5.2 Byte5 인코딩 변경

```python
# 기존
byte5 = (
    ((emo_intensity & 0x07) << 5) |
    ((state_intensity & 0x07) << 2)
)

# 변경
byte5 = (
    ((emo_intensity & 0x07) << 5) |
    ((state_intensity & 0x07) << 2) |
    (int(fatigue) & 0x01)  # bit0 = Fatigue flag
)
```

### 5.3 decode() 함수 변경

```python
# Byte5 디코딩에 fatigue 추가
fatigue = bool(byte5 & 0x01)

return {
    ...
    "fatigue": fatigue,  # 신규
    ...
}
```

---

## 6. 호환성 검증 방법

### 6.1 기존 패킷 비교 테스트

```python
encoder_old = PacketEncoder()  # 기존
encoder_new = PacketEncoderV2()  # 변경

# 동일 입력, Fatigue=False일 때 패킷 동일 확인
for kfer_id in range(7):
    pkt_old = encoder_old.encode(kfer_id, 0.9, 0.5, 0.1)
    pkt_new = encoder_new.encode(kfer_id, 0.9, 0.5, 0.1, fatigue=False)
    assert pkt_old == pkt_new, f"Mismatch at kfer_id={kfer_id}"
```

### 6.2 Fatigue bit 독립성 확인

```python
# Fatigue=True일 때 Byte4는 동일, Byte5 bit0만 차이
pkt_no_fatigue = encoder_new.encode(4, 0.5, 0.3, 0.1, fatigue=False)
pkt_fatigue = encoder_new.encode(4, 0.5, 0.3, 0.1, fatigue=True)

assert pkt_no_fatigue[:5] == pkt_fatigue[:5]  # Byte0~4 동일
assert pkt_no_fatigue[5] | 0x01 == pkt_fatigue[5]  # Byte5 bit0만 차이
# CRC는 Byte5 변경으로 달라짐 (정상)
```

### 6.3 ECU Decoder 역호환 시나리오

| ECU Decoder 동작 | Fatigue bit 영향 | 결과 |
|------------------|-----------------|------|
| Byte5를 전혀 파싱하지 않음 | 영향 없음 | **안전** |
| Byte5 상위 6bit만 파싱 (Emo+State Intensity) | 영향 없음 | **안전** |
| Byte5 전체 8bit 파싱 | bit0 변경 감지됨 | **주의 필요** |
| Reserved bit 0이 아니면 에러 처리 | Fatigue=True 시 에러 | **비호환** |

**권장 조치**: ECU decoder가 Reserved bit을 검증하는지 확인 후 적용

---

## 7. 결론

| 항목 | 호환성 | 비고 |
|------|--------|------|
| Emotion 6개 | **100% 호환** | KFER_TO_PROTOCOL 그대로 |
| Stress | **100% 호환** | StressDetector 그대로 |
| Low Attention | **100% 호환** | AttentionDetector 그대로 |
| Drowsy | **100% 호환** | DrowsyDetector 그대로 |
| Fatigue (신규) | **Byte5 bit0 추가** | 옵션 A 권장 |
| PacketEncoder | **하위 호환** | fatigue=False일 때 기존과 동일 패킷 |
| CRC8 | **자동 호환** | Byte1~5 기반 계산이므로 Byte5 변경 시 자동 갱신 |

**핵심**: Fatigue=False로 두면 기존과 100% 동일한 패킷을 생성한다. Fatigue 기능은 opt-in 방식으로 안전하게 추가 가능.
