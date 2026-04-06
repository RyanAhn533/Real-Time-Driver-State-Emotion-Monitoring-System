# 복합감정인지 USB 통신 프로토콜 변경 보고서

## v1 (기존) → v2 (모트렉스 규격)

---

## 1. 패킷 구조 비교

### v1 (기존 — 8 bytes)

```
Byte 0: SOF       = 0xAA
Byte 1: TYPE      = 0x01 (감정/상태)
Byte 2: SEQ       = 0x00~0xFF (순번)
Byte 3: LEN       = 0x02
Byte 4: BODY[0]   = [Emotion D7~D4] [Stress D3] [LowAttn D2] [Drowsy D1] [END_FLAG D0]
Byte 5: BODY[1]   = [Emo Intensity D7~D5] [State Intensity D4~D2] [NegEmo D1] [Rsvd D0]
Byte 6: CRC8      = polynomial 0x07 (Byte1~Byte5)
Byte 7: EOF       = 0xFE
```

### v2 (모트렉스 규격 — 13 bytes)

```
Byte  0: STX          = 0x02
Byte  1: VERSION      = 0x01
Byte  2: RESERVED_H   = 0x00
Byte  3: RESERVED_L   = 0x00
Byte  4: DEST         = 0x10 (슈퍼게이트→모트렉스) / 0x00 (모트렉스→슈퍼게이트)
Byte  5: DATA_LEN[0]  = 0x03 (LSB first)
Byte  6: DATA_LEN[1]  = 0x00
Byte  7: DATA_LEN[2]  = 0x00
Byte  8: COMMAND      = 0xF0
Byte  9: BODY[0]      = [Emotion D7~D4] [Stress D3] [LowAttn D2] [Drowsy D1] [Rsvd D0]
Byte 10: BODY[1]      = [Emo Intensity D7~D5] [State Intensity D4~D2] [Rsvd D1~D0]
Byte 11: CHECKSUM     = XOR (Byte0 ~ Byte10)
Byte 12: ETX          = 0x03
```

### 구조 변경 요약

| 항목 | v1 | v2 | 비고 |
|------|----|----|------|
| **총 길이** | 8 bytes | **13 bytes** | +5 bytes |
| **시작/종료** | 0xAA / 0xFE | **0x02 / 0x03** | 모트렉스 STX/ETX |
| **VERSION** | 없음 | **0x01** | 프로토콜 버전 관리 |
| **RESERVED** | 없음 | **2 bytes (0x0000)** | 향후 확장용 |
| **DEST** | 없음 | **0x10 / 0x00** | 양방향 통신 지원 |
| **DATA LENGTH** | 1 byte (LEN=2) | **3 bytes, LSB first (값=3)** | COMMAND(1) + BODY(2) |
| **COMMAND** | 없음 | **0xF0** | 감정인지 명령 코드 |
| **SEQ (순번)** | 있음 (0~255) | **삭제** | 규격에 없음 |
| **END_FLAG** | BODY[0] D0 | **삭제 → Reserved** | 규격에 없음 |
| **NegEmo** | BODY[1] D1 | **삭제 → Reserved** | 규격에 없음 |
| **Checksum** | CRC8 (poly 0x07) | **XOR Checksum** | 단순화 |

---

## 2. BODY 비트 필드 비교

### BODY[0] — 감정 코드 + 상태 플래그

```
        D7  D6  D5  D4  |  D3        D2          D1              D0
v1:   [ Emotion Code  ] | [Stress] [LowAttn] [Drowsy]       [END_FLAG]
v2:   [ Emotion Code  ] | [Stress] [LowAttn] [Drowsy Cand.] [Reserved]
```

| 비트 | v1 | v2 | 변경 |
|------|----|----|------|
| D7~D4 | Emotion Code (0~5) | Emotion Code (0~5, F) | 코드 매핑 변경 |
| D3 | Stress | Stress / High Tension | 동일 |
| D2 | Low Attention | Low Attention (주의 산만) | 동일 |
| D1 | Drowsy | Drowsy Candidate (졸음 후보) | 이름만 변경 |
| D0 | **END_FLAG** | **Reserved (0)** | 삭제 |

### BODY[1] — 강도

```
        D7  D6  D5  |  D4  D3  D2  |  D1        D0
v1:   [Emo Intens.] | [State Int.] | [NegEmo] [Rsvd]
v2:   [Emo Intens.] | [State Int.] | [Rsvd]   [Rsvd]
```

| 비트 | v1 | v2 | 변경 |
|------|----|----|------|
| D7~D5 | Emotion Intensity (0~7) | Emotion Intensity (0~7) | 동일 |
| D4~D2 | State Intensity (0~7) | State Intensity (0~7) | 동일 |
| D1 | **NegEmo flag** | **Reserved (0)** | 삭제 |
| D0 | Reserved | Reserved | 동일 |

---

## 3. Emotion Code 매핑 변경

### v1

```
Code 0: 공포(anxious)    Code 1: 놀람(surprised)  Code 2: 분노(angry)
Code 3: 슬픔/혐오(병합)   Code 4: 행복(happy)      Code 5: 중립(neutral)
```

### v2 (PPTX 규격)

```
Code 0: 공포    Code 1: 놀람    Code 2: 분노
Code 3: 슬픔    Code 4: 행복/기쁨  Code 5: 혐오
Code 6~F: Reserved
```

### K-FER → Protocol 매핑 변경

| K-FER Class | K-FER ID | v1 Code | v2 Code | 변경 |
|-------------|----------|---------|---------|------|
| angry | 0 | 2 (분노) | 2 (분노) | — |
| anxious | 1 | 0 (공포) | 0 (공포) | — |
| happy | 2 | 4 (행복) | 4 (행복/기쁨) | 이름만 |
| **hurt** | 3 | **3 (슬픔/혐오, sad 병합)** | **5 (혐오, 독립)** | **분리** |
| **neutral** | 4 | **5 (중립)** | **0xF (Reserved, 표시 안 함)** | **삭제** |
| **sad** | 5 | **3 (슬픔/혐오, hurt 병합)** | **3 (슬픔, 독립)** | **분리** |
| surprised | 6 | 1 (놀람) | 1 (놀람) | — |

**핵심 변경 3가지:**
1. **슬픔/혐오 분리**: v1에서 sad+hurt를 Code 3으로 병합했으나, v2에서 슬픔(3)과 혐오(5)로 분리
2. **중립 → Reserved**: 감정이 없는 상태(neutral)는 표시하지 않음 (PPTX: "감정이 0 또는 Reserved일 경우 표시하지 않음")
3. **6종 유지**: 실질적으로 전송되는 감정은 공포/놀람/분노/슬픔/행복/혐오 6종

---

## 4. Checksum 방식 변경

### v1: CRC8

```python
# polynomial=0x07, init=0x00
# 대상: Byte1 ~ Byte5 (TYPE, SEQ, LEN, BODY[0], BODY[1])
crc = 0x00
for byte in data:
    crc ^= byte
    for _ in range(8):
        if crc & 0x80:
            crc = ((crc << 1) ^ 0x07) & 0xFF
        else:
            crc = (crc << 1) & 0xFF
```

### v2: XOR Checksum

```python
# 대상: Byte0 ~ Byte10 (STX부터 BODY[1]까지 전부)
checksum = 0x00
for byte in data:
    checksum ^= byte
```

| 항목 | v1 (CRC8) | v2 (XOR) |
|------|-----------|----------|
| 알고리즘 | CRC8 (polynomial 0x07) | 단순 XOR |
| 대상 범위 | Byte1~Byte5 (5 bytes) | Byte0~Byte10 (11 bytes) |
| 에러 검출력 | 높음 (다항식 기반) | 낮음 (1-bit 에러만 보장) |
| 연산 비용 | 비트 단위 반복 | 바이트 단위 XOR |

---

## 5. 통신 방향 (v2 신규)

v1은 단방향이었으나, v2는 DEST 필드로 양방향 통신 지원.

```
슈퍼게이트 제어기 ──(DEST=0x10, CMD=0xF0)──→ 모트렉스 제어기
                  ←──(DEST=0x00, CMD=0xF0)──
```

| 방향 | DEST | 설명 |
|------|------|------|
| Request | 0x10 | 슈퍼게이트 → 모트렉스 (감정 데이터 전송) |
| Response | 0x00 | 모트렉스 → 슈퍼게이트 (ACK/감정 에코) |

---

## 6. 표시 및 사운드 규격 (v2 신규)

v1에는 없던 디스플레이 표시 + 사운드 규격이 추가됨.

### 팝업 표시 규칙
- 노멀 상태에서 이벤트 발생 시 팝업 **5회 반복** (0.5s On, 0.5s Off)
- 팝업 중 **비프음** 출력
- 감정/상태가 0 또는 Reserved일 경우 **표시하지 않음**

### 감정별 표시 메시지

| Emotion Code | 감정 | 표시 메시지 |
|:------------:|:----:|:----------:|
| 0 | 공포 | 안전하게 함께 가고 있어요 |
| 1 | 놀람 | 괜찮습니다, 안정적으로 주행 중입니다 |
| 2 | 분노 | 편안한 주행을 도와드릴게요 |
| 3 | 슬픔 | 편안한 주행을 도와드릴게요 |
| 4 | 행복/기쁨 | 안정적으로 주행 중입니다 |
| 5 | 혐오 | 차량이 안정적으로 유지하고 있습니다 |

### 상태별 표시 메시지

| 상태 | 표시 메시지 |
|:----:|:----------:|
| Stress / High Tension | 편안한 주행을 도와드릴게요 |
| Low Attention (주의 산만) | 주행에 집중해 주세요 |
| Drowsy Candidate (졸음 후보) | 잠시 휴식이 필요해 보여요 |

---

## 7. 패킷 예시 비교

### 예시 1: 행복 + 정상 상태

**v1 (8 bytes)**
```
AA 01 00 02 41 E8 xx FE
│  │  │  │  │  │  │  └─ EOF
│  │  │  │  │  │  └──── CRC8
│  │  │  │  │  └─────── BODY[1]: emo_int=7, state_int=2
│  │  │  │  └────────── BODY[0]: happy(4), no flags
│  │  │  └───────────── LEN=2
│  │  └──────────────── SEQ=0
│  └─────────────────── TYPE=1
└────────────────────── SOF
```

**v2 (13 bytes)**
```
02 01 00 00 10 03 00 00 F0 40 E8 48 03
│  │  │  │  │  │  │  │  │  │  │  │  └─ ETX
│  │  │  │  │  │  │  │  │  │  │  └──── XOR Checksum
│  │  │  │  │  │  │  │  │  │  └─────── BODY[1]: emo_int=7, state_int=2
│  │  │  │  │  │  │  │  │  └────────── BODY[0]: happy(4), no flags
│  │  │  │  │  │  │  │  └───────────── COMMAND=0xF0
│  │  │  │  │  └─┴──┴──────────────── DATA_LENGTH=3 (LSB first)
│  │  │  │  └───────────────────────── DEST=0x10 (모트렉스)
│  │  └──┴──────────────────────────── RESERVED=0x0000
│  └────────────────────────────────── VERSION=0x01
└───────────────────────────────────── STX
```

### 예시 2: 분노 + 스트레스

```
v1:  AA 01 01 02 28 D8 xx FE          (8B)
v2:  02 01 00 00 10 03 00 00 F0 28 D8 10 03  (13B)
```

### 예시 3: 졸음 (neutral + PERCLOS ≥ 0.4)

```
v1:  AA 01 02 02 52 64 xx FE          (8B)  ← neutral=Code 5, drowsy flag
v2:  02 01 00 00 10 03 00 00 F0 F2 64 76 03  (13B)  ← neutral=Code F(Reserved), drowsy flag
```

→ v2에서는 감정이 Reserved이므로 감정은 표시 안 하고, 상태 메시지 "잠시 휴식이 필요해 보여요"만 표시

---

## 8. 코드 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `packet_encoder.py` | 13B 패킷 구조, XOR checksum, DEST/CMD/DLEN 추가, Emotion Code 분리, 표시 메시지 |
| `gateway_sender.py` | 13B 검증, `core.logger` → `logging` 표준 모듈, protocol 필드 추가 |
| `demo_pipeline.py` | v2 encoder 연동, `display_message`/`packet_bytes` 출력, `end_flag`/`seq` 제거 |
| `__init__.py` | `PROTOCOL_MESSAGES`, `STATUS_MESSAGES` export 추가 |

---

## 9. 하위 호환성

**v2는 v1과 호환되지 않음.**

- 패킷 크기가 다름 (8B vs 13B)
- STX/ETX가 다름 (0xAA/0xFE vs 0x02/0x03)
- Checksum 방식이 다름 (CRC8 vs XOR)
- Emotion Code 매핑이 다름 (neutral: 5 vs Reserved)

수신 측에서 패킷 첫 바이트로 버전 판별 가능:
- `0xAA` → v1 (8-byte)
- `0x02` → v2 (13-byte 모트렉스)

---

*작성일: 2026-03-30*
*근거: 복합감정인지 통신 프로토콜__v1.pptx, 복합감정인지 표시 및 사운드__v1.pptx*
