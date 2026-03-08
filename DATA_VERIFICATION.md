# K-EmoCon 전처리 데이터 검증 결과

## 검증 항목 요약

| 항목 | 결과 | 심각도 |
|------|------|--------|
| 라벨 값 일치 | ✓ 100% 일치 | - |
| PID 제외 사유 | ✓ 정당 (E4 데이터 없음) | - |
| PID별 segment 수 | ✓ 24/28 PID 정확 일치 | - |
| Train/Val PID 분리 | ✓ 겹침 없음 | - |
| **Normalization 비대칭** | ✗ 문제 있음 | **높음** |
| **Valence 분포 붕괴** | ✗ 89%가 단일값 | **치명적** |
| **대화 쌍 분리** | ⚠ 3쌍 split | **중간** |
| Polar R 계산 | ⚠ 일부 불일치 | 낮음 |
| PID 1 truncation | ⚠ 48 segments 누락 | 낮음 |

---

## 1. 라벨 값 일치 — ✓ PASS

원본 K-EmoCon `aggregated_external_annotations/P{pid}.external.csv`의 arousal/valence와 전처리 `segments_index.csv`의 `label_ext_A/V` 비교:

- **1,011개 샘플 직접 비교 → 불일치 0건 (100% 일치)**
- 라벨 소스: aggregated external annotation (5명 외부 평가자 majority vote)

## 2. PID 제외 — ✓ 정당

4명 제외 (PID 2, 3, 6, 7): 전부 **E4 웨어러블 데이터 없음**
- PID 2: E4 전체 센서 누락 (ACC, BVP, EDA, HR, IBI, TEMP)
- PID 3: E4 + Polar HR 누락
- PID 6: E4 + debate recording 누락
- PID 7: E4 전체 센서 누락

→ Bio 데이터 없이는 multimodal 시스템 학습 불가, 제외 타당

## 3. PID별 Segment 수 — ✓ 대부분 일치

28개 PID 중 24개 정확 일치. 차이나는 경우:
- **PID 1**: 원본 172 → 전처리 124 (48 segments 누락)
  - 원인: annotation은 860초까지, 전처리는 620초에서 끊김
  - 추정: E4 데이터가 620초에서 종료 (배터리/연결 끊김)
  - PID 1의 debate duration = 852초이므로, 마지막 ~230초 bio 데이터 없음

## 4. Train/Val Split — ✓ PID 레벨 분리, ⚠ 대화 쌍 일부 분리

**PID 겹침: 없음 (data leakage 없음)**

그러나 K-EmoCon은 2인 1조 토론이므로 **대화 쌍 분리** 검증 필요:

```
K-EmoCon 16개 대화 쌍:
  ( 1,  2) → val/제외     ( 3,  4) → 제외/train    ( 5,  6) → train/제외
  ( 7,  8) → 제외/train   ( 9, 10) → both_train ✓  (11, 12) → both_train ✓
  (13, 14) → both_val ✓   (15, 16) → both_train ✓  (17, 18) → val/train ⚠️
  (19, 20) → both_train ✓ (21, 22) → both_train ✓  (23, 24) → both_train ✓
  (25, 26) → train/val ⚠️ (27, 28) → both_train ✓  (29, 30) → train/val ⚠️
  (31, 32) → both_train ✓
```

**3쌍이 train/val로 갈림: (17,18), (25,26), (29,30)**

의미:
- 같은 오디오 파일을 공유 (p17.p18.wav 등)
- 같은 영상에 함께 등장
- 대화 토픽/맥락/반응 패턴이 상관됨
- 엄밀한 information leakage는 아니지만 (각자 다른 label), confound 가능성

**권장: 6-fold GroupKFold (V4에서 이미 사용) 시 대화 쌍 단위로 그룹핑**

---

## 5. Normalization — ✗ 비대칭 문제

### 현재 공식
```
A_norm = 0.5 × A_raw - 1.5
V_norm = 0.5 × V_raw - 1.5
```

### 결과 범위

| | Raw 범위 | Normalized 범위 | 문제 |
|---|----------|----------------|------|
| Arousal | [1, 4] | **[-1.0, +0.5]** | 비대칭: negative 쪽이 2배 넓음 |
| Valence | [2, 4] | **[-0.5, +0.5]** | 범위가 arousal의 절반 |

### 왜 문제인가

1. **모델 학습 불리**: MSE loss에서 A_norm = -1.0 (raw=1)과 A_norm = +0.5 (raw=4)의 예측 오차 가중치가 다름
2. **Polar head 왜곡**: R = sqrt(A² + V²)에서 arousal이 valence보다 2배 영향력
3. **Quadrant 경계 왜곡**: A_norm=0 (raw=3), V_norm=0 (raw=3)이 neutral이 아니라 "약간 높음"

### 올바른 정규화 (참고)
```
# 이론적 스케일 1-5 기준 (K-EmoCon 원래 설계)
A_norm = (A_raw - 3) / 2      # [1,5] → [-1, +1]
V_norm = (V_raw - 3) / 2      # [1,5] → [-1, +1]

# 실제 데이터 범위 기준 (aggregated external)
A_norm = 2*(A_raw - 1)/3 - 1  # [1,4] → [-1, +1]
V_norm = V_raw - 3             # [2,4] → [-1, +1]
```

---

## 6. Valence 분포 — ✗ 치명적 불균형

```
Valence 분포 (전체 3,577 samples):
  V=2:   184개 (5.1%)
  V=3: 3,186개 (89.1%)  ← 거의 전부
  V=4:   207개 (5.8%)
```

### 원인
- K-EmoCon은 토론 데이터 → 대부분 중립적(cheerful) 대화
- External rater 5명 중 대부분이 valence=3 평가
- Majority vote 후 더 극단값이 사라짐

### 결과
- **CCC(V) = 0.094**: 사실상 예측 불가능
- **Quadrant 분류가 실질적 2-class 문제**: Low/High Arousal만 의미
- V=3 (89%)이 V_norm=0 → V≥0 = "High Valence"로 분류
  → Q1+Q3 (High V)의 94%가 사실은 V=3

### Discrete emotion 분포도 동일 문제
```
cheerful: 3,810/4,187 = 91%
nervous:    330개
happy:       32개
angry:       15개
```

---

## 7. Polar R 계산 — ⚠ 일부 불일치

R = sqrt(A_norm² + V_norm²)이 대부분 맞지만, **210/3,577 samples (5.9%)에서 오차 존재**

| (A_norm, V_norm) | R_stored | R_expected | 차이 |
|-------------------|----------|------------|------|
| (±0.5, ±0.5) | 0.6614 | 0.7071 | -0.0457 |
| (±1.0, ±0.5) | 1.0000 | 1.1180 | clamp |

R이 [0, 1]로 clamp되어 있고, 대각선 값에서 원인 불명의 축소가 있음.
V4 모델은 R을 직접 예측하므로 실사용 영향은 적지만, **label로 R을 쓸 경우 재계산 필요**.

---

## 8. K-EmoCon 원본 스케일 정리

| Annotation 유형 | Arousal | Valence | 비고 |
|-----------------|---------|---------|------|
| Self | 1-5 | 1-5 | 참가자 본인 평가 |
| External (개별 rater) | 1-5 | 1-5 | 극소수 5 존재 |
| **Aggregated external** | **1-4** | **2-4** | majority vote로 극단값 소멸 |
| 전처리 사용 | 1-4 | 2-4 | aggregated external 사용 ✓ |

**주의**: K-EmoCon 논문은 1-5 스케일로 설계했지만, aggregated에서는 5와 1(valence)이 majority를 달성하지 못해 사라짐.

---

## 9. 전체 결론

### 잘 된 것
- 원본 라벨과 100% 일치
- E4 없는 PID 제외 타당
- PID 기준 train/val 분리 올바름
- Bio/Audio/Video 파일 경로 구조 정확

### 고쳐야 할 것

**[필수] Normalization 재설계**
현재 [-1, 0.5] / [-0.5, 0.5] 비대칭 → [-1, 1] 대칭으로 변경 필요

**[필수] Valence 전략 재고**
V=3이 89%인 상태에서 VA regression은 의미 없음. 선택지:
1. Valence를 binary (V≤2 vs V≥4)로 변환 → 극단값만 분류
2. Arousal-only 모델로 전환
3. Self-annotation 사용 (1-5 스케일, 분포 더 넓음)

**[권장] 대화 쌍 단위 CV**
GroupKFold에서 대화 쌍(pid pair)을 group으로 묶어야 함
현재 3쌍이 train/val로 갈려있음

**[권장] Polar R 재계산**
R을 label로 사용할 경우 sqrt(A_norm² + V_norm²)로 직접 재계산
