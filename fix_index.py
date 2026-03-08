"""
K-EmoCon segments_index CSV 수정 스크립트

변경 사항:
1. Binary label 추가: Low(1,2)=0, High(3,4,5)=1
2. Normalization 대칭 수정: [-1, +1]
3. 대화 쌍(pair_id) 컬럼 추가
4. Train/Val split 재구성 (대화 쌍 단위 분리)
5. Polar R/theta 재계산
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("data/precessed_data")

# ── K-EmoCon 대화 쌍 (원본 메타데이터 기반) ──
PAIRS = {
    1: 0, 2: 0,
    3: 1, 4: 1,
    5: 2, 6: 2,
    7: 3, 8: 3,
    9: 4, 10: 4,
    11: 5, 12: 5,
    13: 6, 14: 6,
    15: 7, 16: 7,
    17: 8, 18: 8,
    19: 9, 20: 9,
    21: 10, 22: 10,
    23: 11, 24: 11,
    25: 12, 26: 12,
    27: 13, 28: 13,
    29: 14, 30: 14,
    31: 15, 32: 15,
}

# 제외된 PID (E4 데이터 없음): 2, 3, 6, 7
EXCLUDED_PIDS = {2, 3, 6, 7}


def fix_index(df):
    """CSV 인덱스 수정"""

    # 1. Binary labels: Low(1,2)=0, High(3,4,5)=1
    df["label_A_binary"] = (df["label_ext_A"] >= 3).astype(int)
    df["label_V_binary"] = (df["label_ext_V"] >= 3).astype(int)

    # 2. Normalization 수정 — 실제 데이터 범위 기준 대칭 [-1, +1]
    # Arousal: 관측 범위 [1, 4] → [-1, +1]
    # 공식: (raw - 1) / (4 - 1) * 2 - 1 = (raw - 1) * 2/3 - 1
    df["label_ext_A_norm"] = (df["label_ext_A"] - 1.0) * 2.0 / 3.0 - 1.0

    # Valence: 관측 범위 [2, 4] → [-1, +1]
    # 공식: (raw - 2) / (4 - 2) * 2 - 1 = (raw - 2) - 1 = raw - 3
    df["label_ext_V_norm"] = df["label_ext_V"] - 3.0

    # 3. Polar 좌표 재계산
    A = df["label_ext_A_norm"].values
    V = df["label_ext_V_norm"].values
    df["label_ext_R"] = np.sqrt(A**2 + V**2)
    df["label_ext_theta_deg"] = np.degrees(np.arctan2(A, V)) % 360

    # 4. 대화 쌍 ID
    df["pair_id"] = df["pid"].map(PAIRS)

    # 5. Quadrant label (binary 기반)
    # Q1: HA-HV, Q2: LA-HV, Q3: LA-LV, Q4: HA-LV
    df["quadrant"] = df["label_A_binary"] * 2 + df["label_V_binary"]
    # 0=LA-LV(Q3), 1=LA-HV(Q2), 2=HA-LV(Q4), 3=HA-HV(Q1)

    return df


def make_pair_aware_split(df, val_ratio=0.2, seed=42):
    """대화 쌍 단위로 train/val 분리"""
    rng = np.random.RandomState(seed)

    # 유효한 pair_id별 샘플 수
    pair_counts = df.groupby("pair_id").size().reset_index(name="count")
    pair_ids = pair_counts["pair_id"].values.copy()
    rng.shuffle(pair_ids)

    total = len(df)
    target_val = int(total * val_ratio)

    val_pairs = []
    val_count = 0
    for pid in pair_ids:
        cnt = pair_counts[pair_counts["pair_id"] == pid]["count"].values[0]
        if val_count + cnt <= target_val * 1.3:  # 약간의 여유
            val_pairs.append(pid)
            val_count += cnt
            if val_count >= target_val:
                break

    train_pairs = [p for p in pair_counts["pair_id"] if p not in val_pairs]

    val_mask = df["pair_id"].isin(val_pairs)
    train_mask = ~val_mask

    return df[train_mask].copy(), df[val_mask].copy(), train_pairs, val_pairs


def main():
    print("=" * 60)
    print("K-EmoCon segments_index 수정")
    print("=" * 60)

    # 원본 로드
    df = pd.read_csv(BASE / "segments_index.csv")
    print(f"\n전체: {len(df)} samples, {df['pid'].nunique()} PIDs")

    # 수정 적용
    df = fix_index(df)

    # ── 수정 결과 확인 ──
    print("\n--- Normalization (수정 후) ---")
    print(f"A_norm: [{df['label_ext_A_norm'].min():.4f}, {df['label_ext_A_norm'].max():.4f}]")
    print(f"V_norm: [{df['label_ext_V_norm'].min():.4f}, {df['label_ext_V_norm'].max():.4f}]")

    print("\n--- Binary Labels ---")
    print(f"A_binary: {dict(df['label_A_binary'].value_counts().sort_index())}")
    print(f"V_binary: {dict(df['label_V_binary'].value_counts().sort_index())}")

    print("\n--- Quadrant ---")
    q_names = {0: "LA-LV(Q3)", 1: "LA-HV(Q2)", 2: "HA-LV(Q4)", 3: "HA-HV(Q1)"}
    for q, name in q_names.items():
        cnt = (df["quadrant"] == q).sum()
        print(f"  {name}: {cnt} ({cnt/len(df)*100:.1f}%)")

    # ── Train/Val 분리 (대화 쌍 단위) ──
    train_df, val_df, train_pairs, val_pairs = make_pair_aware_split(df)

    print("\n--- Train/Val Split (pair-aware) ---")
    print(f"Train: {len(train_df)} samples, {train_df['pid'].nunique()} PIDs, pairs={sorted(train_pairs)}")
    print(f"Val:   {len(val_df)} samples, {val_df['pid'].nunique()} PIDs, pairs={sorted(val_pairs)}")

    # 대화 쌍 분리 확인
    train_pids = set(train_df["pid"].unique())
    val_pids = set(val_df["pid"].unique())
    overlap = train_pids & val_pids
    print(f"PID 겹침: {overlap if overlap else '없음 ✓'}")

    # 같은 pair가 train/val로 갈리는지 확인
    train_pair_set = set(train_df["pair_id"].unique())
    val_pair_set = set(val_df["pair_id"].unique())
    pair_overlap = train_pair_set & val_pair_set
    print(f"Pair 겹침: {pair_overlap if pair_overlap else '없음 ✓'}")

    # Train binary 분포
    print(f"\nTrain A_binary: {dict(train_df['label_A_binary'].value_counts().sort_index())}")
    print(f"Train V_binary: {dict(train_df['label_V_binary'].value_counts().sort_index())}")
    print(f"Val   A_binary: {dict(val_df['label_A_binary'].value_counts().sort_index())}")
    print(f"Val   V_binary: {dict(val_df['label_V_binary'].value_counts().sort_index())}")

    # Val 각 PID 출력
    print(f"\nTrain PIDs: {sorted(train_df['pid'].unique())}")
    print(f"Val   PIDs: {sorted(val_df['pid'].unique())}")

    # ── 저장 ──
    # 백업
    for f in ["segments_index.csv", "segments_index_train.csv", "segments_index_val.csv"]:
        src = BASE / f
        bak = BASE / f"{f}.bak"
        if src.exists() and not bak.exists():
            import shutil
            shutil.copy2(src, bak)
            print(f"\n백업: {f} → {f}.bak")

    # 컬럼 순서 정리
    label_cols = [
        "pid", "seg_idx", "pair_id", "seconds", "t_start", "t_end", "path",
        "label_ext_A", "label_ext_V",
        "label_A_binary", "label_V_binary", "quadrant",
        "label_ext_A_norm", "label_ext_V_norm",
        "label_ext_R", "label_ext_theta_deg",
    ]
    other_cols = [c for c in df.columns if c not in label_cols]
    col_order = label_cols + other_cols

    df[col_order].to_csv(BASE / "segments_index.csv", index=False)
    train_df[col_order].to_csv(BASE / "segments_index_train.csv", index=False)
    val_df[col_order].to_csv(BASE / "segments_index_val.csv", index=False)

    print("\n✓ 저장 완료:")
    print(f"  {BASE / 'segments_index.csv'}")
    print(f"  {BASE / 'segments_index_train.csv'}")
    print(f"  {BASE / 'segments_index_val.csv'}")


if __name__ == "__main__":
    main()
