#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV Validator
==============
AU CSV 품질 검증 + 자동 정제.
NaN/inf 행 제거, 누락 파일 제거, 통계 출력.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def validate_csv(csv_path: str, fix: bool = True) -> Tuple[str, dict]:
    """
    CSV 검증 및 선택적 정제.

    Args:
        csv_path: 검증할 CSV 경로
        fix: True면 문제 행 제거한 clean CSV 저장

    Returns:
        (clean_csv_path, stats_dict)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    original_len = len(df)

    stats = {
        "original_rows": original_len,
        "columns": len(df.columns),
    }

    # 1) NaN 체크
    nan_counts = df.isna().sum()
    nan_total = nan_counts.sum()
    stats["nan_total"] = int(nan_total)
    if nan_total > 0:
        print(f"[WARN] NaN 발견 ({nan_total}개):")
        for col, cnt in nan_counts.items():
            if cnt > 0:
                print(f"  {col}: {cnt}")

    # 2) inf 체크
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    inf_total = inf_counts.sum()
    stats["inf_total"] = int(inf_total)
    if inf_total > 0:
        print(f"[WARN] inf 발견 ({inf_total}개):")
        for col, cnt in inf_counts.items():
            if cnt > 0:
                print(f"  {col}: {cnt}")

    # 3) 파일 존재 여부
    missing = 0
    exists_mask = []
    for p in df["path"]:
        e = Path(str(p)).exists()
        exists_mask.append(e)
        if not e:
            missing += 1
    stats["missing_files"] = missing
    if missing > 0:
        print(f"[WARN] 존재하지 않는 파일: {missing}개")

    # 4) 클래스 분포
    class_dist = df["label"].value_counts().to_dict()
    stats["class_distribution"] = class_dist
    print(f"\n[INFO] 클래스 분포:")
    for cls, cnt in sorted(class_dist.items()):
        print(f"  {cls}: {cnt}")

    # 5) 좌표 범위 체크
    coord_cols = [c for c in df.columns if c.endswith("_cx") or c.endswith("_cy")]
    if coord_cols:
        coord_df = df[coord_cols].astype(float)
        neg_count = (coord_df < 0).sum().sum()
        stats["negative_coords"] = int(neg_count)
        if neg_count > 0:
            print(f"[WARN] 음수 좌표: {neg_count}개")

    # Fix
    clean_path = csv_path
    if fix and (nan_total > 0 or inf_total > 0 or missing > 0):
        clean_df = df.copy()

        # NaN/inf 행 제거
        if nan_total > 0:
            clean_df = clean_df.dropna()
        if inf_total > 0:
            mask = ~np.isinf(clean_df[numeric_cols]).any(axis=1)
            clean_df = clean_df[mask]

        # 누락 파일 제거
        if missing > 0:
            clean_df = clean_df[exists_mask[:len(clean_df)]]

        clean_path = csv_path.parent / f"{csv_path.stem}_clean.csv"
        clean_df.to_csv(clean_path, index=False)
        stats["clean_rows"] = len(clean_df)
        stats["removed_rows"] = original_len - len(clean_df)
        print(f"\n[INFO] Clean CSV 저장: {clean_path}")
        print(f"  원본: {original_len}행 → 정제: {len(clean_df)}행 (제거: {stats['removed_rows']}행)")
    else:
        stats["clean_rows"] = original_len
        stats["removed_rows"] = 0
        print(f"\n[INFO] 문제 없음. 원본 그대로 사용.")

    return str(clean_path), stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--no-fix", action="store_true")
    args = ap.parse_args()
    validate_csv(args.csv, fix=not args.no_fix)
