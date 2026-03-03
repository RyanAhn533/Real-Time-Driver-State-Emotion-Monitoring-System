#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Pipeline: CSV 추출 → 검증 → 학습
==========================================

CSV 없으면 자동 생성, 있으면 스킵.

사용법:
  python scripts/run_pipeline.py --config configs/mobilevit_fer.yaml

또는 단계별:
  python scripts/run_pipeline.py --config configs/mobilevit_fer.yaml --only preprocess
  python scripts/run_pipeline.py --config configs/mobilevit_fer.yaml --only train
"""

import sys
import argparse
import yaml
from pathlib import Path
from multiprocessing import set_start_method

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def step_preprocess(cfg: dict):
    """Stage 1-2: AU 좌표 CSV 추출 (MediaPipe FaceMesh)."""
    from data.preprocessors.au_extractor import extract_au_csv

    paths = cfg["paths"]
    pp = cfg.get("preprocess", {})

    train_root = Path(paths["raw_train"])
    val_root = Path(paths["raw_val"])
    csv_dir = Path(paths["csv_dir"])
    csv_dir.mkdir(parents=True, exist_ok=True)

    train_csv = csv_dir / "index_train.csv"
    val_csv = csv_dir / "index_val.csv"

    work_short = pp.get("work_short_side", 800)
    workers = pp.get("workers", None)

    # Train CSV
    if train_csv.exists():
        print(f"[SKIP] Train CSV already exists: {train_csv}")
    else:
        print(f"\n{'='*60}")
        print(f"[STEP 1] Extracting AU coords: {train_root}")
        print(f"{'='*60}")
        extract_au_csv(train_root, train_csv, work_short=work_short, workers=workers)

    # Val CSV
    if val_csv.exists():
        print(f"[SKIP] Val CSV already exists: {val_csv}")
    else:
        print(f"\n{'='*60}")
        print(f"[STEP 1] Extracting AU coords: {val_root}")
        print(f"{'='*60}")
        extract_au_csv(val_root, val_csv, work_short=work_short, workers=workers)

    # Update config with CSV paths
    cfg["paths"]["train_csv"] = str(train_csv)
    cfg["paths"]["val_csv"] = str(val_csv)

    return train_csv, val_csv


def step_validate(cfg: dict, train_csv: Path, val_csv: Path):
    """Stage 3: CSV 검증 + 정제."""
    from data.validators.csv_validator import validate_csv

    print(f"\n{'='*60}")
    print(f"[STEP 2] Validating CSVs")
    print(f"{'='*60}")

    clean_train, train_stats = validate_csv(str(train_csv), fix=True)
    clean_val, val_stats = validate_csv(str(val_csv), fix=True)

    # Use clean versions
    cfg["paths"]["train_csv"] = clean_train
    cfg["paths"]["val_csv"] = clean_val

    print(f"\n[SUMMARY]")
    print(f"  Train: {train_stats['clean_rows']} rows "
          f"(removed {train_stats['removed_rows']})")
    print(f"  Val:   {val_stats['clean_rows']} rows "
          f"(removed {val_stats['removed_rows']})")

    return clean_train, clean_val


def step_train(cfg: dict):
    """Stage 4: 모델 학습."""
    from training.trainer import train as run_training

    print(f"\n{'='*60}")
    print(f"[STEP 3] Training")
    print(f"{'='*60}")

    # Save resolved config to temp file for trainer
    import tempfile
    import json

    tmp_cfg = Path(cfg["paths"]["output_dir"]) / "resolved_config.yaml"
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_cfg, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    run_training(str(tmp_cfg))


def main():
    ap = argparse.ArgumentParser(description="Unified FER Pipeline")
    ap.add_argument("--config", type=str, required=True,
                    help="Config YAML path")
    ap.add_argument("--only", type=str, default=None,
                    choices=["preprocess", "validate", "train"],
                    help="Run only this step")
    ap.add_argument("--force-preprocess", action="store_true",
                    help="Re-extract CSV even if exists")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Force re-extract
    if args.force_preprocess:
        csv_dir = Path(cfg["paths"]["csv_dir"])
        for f in csv_dir.glob("index_*.csv"):
            f.unlink()
            print(f"[REMOVED] {f}")

    if args.only == "preprocess":
        step_preprocess(cfg)
        return

    if args.only == "validate":
        train_csv = Path(cfg["paths"].get("train_csv",
                         Path(cfg["paths"]["csv_dir"]) / "index_train.csv"))
        val_csv = Path(cfg["paths"].get("val_csv",
                       Path(cfg["paths"]["csv_dir"]) / "index_val.csv"))
        step_validate(cfg, train_csv, val_csv)
        return

    if args.only == "train":
        step_train(cfg)
        return

    # Full pipeline
    train_csv, val_csv = step_preprocess(cfg)
    clean_train, clean_val = step_validate(cfg, train_csv, val_csv)
    step_train(cfg)

    print(f"\n{'='*60}")
    print(f"[DONE] Full pipeline complete!")
    print(f"  Results: {cfg['paths']['output_dir']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
