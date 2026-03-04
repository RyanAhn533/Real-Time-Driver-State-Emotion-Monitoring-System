#!/usr/bin/env python3
"""
V4-full → Teacher Artifact Extractor
======================================
V4-full (tri-modal) 학습 완료 후 실행.
각 segment에 대해 Teacher의 예측값과 audio embedding을 추출하여 저장.

V4-lite v2가 이 artifact를 Prediction KD + Feature KD에 활용.

파이프라인:
  1) train_mm_v4.py 학습 완료 → checkpoints_mm_v4/fold_XX/best_model.pt
  2) 이 스크립트 실행 → teacher_artifacts/pid_XX/seg_XXXXX.npz
  3) V4-lite trainer에서 --teacher_dir 로 불러서 KD

Usage:
  # Fold-aware 추출 (권장: 각 fold의 val set을 해당 fold checkpoint으로)
  python extract_teacher_artifacts.py \
      --ckpt_dir ./checkpoints_mm_v4 \
      --out_dir /home/jy/260210/precessed_data/teacher_artifacts

  # 전체 데이터셋을 하나의 checkpoint으로 (빠르지만 overfitted 위험)
  python extract_teacher_artifacts.py \
      --ckpt_dir ./checkpoints_mm_v4 \
      --no_fold_aware \
      --out_dir /home/jy/260210/precessed_data/teacher_artifacts

  # GAFMTF bio 버전 Teacher
  python extract_teacher_artifacts.py \
      --ckpt_dir ./checkpoints_mm_v4_gafmtf \
      --bio_encoder_type gafmtf \
      --out_dir /home/jy/260210/precessed_data/teacher_artifacts_gafmtf

생성 결과:
  teacher_artifacts/
  ├── pid_01/
  │   ├── seg_00000.npz
  │   │   ├── A_teacher_cont   float32 scalar  — Teacher의 continuous A 예측
  │   │   ├── V_teacher_cont   float32 scalar  — Teacher의 continuous V 예측
  │   │   ├── z_audio_ast      float32 (128,)  — AST audio token의 mean-pooled embedding
  │   │   ├── z_pooled         float32 (128,)  — Perceiver pooled latent
  │   │   ├── gate_weights     float32 (16,3)  — Gate weights (vid,aud,bio)
  │   │   └── A_true, V_true   float32 scalar  — GT label (편의상 동봉)
  │   └── ...
  ├── teacher_index.csv
  └── extraction_config.json
"""

import argparse
import json
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

# V4-full imports — 이 파일들이 같은 디렉토리에 있어야 함
from dataset_mm_v4 import KEmoconMultiModalDatasetV4, collate_mm_v4
from model_mm_v4 import VisionMER_V4


def load_v4_checkpoint(ckpt_path, device, bio_encoder_type="4ch"):
    """
    V4-full best_model.pt 로드.

    train_mm_v4.py의 저장 형식:
      torch.save({
          "fold": ..., "epoch": ..., "score": ...,
          "model_state": model.state_dict(),   ★ key = "model_state"
          "val": ..., "args": vars(args),
      }, fold_dir / "best_model.pt")
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Config 추출
    config = ckpt.get("args", {})
    bio_type = bio_encoder_type or config.get("bio_encoder_type", "4ch")

    model = VisionMER_V4(
        d_model=config.get("d_model", 128),
        nhead=config.get("nhead", 4),
        num_frames=config.get("num_frames", 16),
        window_size=config.get("window_size", 4),
        enc_depth=tuple(config.get("enc_depth", [2, 2, 2])),
        num_latents=config.get("num_latents", 32),
        fusion_depth=config.get("fusion_depth", 2),
        dropout=0.0,  # inference
        ast_model_name=config.get("ast_model_name",
                                   "MIT/ast-finetuned-audioset-10-10-0.4593"),
        freeze_ast=False,
        use_polar_head=config.get("use_polar_head", True),
        use_binary_head=config.get("use_binary_head", True),
        bio_encoder_type=bio_type,
    )

    # ★ state dict key = "model_state" (train_mm_v4.py line 431)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    if isinstance(state, OrderedDict) or isinstance(state, dict):
        model.load_state_dict(state, strict=False)
    else:
        raise ValueError(f"Cannot load state_dict from {ckpt_path}")

    model.to(device).eval()

    fold = ckpt.get("fold", -1)
    epoch = ckpt.get("epoch", -1)
    score = ckpt.get("score", -1)
    print(f"    Loaded: fold={fold}, epoch={epoch}, score={score:.4f}")

    return model, config


@torch.no_grad()
def extract_batch(model, batch, device):
    """
    V4-full forward(return_artifacts=True) 결과에서 Teacher artifact 추출.

    model_mm_v4.py의 return_artifacts 형식 (line 612-624):
      av_pred:  (B, 2)       — [A, V] continuous prediction
      q_logits: (B, 4)       — quadrant logits
      art: dict
        "pooled_feat":   (B, d_model)  — Perceiver pooled latent
        "gate_weights":  (B, T, 3)     — gate weights (vid, aud, bio)
        "aud_tokens":    (B, T, d)     — AST encoded audio tokens
        "bio_tokens":    (B, T, d)     — bio conditioned tokens
        "vid_tokens":    (B, T, d)     — video tokens
        "av_cartesian":  (B, 2)
        "polar_raw":     (B, 3) or None
        "bin_logits":    (B, 2) or None
    """
    bio, aud, vid, fmask, y, pids, spk_ids, quality = batch
    bio = bio.to(device)
    aud = aud.to(device)
    vid = vid.to(device)
    fmask = fmask.to(device)
    spk_ids = spk_ids.to(device)
    quality = quality.to(device)

    av_pred, q_logits, art = model(
        bio, aud, vid,
        face_mask=fmask,
        speaker_ids=spk_ids,
        quality=quality,
        return_artifacts=True,
    )

    B = av_pred.size(0)
    results = []
    for i in range(B):
        results.append({
            "A_teacher_cont": float(av_pred[i, 0].cpu()),
            "V_teacher_cont": float(av_pred[i, 1].cpu()),
            "A_true": float(y[i, 0]),
            "V_true": float(y[i, 1]),
            # AST audio embedding: mean-pool over time → (d_model,)
            "z_audio_ast": art["aud_tokens"][i].mean(dim=0).cpu().numpy().astype(np.float32),
            # Perceiver pooled latent → (d_model,)
            "z_pooled": art["pooled_feat"][i].cpu().numpy().astype(np.float32),
            # Gate weights → (T, 3)
            "gate_weights": art["gate_weights"][i].cpu().numpy().astype(np.float32),
            # PID
            "pid": int(pids[i]),
        })

    return results


def extract_from_loader(model, loader, device, out_dir, df_subset):
    """DataLoader → Teacher NPZ 저장 + index row 수집."""
    rows = []
    sample_idx = 0

    for batch in tqdm(loader, desc="  Extracting"):
        batch_results = extract_batch(model, batch, device)

        for item in batch_results:
            if sample_idx >= len(df_subset):
                break

            row_data = df_subset.iloc[sample_idx]
            pid = int(row_data["pid"])
            seg_idx = int(row_data.get("seg_idx", sample_idx))

            # 저장 경로
            rel_path = f"pid_{pid:02d}/seg_{seg_idx:05d}.npz"
            abs_path = out_dir / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                str(abs_path),
                A_teacher_cont=np.float32(item["A_teacher_cont"]),
                V_teacher_cont=np.float32(item["V_teacher_cont"]),
                A_true=np.float32(item["A_true"]),
                V_true=np.float32(item["V_true"]),
                z_audio_ast=item["z_audio_ast"],
                z_pooled=item["z_pooled"],
                gate_weights=item["gate_weights"],
            )

            rows.append({
                "pid": pid,
                "seg_idx": seg_idx,
                "teacher_path": rel_path,
                "A_teacher": item["A_teacher_cont"],
                "V_teacher": item["V_teacher_cont"],
                "A_true": item["A_true"],
                "V_true": item["V_true"],
            })
            sample_idx += 1

    return rows


def main():
    parser = argparse.ArgumentParser(description="V4-full → Teacher Artifact Extractor")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="V4-full checkpoint directory")
    parser.add_argument("--base_dir", type=str, default="/home/jy/260210/precessed_data")
    parser.add_argument("--index_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--fold_aware", action="store_true", default=True)
    parser.add_argument("--no_fold_aware", dest="fold_aware", action="store_false")
    parser.add_argument("--n_folds", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bio_encoder_type", type=str, default=None,
                        choices=["4ch", "gafmtf", None])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_csv = args.index_csv or f"{args.base_dir}/segments_index.csv"
    df = pd.read_csv(index_csv)

    # V4-full과 동일한 필터링
    for col in ["has_bio_img", "has_audio_mel", "has_video_face"]:
        if col in df.columns:
            df = df[df[col] == 1]
    df = df.dropna(subset=["label_ext_A_norm", "label_ext_V_norm"]).reset_index(drop=True)

    # Experiment config
    exp_config_path = ckpt_dir / "experiment_config.json"
    exp_config = {}
    if exp_config_path.exists():
        with open(exp_config_path) as f:
            exp_config = json.load(f)

    bio_type = args.bio_encoder_type or exp_config.get("bio_encoder_type", "4ch")
    use_gafmtf = (bio_type == "gafmtf")

    print(f"{'='*60}")
    print(f"V4-full Teacher Artifact Extraction")
    print(f"{'='*60}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Bio encoder:    {bio_type}")
    print(f"Output dir:     {out_dir}")
    print(f"Fold-aware:     {args.fold_aware}")
    print(f"Total samples:  {len(df)}")

    all_rows = []
    t_start = time.time()

    if args.fold_aware:
        # GroupKFold — seed and split must match train_mm_v4.py exactly
        groups = df["pid"].astype(int).to_numpy()
        gkf = GroupKFold(n_splits=args.n_folds)

        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(df, groups=groups), 1):
            # train_mm_v4.py saves as fold_01, fold_02, ...
            ckpt_path = ckpt_dir / f"fold_{fold_idx:02d}" / "best_model.pt"
            if not ckpt_path.exists():
                # fallback: fold_1 (without zero-padding)
                ckpt_path = ckpt_dir / f"fold_{fold_idx}" / "best_model.pt"
            if not ckpt_path.exists():
                print(f"  ⚠️ fold {fold_idx}: checkpoint not found, skipping")
                continue

            val_df = df.iloc[va_idx].reset_index(drop=True)
            print(f"\nFold {fold_idx}: {len(val_df)} val samples")
            print(f"  Val PIDs: {sorted(val_df['pid'].unique().tolist())}")

            model, _ = load_v4_checkpoint(ckpt_path, device, bio_type)

            ds = KEmoconMultiModalDatasetV4(
                val_df, args.base_dir,
                use_bio_gafmtf=use_gafmtf,
            )
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
                collate_fn=collate_mm_v4,
            )

            fold_rows = extract_from_loader(model, loader, device, out_dir, val_df)
            all_rows.extend(fold_rows)
            print(f"  ✅ {len(fold_rows)} segments extracted")

            del model
            torch.cuda.empty_cache()

    else:
        # Single checkpoint
        ckpt_candidates = sorted(ckpt_dir.glob("fold_*/best_model.pt"))
        if not ckpt_candidates:
            print("❌ No checkpoint found!")
            return
        ckpt_path = ckpt_candidates[0]
        print(f"\nUsing single checkpoint: {ckpt_path}")

        model, _ = load_v4_checkpoint(ckpt_path, device, bio_type)
        ds = KEmoconMultiModalDatasetV4(df, args.base_dir, use_bio_gafmtf=use_gafmtf)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            collate_fn=collate_mm_v4,
        )
        all_rows = extract_from_loader(model, loader, device, out_dir, df)

    elapsed = time.time() - t_start

    # Teacher index CSV
    teacher_df = pd.DataFrame(all_rows)
    teacher_df.to_csv(out_dir / "teacher_index.csv", index=False)

    # Config
    extraction_config = {
        "ckpt_dir": str(ckpt_dir),
        "bio_encoder_type": bio_type,
        "fold_aware": args.fold_aware,
        "n_folds": args.n_folds,
        "n_extracted": len(all_rows),
        "elapsed_sec": round(elapsed, 1),
        "teacher_model": "VisionMER_V4",
        "teacher_d_model": exp_config.get("d_model", 128),
    }
    with open(out_dir / "extraction_config.json", "w") as f:
        json.dump(extraction_config, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Extraction complete! ({elapsed/60:.1f} min)")
    print(f"   {len(all_rows)} segments → {out_dir}")

    if all_rows:
        a_arr = np.array([r["A_teacher"] for r in all_rows])
        v_arr = np.array([r["V_teacher"] for r in all_rows])
        print(f"   A_teacher: {a_arr.mean():.3f} ± {a_arr.std():.3f}")
        print(f"   V_teacher: {v_arr.mean():.3f} ± {v_arr.std():.3f}")

    print(f"\n▶ Next: V4-lite v2 학습 (with KD)")
    print(f"  python -m modalities.av_lite.trainer \\")
    print(f"    --teacher_dir {out_dir} \\")
    print(f"    --mu_v4kd 0.3 --mu_ast 0.1")


if __name__ == "__main__":
    main()
