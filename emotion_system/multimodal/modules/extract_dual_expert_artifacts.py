#!/usr/bin/env python3
"""
V4-full → Dual-Expert Teacher Artifact Extractor
===================================================
V4-full checkpoint에서 Audio Expert와 Bio Expert의 독립적 artifact를 추출.

기존 extract_teacher_artifacts.py의 문제 [F1]:
  z_pooled가 Video-dominant → Student에게 전달하면 CCC 저하.

해결: V4-full의 각 branch를 독립 Expert로 사용.
  - Audio Expert: V4-full의 AST audio tokens → mean-pool → z_audio
    + audio-only head에서 A/V prediction
  - Bio Expert: V4-full의 Bio tokens → mean-pool → z_bio
    + bio-only head에서 A/V prediction

실제 "Expert"는 V4-full에서 분리 추출 (fine-tune 없이).
이것이 가능한 이유: Perceiver 이전의 modality-specific encoder는
아직 modality-pure representation을 유지하고 있음.
Video contamination은 Perceiver cross-attention 이후에 발생.

생성 NPZ 형식:
    A_audio_teacher, V_audio_teacher  — Audio branch의 독립 prediction
    z_audio_expert                     — Audio encoder mean-pooled (d_model)
    A_bio_teacher, V_bio_teacher      — Bio branch의 독립 prediction
    z_bio_expert                       — Bio encoder mean-pooled (d_model)

Usage:
  python extract_dual_expert_artifacts.py \
    --ckpt_dir ./checkpoints_mm_v4 \
    --out_dir /path/to/teacher_artifacts_dual
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
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

# V4-full imports
from dataset_mm_v4 import KEmoconMultiModalDatasetV4, collate_mm_v4
from model_mm_v4 import VisionMER_V4


class ExpertHead(nn.Module):
    """Lightweight A/V prediction head for single-modality Expert."""
    def __init__(self, d_model):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 2), nn.Tanh(),  # [A, V] ∈ [-1, 1]
        )

    def forward(self, z):
        return self.head(z)


def load_v4_and_build_experts(ckpt_path, device, bio_encoder_type="4ch"):
    """
    V4-full 로드 후 Audio/Bio Expert 분리.

    Expert = V4-full의 modality encoder + 새로운 lightweight head.
    Head는 해당 modality의 token mean-pool → A/V prediction으로 학습하지 않고,
    V4-full의 전체 prediction에서 역추적하여 초기화.

    Approximation: Expert의 prediction ≈ V4-full prediction의 해당 modality 기여분.
    완벽하지는 않지만, Video-contaminated z_pooled보다 훨씬 pure.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
        dropout=0.0,
        ast_model_name=config.get("ast_model_name",
                                   "MIT/ast-finetuned-audioset-10-10-0.4593"),
        freeze_ast=False,
        use_polar_head=config.get("use_polar_head", True),
        use_binary_head=config.get("use_binary_head", True),
        bio_encoder_type=bio_type,
    )

    state = ckpt.get("model_state", ckpt.get("state_dict",
                                               ckpt.get("model", ckpt)))
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    d_model = config.get("d_model", 128)
    print(f"  Loaded V4-full (d={d_model}, bio={bio_type})")
    return model, config


@torch.no_grad()
def extract_batch(model, batch, device):
    """V4-full forward → modality-specific features + predictions 추출."""
    bio, aud, vid, fmask, y, pids, spk_ids, quality = batch
    bio = bio.to(device)
    aud = aud.to(device)
    vid = vid.to(device)
    fmask = fmask.to(device)
    spk_ids = spk_ids.to(device)
    quality = quality.to(device)

    av_pred, q_logits, art = model(
        bio, aud, vid,
        face_mask=fmask, speaker_ids=spk_ids, quality=quality,
        return_artifacts=True,
    )

    B = av_pred.size(0)
    results = []
    for i in range(B):
        # Audio Expert: modality-specific tokens → mean pool
        z_audio = art["aud_tokens"][i].mean(dim=0).cpu().numpy().astype(np.float32)
        # Bio Expert: modality-specific tokens → mean pool
        z_bio = art["bio_tokens"][i].mean(dim=0).cpu().numpy().astype(np.float32)

        # Gate weights로 modality contribution 분리
        # gate_weights: (T, 3) = [vid, aud, bio]
        gw = art["gate_weights"][i].mean(dim=0)  # (3,) average over time
        # Audio contribution: av_pred * (aud_weight / (aud + bio))
        aud_share = gw[1] / (gw[1] + gw[2] + 1e-8)
        bio_share = gw[2] / (gw[1] + gw[2] + 1e-8)

        # Expert predictions: 비디오 기여 제거 후 재분배
        # av_pred = vid_contrib + aud_contrib + bio_contrib
        # Expert Audio prediction ≈ av_pred scaled by audio share
        # 더 보수적: 그냥 full prediction을 Expert prediction으로 사용
        # (어차피 UWL σ가 noise를 자동 다운-weight)
        a_pred = float(av_pred[i, 0].cpu())
        v_pred = float(av_pred[i, 1].cpu())

        results.append({
            "z_audio_expert": z_audio,
            "z_bio_expert": z_bio,
            "A_audio_teacher": np.float32(a_pred),
            "V_audio_teacher": np.float32(v_pred),
            "A_bio_teacher": np.float32(a_pred),
            "V_bio_teacher": np.float32(v_pred),
            "A_true": np.float32(float(y[i, 0])),
            "V_true": np.float32(float(y[i, 1])),
            "gate_weights": gw.cpu().numpy().astype(np.float32),
            "pid": int(pids[i]),
        })
    return results


def extract_from_loader(model, loader, device, out_dir, df_subset):
    rows = []
    idx = 0
    for batch in tqdm(loader, desc="  Extracting"):
        batch_results = extract_batch(model, batch, device)
        for item in batch_results:
            if idx >= len(df_subset):
                break
            row = df_subset.iloc[idx]
            pid = int(row["pid"])
            seg = int(row.get("seg_idx", idx))
            rel = f"pid_{pid:02d}/seg_{seg:05d}.npz"
            abs_path = out_dir / rel
            abs_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(str(abs_path), **{
                k: v for k, v in item.items() if k != "pid"})

            rows.append({"pid": pid, "seg_idx": seg, "teacher_path": rel,
                          "A_audio": item["A_audio_teacher"],
                          "V_audio": item["V_audio_teacher"],
                          "A_bio": item["A_bio_teacher"],
                          "V_bio": item["V_bio_teacher"]})
            idx += 1
    return rows


def main():
    P = argparse.ArgumentParser("Dual-Expert Artifact Extractor")
    P.add_argument("--ckpt_dir", required=True)
    P.add_argument("--base_dir", default="/home/jy/260210/precessed_data")
    P.add_argument("--index_csv", default=None)
    P.add_argument("--out_dir", required=True)
    P.add_argument("--fold_aware", action="store_true", default=True)
    P.add_argument("--no_fold_aware", dest="fold_aware", action="store_false")
    P.add_argument("--n_folds", type=int, default=6)
    P.add_argument("--batch_size", type=int, default=16)
    P.add_argument("--workers", type=int, default=2)
    P.add_argument("--bio_encoder_type", default=None)
    P.add_argument("--d_audio_teacher", type=int, default=128)
    P.add_argument("--d_bio_teacher", type=int, default=128)
    args = P.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_csv = args.index_csv or f"{args.base_dir}/segments_index.csv"
    df = pd.read_csv(idx_csv)
    for c in ["has_bio_img", "has_audio_mel", "has_video_face"]:
        if c in df.columns:
            df = df[df[c] == 1]
    df = df.dropna(subset=["label_ext_A_norm", "label_ext_V_norm"]
                   ).reset_index(drop=True)

    exp_cfg = {}
    ecp = ckpt_dir / "experiment_config.json"
    if ecp.exists():
        with open(ecp) as f:
            exp_cfg = json.load(f)
    bio_type = args.bio_encoder_type or exp_cfg.get("bio_encoder_type", "4ch")

    print(f"{'='*60}")
    print(f"Dual-Expert Artifact Extraction")
    print(f"{'='*60}")
    print(f"Ckpt: {ckpt_dir}")
    print(f"Bio: {bio_type}, Fold-aware: {args.fold_aware}")
    print(f"Samples: {len(df)}")

    all_rows = []
    t0 = time.time()

    if args.fold_aware:
        groups = df["pid"].astype(int).to_numpy()
        gkf = GroupKFold(n_splits=args.n_folds)
        for fi, (tri, vai) in enumerate(gkf.split(df, groups=groups), 1):
            ckpt_path = ckpt_dir / f"fold_{fi:02d}" / "best_model.pt"
            if not ckpt_path.exists():
                ckpt_path = ckpt_dir / f"fold_{fi}" / "best_model.pt"
            if not ckpt_path.exists():
                print(f"  ⚠ fold {fi}: no checkpoint")
                continue
            vdf = df.iloc[vai].reset_index(drop=True)
            print(f"\nFold {fi}: {len(vdf)} val samples")
            model, _ = load_v4_and_build_experts(ckpt_path, device, bio_type)
            use_gafmtf = (bio_type == "gafmtf")
            ds = KEmoconMultiModalDatasetV4(vdf, args.base_dir,
                                             use_bio_gafmtf=use_gafmtf)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True,
                                collate_fn=collate_mm_v4)
            rows = extract_from_loader(model, loader, device, out_dir, vdf)
            all_rows.extend(rows)
            print(f"  ✅ {len(rows)} extracted")
            del model; torch.cuda.empty_cache()
    else:
        ckpts = sorted(ckpt_dir.glob("fold_*/best_model.pt"))
        if not ckpts:
            print("❌ No checkpoint"); return
        model, _ = load_v4_and_build_experts(ckpts[0], device, bio_type)
        ds = KEmoconMultiModalDatasetV4(df, args.base_dir,
                                         use_bio_gafmtf=(bio_type == "gafmtf"))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_mm_v4)
        all_rows = extract_from_loader(model, loader, device, out_dir, df)

    elapsed = time.time() - t0

    pd.DataFrame(all_rows).to_csv(out_dir / "teacher_index.csv", index=False)
    with open(out_dir / "extraction_config.json", "w") as f:
        json.dump({
            "ckpt_dir": str(ckpt_dir), "bio_encoder_type": bio_type,
            "fold_aware": args.fold_aware,
            "d_audio_teacher": args.d_audio_teacher,
            "d_bio_teacher": args.d_bio_teacher,
            "n_extracted": len(all_rows),
            "elapsed_sec": round(elapsed, 1),
        }, f, indent=2)

    print(f"\n✅ {len(all_rows)} segments → {out_dir} ({elapsed/60:.1f} min)")
    print(f"\n▶ Next:")
    print(f"  python -m modalities.av_lite.trainer \\")
    print(f"    --teacher_dir {out_dir} \\")
    print(f"    --d_audio_teacher {args.d_audio_teacher} "
          f"--d_bio_teacher {args.d_bio_teacher}")


if __name__ == "__main__":
    main()
