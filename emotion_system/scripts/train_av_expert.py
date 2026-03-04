#!/usr/bin/env python3
"""
A/V Expert v2 Training: emotion2vec + Bio Features → A/V Prediction
======================================================================
Trains the lightweight fusion head on pre-extracted features.

Prerequisites:
  1. pip install funasr soundfile scipy
  2. Run feature extraction:
     python -c "
     from models.av_expert import extract_kemocon_features
     extract_kemocon_features(
         base_dir='/home/jy/260210/precessed_data',
         index_csv='/home/jy/260210/precessed_data/segments_index.csv',
         out_dir='pipeline_output/av_features',
     )
     "
  3. Then run this script:
     python scripts/train_av_expert.py \
         --features_dir pipeline_output/av_features \
         --out_dir pipeline_output/av_expert_ckpt

Training details:
  - Input: [768] emotion2vec feats + [15] bio feats
  - Head: ~15k learnable parameters
  - Loss: CCC loss for A/V + CE for emotion quadrant
  - K-fold CV with GroupKFold by participant ID
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold

from models.av_expert import AVExpertV2


class CCCLoss(nn.Module):
    """Concordance Correlation Coefficient loss. L = 1 - CCC."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.mean()
        target_mean = target.mean()
        pred_var = pred.var()
        target_var = target.var()
        covar = ((pred - pred_mean) * (target - target_mean)).mean()
        ccc = (2 * covar) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)
        return 1 - ccc


class AVFeatureDataset(Dataset):
    """Pre-extracted emotion2vec + bio features for A/V Expert training."""

    def __init__(self, features_dir: str):
        self.features_dir = Path(features_dir)

        self.audio_feats = np.load(str(self.features_dir / "audio_feats.npy"))
        self.bio_feats = np.load(str(self.features_dir / "bio_feats.npy"))
        self.labels = np.load(str(self.features_dir / "labels.npy"))

        # Participant IDs for GroupKFold
        meta_path = self.features_dir / "pids.npy"
        if meta_path.exists():
            self.pids = np.load(str(meta_path))
        else:
            self.pids = np.zeros(len(self.labels), dtype=np.int64)

        print(f"[AVDataset] {len(self)} samples loaded")
        print(f"  Audio feats: {self.audio_feats.shape}")
        print(f"  Bio feats: {self.bio_feats.shape}")
        print(f"  Labels (A/V): {self.labels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "audio": torch.tensor(self.audio_feats[idx], dtype=torch.float32),
            "bio": torch.tensor(self.bio_feats[idx], dtype=torch.float32),
            "av": torch.tensor(self.labels[idx], dtype=torch.float32),
            "pid": int(self.pids[idx]),
        }


def collate_av(batch):
    return {
        "audio": torch.stack([b["audio"] for b in batch]),
        "bio": torch.stack([b["bio"] for b in batch]),
        "av": torch.stack([b["av"] for b in batch]),
    }


def compute_ccc(pred, target):
    """Compute CCC numpy."""
    pred_m = pred.mean()
    target_m = target.mean()
    pred_v = pred.var()
    target_v = target.var()
    covar = np.mean((pred - pred_m) * (target - target_m))
    return (2 * covar) / (pred_v + target_v + (pred_m - target_m) ** 2 + 1e-8)


def train_one_fold(model, train_loader, val_loader, device,
                   epochs=100, lr=1e-3):
    """Train A/V Expert for one fold. Returns best CCC."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader), eta_min=1e-6)

    ccc_loss = CCCLoss()
    best_score = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            audio = batch["audio"].to(device)
            bio = batch["bio"].to(device)
            av_gt = batch["av"].to(device)

            pred = model(audio_feats=audio, bio_feats=bio)

            loss_a = ccc_loss(pred["arousal"], av_gt[:, 0])
            loss_v = ccc_loss(pred["valence"], av_gt[:, 1])
            loss = loss_a + loss_v

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        all_pred_a, all_pred_v = [], []
        all_gt_a, all_gt_v = [], []
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                bio = batch["bio"].to(device)
                av_gt = batch["av"]

                pred = model(audio_feats=audio, bio_feats=bio)
                all_pred_a.append(pred["arousal"].cpu().numpy())
                all_pred_v.append(pred["valence"].cpu().numpy())
                all_gt_a.append(av_gt[:, 0].numpy())
                all_gt_v.append(av_gt[:, 1].numpy())

        pred_a = np.concatenate(all_pred_a)
        pred_v = np.concatenate(all_pred_v)
        gt_a = np.concatenate(all_gt_a)
        gt_v = np.concatenate(all_gt_v)

        ccc_a = compute_ccc(pred_a, gt_a)
        ccc_v = compute_ccc(pred_v, gt_v)
        score = (ccc_a + ccc_v) / 2

        if epoch % 10 == 0:
            print(f"  [E{epoch:03d}] CCC_A={ccc_a:.4f} CCC_V={ccc_v:.4f} "
                  f"avg={score:.4f}")

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_score, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--out_dir", default="pipeline_output/av_expert_ckpt")
    ap.add_argument("--n_folds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = AVFeatureDataset(args.features_dir)

    # GroupKFold by participant
    groups = dataset.pids
    gkf = GroupKFold(n_splits=args.n_folds)

    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(
            np.arange(len(dataset)), groups=groups), 1):

        print(f"\n{'='*50}")
        print(f"Fold {fold_idx}/{args.n_folds}")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=args.batch_size,
            shuffle=True, num_workers=2, collate_fn=collate_av)
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=args.batch_size,
            shuffle=False, num_workers=2, collate_fn=collate_av)

        model = AVExpertV2(
            num_classes=7,
            audio_feat_dim=768,
            bio_feat_dim=15,
            d_proj=128,
            dropout=0.3,
        ).to(device)

        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        score, best_state = train_one_fold(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr)

        fold_scores.append(score)
        print(f"  Fold {fold_idx} best CCC avg: {score:.4f}")

        # Save fold checkpoint
        fold_dir = out_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)
        torch.save({
            "model_state": best_state,
            "fold": fold_idx,
            "score": score,
        }, fold_dir / "best.pth")

    print(f"\n{'='*50}")
    print(f"CV Results:")
    for i, s in enumerate(fold_scores, 1):
        print(f"  Fold {i}: {s:.4f}")
    print(f"  Mean: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")


if __name__ == "__main__":
    main()
