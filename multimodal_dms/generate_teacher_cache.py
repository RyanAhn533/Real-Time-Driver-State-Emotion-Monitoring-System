#!/usr/bin/env python3
"""
Generate Teacher Cache for KD (Knowledge Distillation)
======================================================
Trains the best-performing KMERFusion model (no_mask config)
on ALL data, then caches fused_repr + logits for student training.

Usage:
    python generate_teacher_cache.py
    python generate_teacher_cache.py --config masked   # with validity mask
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from fusion.kmer_fusion import KMERFusion, build_valid_mask
from fusion.losses import KMERLoss
from kd.teacher_cache import TeacherCache
from train_kmer import KMERDataset, train_neural, build_dataloaders

FEATURES_V2 = Path(__file__).parent / "features" / "kemocon_features_v2.npz"
CACHE_PATH = Path(__file__).parent / "features" / "teacher_cache.pt"


def generate_cache_cv(dataset, device="cuda", epochs=100, n_folds=6):
    """
    Generate teacher cache using cross-validation predictions.
    Each segment's cache entry comes from a model that never saw it during training.
    This prevents overfitting in the teacher cache.
    """
    cache = TeacherCache(dataset.N, d_model=64)

    groups = dataset.pid
    gkf = GroupKFold(n_splits=n_folds)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(
            np.arange(dataset.N), groups=groups)):
        print(f"\n  Fold {fold+1}/{n_folds}: train={len(train_idx)}, val={len(val_idx)}")

        # Build dataloaders (no mask = best config)
        train_loader, val_loader = build_dataloaders(
            dataset, train_idx, val_idx, batch_size=64,
            device=device, disable_mask=True,
        )

        # Train model
        model = KMERFusion(
            d_model=64, n_heads=4,
            use_valence=False, use_drowsy=False,
        ).to(device)
        loss_fn = KMERLoss(use_valence=False, use_drowsy=False)

        best_uar, best_state, _ = train_neural(
            model, loss_fn, train_loader, val_loader, device,
            lr=1e-3, epochs=epochs,
        )
        print(f"    Fold {fold+1} best UAR: {best_uar:.2f}%")

        # Load best state and predict on validation set
        model.load_state_dict(best_state)
        model.eval()

        # Get val tokens
        tokens = dataset.get_standardized_tokens(val_idx, fit_indices=train_idx)
        token_tensors = {k: torch.from_numpy(v).float().to(device) for k, v in tokens.items()}

        # Build mask (all-ones for no_mask config)
        mask = torch.ones(len(val_idx), 15, dtype=torch.bool).to(device)

        with torch.no_grad():
            batch_size = 128
            for start in range(0, len(val_idx), batch_size):
                end = min(start + batch_size, len(val_idx))
                batch_feat = {k: v[start:end] for k, v in token_tensors.items()}
                batch_mask = mask[start:end]

                outputs = model(batch_feat, valid_mask=batch_mask)

                for i in range(end - start):
                    global_idx = val_idx[start + i]
                    cache.store(
                        idx=global_idx,
                        fused_repr=outputs["fused_repr"][i].cpu().numpy(),
                        arousal_logit=outputs["arousal"][i, 0].cpu().item(),
                        kfer_probs=dataset.tokens["kfer_probs"][global_idx],
                    )

    # Verify coverage
    n_valid = cache.valid.sum()
    print(f"\n  Cache coverage: {n_valid}/{dataset.N} ({n_valid/dataset.N*100:.1f}%)")

    return cache


def main():
    parser = argparse.ArgumentParser(description="Generate Teacher Cache for KD")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--output", default=str(CACHE_PATH))
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dataset
    print(f"Loading features: {FEATURES_V2}")
    dataset = KMERDataset(str(FEATURES_V2))
    print(f"  N={dataset.N}, A_balance: {dataset.label_A.mean():.3f}")

    # Generate cache using CV predictions (prevents overfitting)
    print("\nGenerating teacher cache (CV mode)...")
    cache = generate_cache_cv(dataset, device=device,
                               epochs=args.epochs, n_folds=args.n_folds)

    # Save
    cache.save(args.output)

    # Summary statistics
    print(f"\nTeacher cache statistics:")
    valid_reprs = cache.fused_repr[cache.valid]
    norms = np.linalg.norm(valid_reprs.astype(np.float32), axis=1)
    print(f"  Repr norms: mean={norms.mean():.4f}, std={norms.std():.4f}")
    print(f"  Arousal logits: mean={cache.arousal_logit[cache.valid].mean():.4f}, "
          f"std={cache.arousal_logit[cache.valid].std():.4f}")
    print(f"  File size: {Path(args.output).stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
