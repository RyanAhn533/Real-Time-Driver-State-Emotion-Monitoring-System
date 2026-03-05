#!/usr/bin/env python3
"""
K-MER Training + Ablation
===========================
Train KMERFusion on kemocon_features_v2.npz with GroupKFold(6).
8 ablation experiments for systematic evaluation.

Usage:
    python train_kmer.py                         # Run all ablations
    python train_kmer.py --ablation 5            # Run specific ablation
    python train_kmer.py --ablation 3 --device cuda
"""

import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from fusion.kmer_fusion import KMERFusion, build_valid_mask
from fusion.losses import KMERLoss
from fusion.dynamic_alpha import DynamicAlpha, alpha_rule_based

# ── Paths ──
FEATURES_V2 = Path(__file__).parent / "features" / "kemocon_features_v2.npz"
FEATURES_V1 = Path(__file__).parent / "features" / "kemocon_features.npz"
RESULT_DIR = Path(__file__).parent / "results_kmer"


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

class KMERDataset:
    """Load V2 features and prepare token-level inputs."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.N = len(data["pid"])

        # Labels
        self.label_A = data["label_A_bin"].astype(np.float32)
        self.label_V = data["label_V_bin"].astype(np.float32)
        self.pid = data["pid"]
        self.pair_id = data["pair_id"]

        # Token features (matching KMERFusion TOKEN_INPUT_DIMS)
        self.tokens = {
            "kfer_probs":       data["kfer_probs"].astype(np.float32),           # (N, 7)
            "kfer_meta":        np.stack([
                                    data["kfer_quality"].astype(np.float32),
                                    data["kfer_entropy"].astype(np.float32),
                                ], axis=1),                                       # (N, 2)
            "face_stats":       data["face_stats"].astype(np.float32),           # (N, 3)
            "emo2vec_probs":    data["emo2vec_probs"].astype(np.float32),        # (N, 9)
            "audeering_avd":    data["audeering_avd"].astype(np.float32),        # (N, 3)
            "audio_quality":    data["audio_quality"].astype(np.float32),        # (N, 3)
            "bvp_features":     data["bio_features"][:, 0:4].astype(np.float32),   # (N, 4)
            "eda_features":     data["bio_features"][:, 4:9].astype(np.float32),   # (N, 5)
            "hr_temp_features": data["bio_features"][:, 9:15].astype(np.float32),  # (N, 6)
            "bio_quality":      np.stack([
                                    data["bvp_valid"].astype(np.float32),
                                    data["eda_valid"].astype(np.float32),
                                    data["hr_valid"].astype(np.float32),
                                ], axis=1),                                       # (N, 3)
            "perclos_ear":      np.stack([
                                    data["perclos"].astype(np.float32),
                                    data["ear_mean"].astype(np.float32),
                                ], axis=1),                                       # (N, 2)
            "facs_scores":      data["facs_scores"].astype(np.float32),          # (N, 6)
            "cross_modal":      data["cross_modal"].astype(np.float32),          # (N, 3)
            "validity_flags":   data["validity_flags"].astype(np.float32),       # (N, 3)
        }

        # Validity masks
        self.face_valid = data["face_valid"].astype(bool)
        self.audio_valid = (data["emo2vec_valid"] | data["audeering_valid"]).astype(bool)
        self.bio_valid = data["bio_valid"].astype(bool)

        # K-FER specific
        self.kfer_valid = data["kfer_valid"].astype(bool)
        self.kfer_entropy = data["kfer_entropy"].astype(np.float32)

        # Standardize continuous features (fit on all data, transform per-split later)
        self._compute_stats()

    def _compute_stats(self):
        """Compute mean/std for standardization."""
        self.stats = {}
        continuous_keys = ["audeering_avd", "bvp_features", "eda_features",
                           "hr_temp_features", "cross_modal"]
        for key in continuous_keys:
            arr = self.tokens[key]
            self.stats[key] = {
                "mean": np.nanmean(arr, axis=0),
                "std": np.nanstd(arr, axis=0) + 1e-8,
            }

    def get_standardized_tokens(self, indices, fit_indices=None):
        """Get standardized token features for given indices."""
        tokens = {}
        continuous_keys = set(self.stats.keys())

        for key, arr in self.tokens.items():
            subset = arr[indices].copy()
            if key in continuous_keys:
                if fit_indices is not None:
                    mean = np.nanmean(arr[fit_indices], axis=0)
                    std = np.nanstd(arr[fit_indices], axis=0) + 1e-8
                else:
                    mean = self.stats[key]["mean"]
                    std = self.stats[key]["std"]
                subset = (subset - mean) / std
            # NaN → 0
            subset = np.nan_to_num(subset, nan=0.0, posinf=0.0, neginf=0.0)
            tokens[key] = subset

        return tokens

    def get_lgbm_features(self, indices, mode="compact_plus"):
        """Build LightGBM feature matrix (for hybrid baseline)."""
        if mode == "compact":
            # 26d: emo2vec_probs(9) + audeering(3) + bio(15) - 1 = 26
            feats = np.hstack([
                self.tokens["emo2vec_probs"][indices],    # 9
                self.tokens["audeering_avd"][indices],    # 3
                self.tokens["bvp_features"][indices],     # 4
                self.tokens["eda_features"][indices],     # 5
                self.tokens["hr_temp_features"][indices],  # 6
            ])  # total = 27
        elif mode == "compact_plus":
            # +K-FER(7) + kfer_meta(2) + face_stats(3) = 39
            feats = np.hstack([
                self.tokens["emo2vec_probs"][indices],
                self.tokens["audeering_avd"][indices],
                self.tokens["bvp_features"][indices],
                self.tokens["eda_features"][indices],
                self.tokens["hr_temp_features"][indices],
                self.tokens["kfer_probs"][indices],
                self.tokens["kfer_meta"][indices],
                self.tokens["face_stats"][indices],
            ])  # total = 39
        else:
            # All features (55d)
            feats = np.hstack([v[indices] for v in self.tokens.values()])

        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════

def train_neural(model, loss_fn, train_loader, val_loader, device,
                 lr=1e-3, epochs=100, weight_decay=0.01,
                 val_dataset_extras=None):
    """Train KMERFusion model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_uar = 0.0
    best_state = None
    history = []

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            features, mask, labels_a, labels_v = batch
            features = {k: v.to(device) for k, v in features.items()}
            mask = mask.to(device)
            labels_a = labels_a.to(device)

            outputs = model(features, valid_mask=mask)
            targets = {"arousal": labels_a}
            if "valence" in outputs and labels_v is not None:
                targets["valence"] = labels_v.to(device)

            losses = loss_fn(outputs, targets)
            loss = losses["total"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item() * labels_a.shape[0]
            n_train += labels_a.shape[0]

        scheduler.step()

        # ── Validate ──
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                features, mask, labels_a, labels_v = batch
                features = {k: v.to(device) for k, v in features.items()}
                mask = mask.to(device)

                outputs = model(features, valid_mask=mask)
                preds = (outputs["arousal"].squeeze(-1) > 0.5).long().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels_a.numpy())

        val_uar = balanced_accuracy_score(all_labels, all_preds) * 100

        if val_uar > best_uar:
            best_uar = val_uar
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = train_loss / max(n_train, 1)
            print(f"    Epoch {epoch+1:3d}: loss={avg_loss:.4f}  val_UAR={val_uar:.2f}%  "
                  f"best={best_uar:.2f}%")

        history.append({"epoch": epoch + 1, "train_loss": train_loss / max(n_train, 1),
                         "val_uar": val_uar})

    return best_uar, best_state, history


def train_lgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM baseline."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("    LightGBM not available, skipping")
        return 50.0, None

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42,
    }

    model = lgb.train(
        params, dtrain,
        num_boost_round=300,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    preds = (model.predict(X_val) > 0.5).astype(int)
    uar = balanced_accuracy_score(y_val, preds) * 100

    return uar, model


# ══════════════════════════════════════════════════════════════
# Data Loader Builder
# ══════════════════════════════════════════════════════════════

def build_dataloaders(dataset, train_idx, val_idx, batch_size=64, device="cpu",
                      disable_mask=False, remove_kfer=False, remove_bio=False):
    """Build train/val DataLoaders from dataset + index splits."""

    def make_loader(indices, shuffle, fit_idx=None):
        tokens = dataset.get_standardized_tokens(indices, fit_indices=fit_idx)

        # Apply ablation modifications
        if remove_kfer:
            tokens["kfer_probs"] = np.zeros_like(tokens["kfer_probs"])
            tokens["kfer_meta"] = np.zeros_like(tokens["kfer_meta"])
        if remove_bio:
            tokens["bvp_features"] = np.zeros_like(tokens["bvp_features"])
            tokens["eda_features"] = np.zeros_like(tokens["eda_features"])
            tokens["hr_temp_features"] = np.zeros_like(tokens["hr_temp_features"])
            tokens["bio_quality"] = np.zeros_like(tokens["bio_quality"])

        token_tensors = {k: torch.from_numpy(v).float() for k, v in tokens.items()}

        # Build validity mask
        face_v = torch.from_numpy(dataset.face_valid[indices])
        audio_v = torch.from_numpy(dataset.audio_valid[indices])
        bio_v = torch.from_numpy(dataset.bio_valid[indices])

        if remove_bio:
            bio_v = torch.zeros_like(bio_v)

        if disable_mask:
            mask = torch.ones(len(indices), 15, dtype=torch.bool)
        else:
            mask = build_valid_mask(face_v, audio_v, bio_v)

        labels_a = torch.from_numpy(dataset.label_A[indices]).float()
        labels_v = torch.from_numpy(dataset.label_V[indices]).float()

        # Custom collate for dict features
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, token_dict, masks, la, lv):
                self.token_dict = token_dict
                self.masks = masks
                self.la = la
                self.lv = lv

            def __len__(self):
                return len(self.la)

            def __getitem__(self, idx):
                return (
                    {k: v[idx] for k, v in self.token_dict.items()},
                    self.masks[idx],
                    self.la[idx],
                    self.lv[idx],
                )

        ds = DictDataset(token_tensors, mask, labels_a, labels_v)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    train_loader = make_loader(train_idx, shuffle=True, fit_idx=train_idx)
    val_loader = make_loader(val_idx, shuffle=False, fit_idx=train_idx)

    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════
# Ablation Definitions
# ══════════════════════════════════════════════════════════════

ABLATIONS = {
    1: {
        "name": "LGBM_4expert_27d",
        "desc": "Baseline LightGBM with 4 experts (no K-FER)",
        "type": "lgbm",
        "lgbm_mode": "compact",
    },
    2: {
        "name": "LGBM_5expert_39d",
        "desc": "LightGBM + K-FER (5 experts)",
        "type": "lgbm",
        "lgbm_mode": "compact_plus",
    },
    3: {
        "name": "KMERFusion_15tok",
        "desc": "Pool-FFN + MHSA neural fusion (15 tokens)",
        "type": "neural",
    },
    4: {
        "name": "Hybrid_static_alpha",
        "desc": "Neural + LightGBM, α=0.5",
        "type": "hybrid",
        "alpha_mode": "static",
    },
    5: {
        "name": "Hybrid_dynamic_alpha",
        "desc": "Neural + LightGBM, dynamic α ★",
        "type": "hybrid",
        "alpha_mode": "dynamic",
    },
    6: {
        "name": "Hybrid_no_kfer",
        "desc": "Hybrid without K-FER (contribution test)",
        "type": "hybrid",
        "alpha_mode": "static",
        "remove_kfer": True,
    },
    7: {
        "name": "Hybrid_no_bio",
        "desc": "Hybrid without Bio (contribution test)",
        "type": "hybrid",
        "alpha_mode": "static",
        "remove_bio": True,
    },
    8: {
        "name": "KMERFusion_no_mask",
        "desc": "Neural fusion without validity mask",
        "type": "neural",
        "disable_mask": True,
    },
}


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def run_ablation(ablation_id: int, dataset: KMERDataset,
                 device: str = "cuda", n_folds: int = 6,
                 epochs: int = 100, lr: float = 1e-3,
                 batch_size: int = 64) -> dict:
    """Run single ablation experiment with GroupKFold."""
    config = ABLATIONS[ablation_id]
    print(f"\n{'='*60}")
    print(f"Ablation {ablation_id}: {config['name']}")
    print(f"  {config['desc']}")
    print(f"{'='*60}")

    groups = dataset.pid
    gkf = GroupKFold(n_splits=n_folds)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(
            np.arange(dataset.N), groups=groups)):

        print(f"\n  Fold {fold+1}/{n_folds}: train={len(train_idx)}, val={len(val_idx)}")

        remove_kfer = config.get("remove_kfer", False)
        remove_bio = config.get("remove_bio", False)
        disable_mask = config.get("disable_mask", False)

        if config["type"] == "lgbm":
            # LightGBM only
            X_tr = dataset.get_lgbm_features(train_idx, config["lgbm_mode"])
            y_tr = dataset.label_A[train_idx]
            X_va = dataset.get_lgbm_features(val_idx, config["lgbm_mode"])
            y_va = dataset.label_A[val_idx]

            uar, _ = train_lgbm(X_tr, y_tr, X_va, y_va)
            fold_results.append({"fold": fold + 1, "A_UAR": uar})
            print(f"    → A_UAR = {uar:.2f}%")

        elif config["type"] == "neural":
            # Neural only
            train_loader, val_loader = build_dataloaders(
                dataset, train_idx, val_idx, batch_size, device,
                disable_mask=disable_mask, remove_kfer=remove_kfer, remove_bio=remove_bio,
            )

            model = KMERFusion(
                d_model=64, n_heads=4,
                use_valence=False, use_drowsy=False,
            ).to(device)

            loss_fn = KMERLoss(use_valence=False, use_drowsy=False)

            uar, _, _ = train_neural(
                model, loss_fn, train_loader, val_loader, device,
                lr=lr, epochs=epochs,
            )
            fold_results.append({"fold": fold + 1, "A_UAR": uar})
            print(f"    → A_UAR = {uar:.2f}%")

        elif config["type"] == "hybrid":
            # Train neural
            train_loader, val_loader = build_dataloaders(
                dataset, train_idx, val_idx, batch_size, device,
                disable_mask=disable_mask, remove_kfer=remove_kfer, remove_bio=remove_bio,
            )

            model = KMERFusion(
                d_model=64, n_heads=4,
                use_valence=False, use_drowsy=False,
            ).to(device)

            loss_fn = KMERLoss(use_valence=False, use_drowsy=False)

            neural_uar, best_state, _ = train_neural(
                model, loss_fn, train_loader, val_loader, device,
                lr=lr, epochs=epochs,
            )

            # Get neural predictions on val set
            model.load_state_dict(best_state)
            model.eval()
            neural_preds = []
            with torch.no_grad():
                for batch in val_loader:
                    features, mask, _, _ = batch
                    features = {k: v.to(device) for k, v in features.items()}
                    mask = mask.to(device)
                    outputs = model(features, valid_mask=mask)
                    neural_preds.extend(outputs["arousal"].squeeze(-1).cpu().numpy())
            neural_preds = np.array(neural_preds)

            # Train LightGBM
            lgbm_mode = "compact" if remove_kfer else "compact_plus"
            X_tr = dataset.get_lgbm_features(train_idx, lgbm_mode)
            y_tr = dataset.label_A[train_idx]
            X_va = dataset.get_lgbm_features(val_idx, lgbm_mode)
            y_va = dataset.label_A[val_idx]

            lgbm_uar, lgbm_model = train_lgbm(X_tr, y_tr, X_va, y_va)

            if lgbm_model is not None:
                lgbm_preds = lgbm_model.predict(X_va)
            else:
                lgbm_preds = np.full_like(neural_preds, 0.5)

            # Combine
            if config.get("alpha_mode") == "dynamic":
                # Rule-based dynamic alpha
                face_v = dataset.face_valid[val_idx]
                audio_v = dataset.audio_valid[val_idx]
                # Neural entropy
                p = np.clip(neural_preds, 1e-6, 1 - 1e-6)
                neural_ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
                # LGBM margin
                lgbm_margin = np.abs(lgbm_preds - 0.5) * 2

                alpha = alpha_rule_based(face_v, audio_v, neural_ent, lgbm_margin)
            else:
                alpha = np.full(len(val_idx), 0.5)

            hybrid_preds = alpha * neural_preds + (1 - alpha) * lgbm_preds
            hybrid_binary = (hybrid_preds > 0.5).astype(int)
            uar = balanced_accuracy_score(y_va, hybrid_binary) * 100

            fold_results.append({
                "fold": fold + 1,
                "A_UAR": uar,
                "neural_UAR": neural_uar,
                "lgbm_UAR": lgbm_uar,
                "alpha_mean": float(np.mean(alpha)),
                "alpha_std": float(np.std(alpha)),
            })
            print(f"    → Hybrid A_UAR={uar:.2f}% (neural={neural_uar:.2f}%, "
                  f"lgbm={lgbm_uar:.2f}%, α={np.mean(alpha):.3f}±{np.std(alpha):.3f})")

    # Aggregate
    mean_uar = np.mean([r["A_UAR"] for r in fold_results])
    std_uar = np.std([r["A_UAR"] for r in fold_results])

    result = {
        "ablation_id": ablation_id,
        "name": config["name"],
        "desc": config["desc"],
        "mean_A_UAR": mean_uar,
        "std_A_UAR": std_uar,
        "folds": fold_results,
    }

    print(f"\n  ★ {config['name']}: A_UAR = {mean_uar:.2f}% ± {std_uar:.2f}%")
    return result


def main():
    parser = argparse.ArgumentParser(description="K-MER Training + Ablation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ablation", type=int, default=0,
                        help="Run specific ablation (0=all)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-folds", type=int, default=6)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dataset
    npz_path = str(FEATURES_V2) if FEATURES_V2.exists() else str(FEATURES_V1)
    print(f"Loading features: {npz_path}")
    dataset = KMERDataset(npz_path)
    print(f"  N={dataset.N}, A_balance: {dataset.label_A.mean():.3f}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    all_results = []

    if args.ablation > 0:
        # Run specific ablation
        result = run_ablation(
            args.ablation, dataset, device,
            n_folds=args.n_folds, epochs=args.epochs,
            lr=args.lr, batch_size=args.batch_size,
        )
        all_results.append(result)
    else:
        # Run all ablations
        for aid in sorted(ABLATIONS.keys()):
            result = run_ablation(
                aid, dataset, device,
                n_folds=args.n_folds, epochs=args.epochs,
                lr=args.lr, batch_size=args.batch_size,
            )
            all_results.append(result)

    elapsed = time.time() - t0

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  K-MER ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'ID':>3}  {'Name':<28}  {'A_UAR':>8}  {'±std':>6}")
    print(f"{'-'*50}")
    for r in all_results:
        star = " ★" if "dynamic" in r["name"].lower() else ""
        print(f"{r['ablation_id']:>3}  {r['name']:<28}  "
              f"{r['mean_A_UAR']:>7.2f}%  ±{r['std_A_UAR']:.2f}{star}")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save results
    out_path = RESULT_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
