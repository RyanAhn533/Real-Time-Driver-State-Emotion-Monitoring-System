#!/usr/bin/env python3
"""
K-MER Student Training with Knowledge Distillation
====================================================
Distill KMERFusion teacher → lightweight Jetson student.

Teacher: KMERFusion (15 tokens, Pool-FFN + MHSA, ~97K params)
Student: KMERStudent (face + bio + audio MLPs, ~12K params)

KD Loss = 0.5×FocalCE(arousal) + 0.3×MSE(fused_repr) + 0.2×KL(arousal_logit)

Usage:
    python train_kd.py                  # Full 6-fold CV training
    python train_kd.py --no-kd          # Student without KD (baseline)
    python train_kd.py --ablation all   # Run all ablations
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fusion.losses import BinaryFocalLoss, KDLoss
from kd.teacher_cache import TeacherCache

# ── Paths ──
FEATURES_V2 = Path(__file__).parent / "features" / "kemocon_features_v2.npz"
TEACHER_CACHE = Path(__file__).parent / "features" / "teacher_cache.pt"
RESULT_DIR = Path(__file__).parent / "results_kd"


# ══════════════════════════════════════════════════════════════
# Student Model
# ══════════════════════════════════════════════════════════════

class KMERStudent(nn.Module):
    """
    Lightweight student for Jetson Orin deployment.

    Architecture:
      Face path:  kfer_probs(7) + kfer_meta(2) + face_stats(3) = 12d → 64d
      Bio path:   bvp(4) + eda(5) + hr_temp(6) + bio_quality(3) = 18d → 64d
      Audio path: emo2vec_probs(9) + audeering_avd(3) + audio_quality(3) = 15d → 64d
      Fusion:     concat(192d) → d_repr(64d)
      Heads:      arousal (64→1, sigmoid)

    Total: ~12K params (vs teacher's 97K)
    Jetson note: face/bio/audio encoders correspond to what Jetson sensors produce.
    """

    def __init__(self, d_repr: int = 64, dropout: float = 0.1,
                 use_audio: bool = True):
        super().__init__()
        self.d_repr = d_repr
        self.use_audio = use_audio

        # Face path: kfer_probs(7) + kfer_meta(2) + face_stats(3) = 12d
        self.face_encoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
        )

        # Bio path: bvp(4) + eda(5) + hr_temp(6) + bio_quality(3) = 18d
        self.bio_encoder = nn.Sequential(
            nn.Linear(18, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
        )

        # Audio path: emo2vec(9) + audeering(3) + quality(3) = 15d
        if use_audio:
            self.audio_encoder = nn.Sequential(
                nn.Linear(15, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 64),
            )
            fusion_dim = 192  # 64 * 3
        else:
            fusion_dim = 128  # 64 * 2

        # Gating: learn soft weights between modalities
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, fusion_dim),
            nn.Sigmoid(),
        )

        # Fusion → representation
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, d_repr),
            nn.LayerNorm(d_repr),
            nn.GELU(),
        )

        # Arousal head
        self.arousal_head = nn.Sequential(
            nn.Linear(d_repr, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, face_feat, bio_feat, audio_feat=None):
        """
        Args:
            face_feat:  (B, 12) kfer_probs + kfer_meta + face_stats
            bio_feat:   (B, 18) bvp + eda + hr_temp + bio_quality
            audio_feat: (B, 15) emo2vec_probs + audeering + quality (optional)

        Returns:
            {
                "arousal": (B, 1),
                "fused_repr": (B, d_repr),
            }
        """
        face_enc = self.face_encoder(face_feat)   # (B, 64)
        bio_enc = self.bio_encoder(bio_feat)       # (B, 64)

        if self.use_audio and audio_feat is not None:
            audio_enc = self.audio_encoder(audio_feat)  # (B, 64)
            concat = torch.cat([face_enc, bio_enc, audio_enc], dim=1)  # (B, 192)
        else:
            concat = torch.cat([face_enc, bio_enc], dim=1)  # (B, 128)

        # Gated fusion
        gate_w = self.gate(concat)
        gated = concat * gate_w

        fused_repr = self.fusion(gated)  # (B, d_repr)
        arousal = self.arousal_head(fused_repr)  # (B, 1)

        return {
            "arousal": arousal,
            "fused_repr": fused_repr,
        }


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

class KDDataset(Dataset):
    """Dataset for KD student training."""

    def __init__(self, npz_path: str, teacher_cache: TeacherCache = None):
        data = np.load(npz_path, allow_pickle=True)
        self.N = len(data["pid"])

        # Labels
        self.label_A = data["label_A_bin"].astype(np.float32)
        self.label_V = data["label_V_bin"].astype(np.float32)
        self.pid = data["pid"]

        # Face features: kfer_probs(7) + kfer_meta(2) + face_stats(3) = 12d
        self.face_feat = np.hstack([
            data["kfer_probs"].astype(np.float32),              # 7
            np.stack([
                data["kfer_quality"].astype(np.float32),
                data["kfer_entropy"].astype(np.float32),
            ], axis=1),                                          # 2
            data["face_stats"].astype(np.float32),               # 3
        ])  # (N, 12)

        # Bio features: bvp(4) + eda(5) + hr_temp(6) + bio_quality(3) = 18d
        self.bio_feat = np.hstack([
            data["bio_features"][:, 0:4].astype(np.float32),    # bvp 4
            data["bio_features"][:, 4:9].astype(np.float32),    # eda 5
            data["bio_features"][:, 9:15].astype(np.float32),   # hr_temp 6
            np.stack([
                data["bvp_valid"].astype(np.float32),
                data["eda_valid"].astype(np.float32),
                data["hr_valid"].astype(np.float32),
            ], axis=1),                                          # quality 3
        ])  # (N, 18)

        # Audio features: emo2vec_probs(9) + audeering_avd(3) + audio_quality(3) = 15d
        self.audio_feat = np.hstack([
            data["emo2vec_probs"].astype(np.float32),            # 9
            data["audeering_avd"].astype(np.float32),            # 3
            data["audio_quality"].astype(np.float32),            # 3
        ])  # (N, 15)

        # Teacher cache
        self.teacher_cache = teacher_cache

        # NaN cleanup
        self.face_feat = np.nan_to_num(self.face_feat, nan=0.0, posinf=0.0, neginf=0.0)
        self.bio_feat = np.nan_to_num(self.bio_feat, nan=0.0, posinf=0.0, neginf=0.0)
        self.audio_feat = np.nan_to_num(self.audio_feat, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_stats(self, indices):
        """Compute standardization stats from training indices."""
        stats = {}
        for name, arr in [("face", self.face_feat), ("bio", self.bio_feat),
                          ("audio", self.audio_feat)]:
            stats[name] = {
                "mean": np.nanmean(arr[indices], axis=0),
                "std": np.nanstd(arr[indices], axis=0) + 1e-8,
            }
        return stats

    def get_standardized(self, indices, stats):
        """Get standardized features for indices using given stats."""
        face = (self.face_feat[indices] - stats["face"]["mean"]) / stats["face"]["std"]
        bio = (self.bio_feat[indices] - stats["bio"]["mean"]) / stats["bio"]["std"]
        audio = (self.audio_feat[indices] - stats["audio"]["mean"]) / stats["audio"]["std"]

        face = np.nan_to_num(face, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        bio = np.nan_to_num(bio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return face, bio, audio


class FoldDataset(Dataset):
    """Wrapper for one fold's data."""

    def __init__(self, face, bio, audio, labels_a, teacher_reprs=None, teacher_logits=None):
        self.face = torch.from_numpy(face)
        self.bio = torch.from_numpy(bio)
        self.audio = torch.from_numpy(audio)
        self.labels_a = torch.from_numpy(labels_a)
        self.teacher_reprs = torch.from_numpy(teacher_reprs) if teacher_reprs is not None else None
        self.teacher_logits = torch.from_numpy(teacher_logits) if teacher_logits is not None else None

    def __len__(self):
        return len(self.labels_a)

    def __getitem__(self, idx):
        item = {
            "face": self.face[idx],
            "bio": self.bio[idx],
            "audio": self.audio[idx],
            "label_a": self.labels_a[idx],
        }
        if self.teacher_reprs is not None:
            item["teacher_repr"] = self.teacher_reprs[idx]
            item["teacher_logit"] = self.teacher_logits[idx]
        return item


# ══════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════

def train_student(model, train_loader, val_loader, device,
                  lr=1e-3, epochs=100, weight_decay=0.01,
                  use_kd=True, kd_feat_w=0.3, kd_logit_w=0.2, kd_temp=4.0):
    """Train KMERStudent with optional KD."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    task_loss_fn = BinaryFocalLoss(gamma=2.0, alpha=0.5)
    kd_loss_fn = KDLoss(temperature=kd_temp, feat_weight=kd_feat_w, logit_weight=kd_logit_w)

    best_uar = 0.0
    best_state = None
    history = []

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            face = batch["face"].to(device)
            bio = batch["bio"].to(device)
            audio = batch["audio"].to(device)
            labels_a = batch["label_a"].to(device)

            outputs = model(face, bio, audio)

            # Task loss
            loss = 0.5 * task_loss_fn(outputs["arousal"], labels_a)

            # KD loss
            if use_kd and "teacher_repr" in batch:
                teacher_repr = batch["teacher_repr"].to(device)
                teacher_logit = batch["teacher_logit"].to(device)

                kd = kd_loss_fn(
                    outputs["fused_repr"], teacher_repr,
                    outputs["arousal"], teacher_logit,
                )
                loss = loss + kd

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * len(labels_a)
            n_samples += len(labels_a)

        scheduler.step()

        # ── Validate ──
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                face = batch["face"].to(device)
                bio = batch["bio"].to(device)
                audio = batch["audio"].to(device)

                outputs = model(face, bio, audio)
                preds = (outputs["arousal"].squeeze(-1) > 0.5).long().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["label_a"].numpy())

        val_uar = balanced_accuracy_score(all_labels, all_preds) * 100

        if val_uar > best_uar:
            best_uar = val_uar
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = total_loss / max(n_samples, 1)
            print(f"    Epoch {epoch+1:3d}: loss={avg_loss:.4f}  val_UAR={val_uar:.2f}%  "
                  f"best={best_uar:.2f}%")

        history.append({"epoch": epoch + 1, "train_loss": total_loss / max(n_samples, 1),
                         "val_uar": val_uar})

    return best_uar, best_state, history


# ══════════════════════════════════════════════════════════════
# Ablations
# ══════════════════════════════════════════════════════════════

ABLATIONS = {
    1: {
        "name": "Student_no_KD",
        "desc": "Student trained without KD (task loss only)",
        "use_kd": False,
        "use_audio": True,
    },
    2: {
        "name": "Student_KD_full",
        "desc": "Student + KD (0.5×CE + 0.3×MSE + 0.2×KL) ★",
        "use_kd": True,
        "use_audio": True,
        "kd_feat_w": 0.3,
        "kd_logit_w": 0.2,
    },
    3: {
        "name": "Student_KD_feat_only",
        "desc": "Student + feature KD only (MSE repr)",
        "use_kd": True,
        "use_audio": True,
        "kd_feat_w": 0.5,
        "kd_logit_w": 0.0,
    },
    4: {
        "name": "Student_KD_logit_only",
        "desc": "Student + logit KD only (KL arousal)",
        "use_kd": True,
        "use_audio": True,
        "kd_feat_w": 0.0,
        "kd_logit_w": 0.5,
    },
    5: {
        "name": "Student_no_audio",
        "desc": "Student + KD, no audio path (face+bio only)",
        "use_kd": True,
        "use_audio": False,
        "kd_feat_w": 0.3,
        "kd_logit_w": 0.2,
    },
    6: {
        "name": "Student_heavy_KD",
        "desc": "Student + heavy KD (0.3×CE + 0.4×MSE + 0.3×KL)",
        "use_kd": True,
        "use_audio": True,
        "kd_feat_w": 0.4,
        "kd_logit_w": 0.3,
    },
}


def run_ablation(ablation_id: int, dataset: KDDataset,
                 teacher_cache: TeacherCache,
                 device: str = "cuda", n_folds: int = 6,
                 epochs: int = 100, lr: float = 1e-3,
                 batch_size: int = 64) -> dict:
    """Run single ablation with GroupKFold."""
    config = ABLATIONS[ablation_id]
    print(f"\n{'='*60}")
    print(f"Ablation {ablation_id}: {config['name']}")
    print(f"  {config['desc']}")
    print(f"{'='*60}")

    use_kd = config.get("use_kd", True)
    use_audio = config.get("use_audio", True)
    kd_feat_w = config.get("kd_feat_w", 0.3)
    kd_logit_w = config.get("kd_logit_w", 0.2)

    groups = dataset.pid
    gkf = GroupKFold(n_splits=n_folds)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(
            np.arange(dataset.N), groups=groups)):

        print(f"\n  Fold {fold+1}/{n_folds}: train={len(train_idx)}, val={len(val_idx)}")

        # Standardize using train stats
        stats = dataset.compute_stats(train_idx)
        train_face, train_bio, train_audio = dataset.get_standardized(train_idx, stats)
        val_face, val_bio, val_audio = dataset.get_standardized(val_idx, stats)

        # Teacher cache for training data
        if use_kd and teacher_cache is not None:
            t_batch = teacher_cache.get_batch(train_idx)
            train_t_repr = t_batch["fused_repr"].numpy()
            train_t_logit = t_batch["arousal_logit"].numpy()

            t_batch_val = teacher_cache.get_batch(val_idx)
            val_t_repr = t_batch_val["fused_repr"].numpy()
            val_t_logit = t_batch_val["arousal_logit"].numpy()
        else:
            train_t_repr = None
            train_t_logit = None
            val_t_repr = None
            val_t_logit = None

        # Build datasets
        train_ds = FoldDataset(train_face, train_bio, train_audio,
                               dataset.label_A[train_idx],
                               train_t_repr, train_t_logit)
        val_ds = FoldDataset(val_face, val_bio, val_audio,
                             dataset.label_A[val_idx],
                             val_t_repr, val_t_logit)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # Build student
        model = KMERStudent(d_repr=64, dropout=0.1, use_audio=use_audio).to(device)

        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Student params: {n_params:,}")

        # Train
        uar, best_state, _ = train_student(
            model, train_loader, val_loader, device,
            lr=lr, epochs=epochs, use_kd=use_kd,
            kd_feat_w=kd_feat_w, kd_logit_w=kd_logit_w,
        )

        fold_results.append({"fold": fold + 1, "A_UAR": uar})
        print(f"    → A_UAR = {uar:.2f}%")

    # Aggregate
    mean_uar = np.mean([r["A_UAR"] for r in fold_results])
    std_uar = np.std([r["A_UAR"] for r in fold_results])

    result = {
        "ablation_id": ablation_id,
        "name": config["name"],
        "desc": config["desc"],
        "mean_A_UAR": float(mean_uar),
        "std_A_UAR": float(std_uar),
        "folds": fold_results,
    }

    print(f"\n  ★ {config['name']}: A_UAR = {mean_uar:.2f}% ± {std_uar:.2f}%")
    return result


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="K-MER Student KD Training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ablation", type=str, default="2",
                        help="Ablation ID (1-6) or 'all'")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--no-kd", action="store_true",
                        help="Disable KD (same as --ablation 1)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dataset
    print(f"Loading features: {FEATURES_V2}")
    dataset = KDDataset(str(FEATURES_V2))
    print(f"  N={dataset.N}, A_balance: {dataset.label_A.mean():.3f}")

    # Load teacher cache
    teacher_cache = None
    if TEACHER_CACHE.exists() and not args.no_kd:
        print(f"Loading teacher cache: {TEACHER_CACHE}")
        teacher_cache = TeacherCache.load(str(TEACHER_CACHE))
        n_valid = teacher_cache.valid.sum()
        print(f"  Valid: {n_valid}/{teacher_cache.n_segments}")
    else:
        print("  No teacher cache — training without KD")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    all_results = []

    if args.no_kd:
        ablation_ids = [1]
    elif args.ablation.lower() == "all":
        ablation_ids = sorted(ABLATIONS.keys())
    else:
        ablation_ids = [int(x) for x in args.ablation.split(",")]

    for aid in ablation_ids:
        result = run_ablation(
            aid, dataset, teacher_cache, device,
            n_folds=args.n_folds, epochs=args.epochs,
            lr=args.lr, batch_size=args.batch_size,
        )
        all_results.append(result)

    elapsed = time.time() - t0

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  K-MER KD STUDENT ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'ID':>3}  {'Name':<28}  {'A_UAR':>8}  {'±std':>6}")
    print(f"{'-'*50}")
    for r in all_results:
        star = " ★" if "full" in r["name"].lower() else ""
        print(f"{r['ablation_id']:>3}  {r['name']:<28}  "
              f"{r['mean_A_UAR']:>7.2f}%  ±{r['std_A_UAR']:.2f}{star}")

    # Compare with teacher
    teacher_results_path = Path(__file__).parent / "results_kmer" / "ablation_results.json"
    if teacher_results_path.exists():
        with open(teacher_results_path) as f:
            teacher_results = json.load(f)
        best_teacher = max(teacher_results, key=lambda x: x["mean_A_UAR"])
        print(f"\n  Teacher best: {best_teacher['name']} = {best_teacher['mean_A_UAR']:.2f}%")
        if all_results:
            best_student = max(all_results, key=lambda x: x["mean_A_UAR"])
            gap = best_student["mean_A_UAR"] - best_teacher["mean_A_UAR"]
            print(f"  Student best: {best_student['name']} = {best_student['mean_A_UAR']:.2f}%")
            print(f"  Gap: {gap:+.2f}%")
            ratio = best_student["mean_A_UAR"] / best_teacher["mean_A_UAR"] * 100
            print(f"  Student retains {ratio:.1f}% of teacher performance")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save results
    out_path = RESULT_DIR / "kd_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
