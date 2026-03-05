"""
Late Fusion Training + Evaluation on Pre-extracted Features
============================================================
3가지 fusion 방법 비교:
  1. Simple Average (학습 불필요)
  2. Stacking MLP (PyTorch, ~100k params)
  3. LightGBM Stacking (자동 하이퍼파라미터)

+ Ablation: 각 모달리티 빼고 비교
+ 비교: V4, majority baseline, 논문 baseline

Usage:
  python train_fusion.py
  python train_fusion.py --feature_path features/kemocon_features.npz
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════
def load_features(path):
    data = dict(np.load(path))
    return data


def build_feature_matrix(data, mode="all"):
    """
    Build feature matrix from pre-extracted features.
    mode: "all", "compact", "compact_plus", "face_only", "audio_only",
          "bio_only", "no_face", "no_audio", "no_bio", "probs_only"

    "compact"      = face_probs + audeering_avd + bio (NO emo2vec)
    "compact_plus" = face_probs + emo2vec_probs(9d) + audeering_avd + bio (NO emo2vec embed)
    "all"          = face_probs + emo2vec_embed(1024d) + emo2vec_probs + audeering_avd + bio
    """
    parts = []
    names = []

    use_face = mode in ("all", "compact", "compact_plus", "face_only", "no_audio", "no_bio", "probs_only")
    use_audio = mode in ("all", "compact", "compact_plus", "audio_only", "no_face", "no_bio", "probs_only")
    use_bio = mode in ("all", "compact", "compact_plus", "bio_only", "no_face", "no_audio")

    if use_face:
        parts.append(data["face_probs"])  # (N, 8)
        names.extend([f"face_p{i}" for i in range(8)])
        parts.append(data["face_valid"].reshape(-1, 1).astype(np.float32))
        names.append("face_valid")

    if use_audio:
        if mode == "compact":
            # compact: skip all emo2vec features
            parts.append(data["audeering_avd"])  # (N, 3)
            names.extend(["aud_arousal", "aud_valence", "aud_dominance"])
        elif mode in ("compact_plus", "probs_only"):
            # compact_plus: add emo2vec probs (9d) but NOT 1024d embedding
            emo2vec_valid_ratio = data["emo2vec_valid"].mean() if "emo2vec_valid" in data else 0
            if emo2vec_valid_ratio > 0.01:
                probs_dim = data["emo2vec_probs"].shape[1]
                parts.append(data["emo2vec_probs"])  # (N, 9)
                names.extend([f"e2v_p{i}" for i in range(probs_dim)])
                parts.append(data["emo2vec_valid"].reshape(-1, 1).astype(np.float32))
                names.append("emo2vec_valid")
            parts.append(data["audeering_avd"])  # (N, 3)
            names.extend(["aud_arousal", "aud_valence", "aud_dominance"])
        else:
            # "all" includes full emo2vec embedding + probs
            emo2vec_valid_ratio = data["emo2vec_valid"].mean() if "emo2vec_valid" in data else 0
            if emo2vec_valid_ratio > 0.01:
                embed_dim = data["emo2vec_embed"].shape[1]
                parts.append(data["emo2vec_embed"])  # (N, 1024)
                names.extend([f"e2v_{i}" for i in range(embed_dim)])
                probs_dim = data["emo2vec_probs"].shape[1]
                parts.append(data["emo2vec_probs"])  # (N, 9)
                names.extend([f"e2v_p{i}" for i in range(probs_dim)])
                parts.append(data["emo2vec_valid"].reshape(-1, 1).astype(np.float32))
                names.append("emo2vec_valid")
            parts.append(data["audeering_avd"])  # (N, 3)
            names.extend(["aud_arousal", "aud_valence", "aud_dominance"])
        parts.append(data["audeering_valid"].reshape(-1, 1).astype(np.float32))
        names.append("audio_valid")

    if use_bio:
        parts.append(data["bio_features"])  # (N, 15)
        from experts.bio_expert import BIO_FEATURE_NAMES
        names.extend(BIO_FEATURE_NAMES)
        parts.append(data["bio_valid"].reshape(-1, 1).astype(np.float32))
        names.append("bio_valid")

    X = np.concatenate(parts, axis=1)
    return X, names


# ═══════════════════════════════════════════
# Fusion Method 1: Simple Average
# ═══════════════════════════════════════════
def simple_average_predict(data, target="A"):
    """
    Rule-based: average expert A/V predictions → binary threshold
    """
    N = len(data["label_A_bin"])
    votes = np.zeros(N, dtype=np.float32)
    weights = np.zeros(N, dtype=np.float32)

    # audeering gives A/V directly
    if target == "A":
        aud_pred = data["audeering_avd"][:, 0]  # arousal
    else:
        aud_pred = data["audeering_avd"][:, 1]  # valence

    aud_valid = data["audeering_valid"]
    votes += aud_pred * aud_valid
    weights += aud_valid

    # HSEmotion → emotion-to-AV mapping
    # Happy/Surprise → high V, Angry/Fear → high A, Neutral/Sad → low A
    if target == "A":
        face_arousal = (
            data["face_probs"][:, 0] * 0.8 +   # Angry → high A
            data["face_probs"][:, 3] * 0.7 +   # Fear → high A
            data["face_probs"][:, 4] * 0.3 +   # Happy → mid A
            data["face_probs"][:, 5] * (-0.5) + # Neutral → low A
            data["face_probs"][:, 6] * (-0.3) + # Sad → low A
            data["face_probs"][:, 7] * 0.6      # Surprise → high A
        )
        face_pred = face_arousal
    else:
        face_valence = (
            data["face_probs"][:, 0] * (-0.7) + # Angry → low V
            data["face_probs"][:, 3] * (-0.6) + # Fear → low V
            data["face_probs"][:, 4] * 0.8 +    # Happy → high V
            data["face_probs"][:, 5] * 0.0 +    # Neutral → mid V
            data["face_probs"][:, 6] * (-0.5) + # Sad → low V
            data["face_probs"][:, 7] * 0.2       # Surprise → mid V
        )
        face_pred = face_valence

    face_valid = data["face_valid"]
    votes += face_pred * face_valid
    weights += face_valid

    avg = np.divide(votes, weights, out=np.zeros_like(votes), where=weights > 0)
    return (avg >= 0).astype(int)


# ═══════════════════════════════════════════
# Fusion Method 2: Stacking MLP (PyTorch)
# ═══════════════════════════════════════════
def train_mlp_fold(X_train, y_train, X_val, y_val, input_dim, pos_weight=1.0,
                   hidden=128, epochs=50, lr=1e-3, dropout=0.3):
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class FusionMLP(nn.Module):
        def __init__(self, in_dim, hidden, dropout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = FusionMLP(input_dim, hidden, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pw = torch.tensor([pos_weight], device=device)

    X_tr = torch.from_numpy(X_train).float().to(device)
    y_tr = torch.from_numpy(y_train).float().to(device)
    X_va = torch.from_numpy(X_val).float().to(device)

    best_f1 = -1
    best_preds = None

    for ep in range(epochs):
        model.train()
        logits = model(X_tr)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tr, pos_weight=pw)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va)
            val_preds = (val_logits >= 0).long().cpu().numpy()
            f1 = f1_score(y_val, val_preds, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_preds = val_preds.copy()

    return best_preds


# ═══════════════════════════════════════════
# Fusion Method 3: LightGBM Stacking
# ═══════════════════════════════════════════
def train_lgbm_fold(X_train, y_train, X_val, scale_pos_weight=1.0, tuned=False):
    import lightgbm as lgb
    if tuned:
        model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.02,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=-1,
            n_jobs=4,
        )
    else:
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=-1,
            n_jobs=4,
        )
    model.fit(X_train, y_train)
    return model.predict(X_val), model


# ═══════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════
def evaluate_binary(y_true, y_pred, name=""):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    uar = recall_score(y_true, y_pred, average="macro", zero_division=0)
    return {"acc": acc, "f1": f1, "uar": uar}


def run_cv(data, method="lgbm", feature_mode="all", n_splits=6, seed=42):
    """Run GroupKFold CV with specified fusion method and feature mode."""
    X, feat_names = build_feature_matrix(data, mode=feature_mode)
    y_A = data["label_A_bin"]
    y_V = data["label_V_bin"]
    pair_ids = data["pair_id"]

    gkf = GroupKFold(n_splits=n_splits)
    results_A, results_V = [], []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, groups=pair_ids)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        yA_tr, yA_va = y_A[tr_idx], y_A[va_idx]
        yV_tr, yV_va = y_V[tr_idx], y_V[va_idx]

        # Normalize
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)

        # Replace NaN
        X_tr = np.nan_to_num(X_tr, 0)
        X_va = np.nan_to_num(X_va, 0)

        if method in ("lgbm", "lgbm_tuned"):
            tuned = (method == "lgbm_tuned")
            pw_A = max(np.sum(yA_tr == 0), 1) / max(np.sum(yA_tr == 1), 1)
            pw_V = max(np.sum(yV_tr == 0), 1) / max(np.sum(yV_tr == 1), 1)
            predA, _ = train_lgbm_fold(X_tr, yA_tr, X_va, scale_pos_weight=pw_A, tuned=tuned)
            predV, _ = train_lgbm_fold(X_tr, yV_tr, X_va, scale_pos_weight=pw_V, tuned=tuned)

        elif method == "mlp":
            pw_A = max(np.sum(yA_tr == 0), 1) / max(np.sum(yA_tr == 1), 1)
            pw_V = max(np.sum(yV_tr == 0), 1) / max(np.sum(yV_tr == 1), 1)
            predA = train_mlp_fold(X_tr, yA_tr, X_va, yA_va, X_tr.shape[1], pos_weight=pw_A)
            predV = train_mlp_fold(X_tr, yV_tr, X_va, yV_va, X_tr.shape[1], pos_weight=pw_V)

        elif method == "simple_avg":
            # Use all data for simple average (no training)
            predA = simple_average_predict(
                {k: v[va_idx] if isinstance(v, np.ndarray) and len(v) == len(X) else v
                 for k, v in data.items()}, target="A")
            predV = simple_average_predict(
                {k: v[va_idx] if isinstance(v, np.ndarray) and len(v) == len(X) else v
                 for k, v in data.items()}, target="V")

        results_A.append(evaluate_binary(yA_va, predA))
        results_V.append(evaluate_binary(yV_va, predV))

    return results_A, results_V


def summarize_cv(results_list, name=""):
    """Summarize CV results."""
    metrics = defaultdict(list)
    for r in results_list:
        for k, v in r.items():
            metrics[k].append(v)

    summary = {}
    for k, vals in metrics.items():
        arr = np.array(vals)
        summary[k] = f"{arr.mean()*100:.1f}±{arr.std()*100:.1f}%"
        summary[f"{k}_mean"] = float(arr.mean())
        summary[f"{k}_std"] = float(arr.std())
    return summary


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", default="features/kemocon_features.npz")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--n_splits", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading features...")
    data = load_features(args.feature_path)
    N = len(data["label_A_bin"])

    print(f"Samples: {N}")
    print(f"Face valid: {data['face_valid'].sum()} ({data['face_valid'].mean()*100:.1f}%)")
    print(f"Audio valid: {data['audeering_valid'].sum()} ({data['audeering_valid'].mean()*100:.1f}%)")
    print(f"Bio valid: {data['bio_valid'].sum()} ({data['bio_valid'].mean()*100:.1f}%)")
    print(f"A_binary: 0={np.sum(data['label_A_bin']==0)}, 1={np.sum(data['label_A_bin']==1)}")
    print(f"V_binary: 0={np.sum(data['label_V_bin']==0)}, 1={np.sum(data['label_V_bin']==1)}")

    # ── Majority Baseline ──
    maj_A = int(np.bincount(data["label_A_bin"]).argmax())
    maj_V = int(np.bincount(data["label_V_bin"]).argmax())
    maj_A_acc = float(np.mean(data["label_A_bin"] == maj_A))
    maj_V_acc = float(np.mean(data["label_V_bin"] == maj_V))

    all_results = {}
    all_results["majority"] = {
        "A": {"acc": f"{maj_A_acc*100:.1f}%", "f1": "0.0%", "uar": "50.0%"},
        "V": {"acc": f"{maj_V_acc*100:.1f}%", "f1": "0.0%", "uar": "50.0%"},
    }

    # ── Run Experiments ──
    experiments = [
        # (name, method, feature_mode)
        ("LGBM_compact", "lgbm", "compact"),           # face_probs+audeering+bio (26d)
        ("LGBM_compact_plus", "lgbm", "compact_plus"), # ★ +emo2vec_probs(9d) → 36d
        ("LGBM_compact_tuned", "lgbm_tuned", "compact"),
        ("LGBM_cplus_tuned", "lgbm_tuned", "compact_plus"),  # ★ tuned + emo2vec probs
        ("LGBM_all", "lgbm", "all"),                   # full emo2vec 1024d (overfits)
        ("LGBM_probs", "lgbm", "probs_only"),
        ("LGBM_bio_only", "lgbm", "bio_only"),
        ("LGBM_audio_only", "lgbm", "audio_only"),
        ("LGBM_face_only", "lgbm", "face_only"),
        ("LGBM_no_face", "lgbm", "no_face"),
        ("LGBM_no_audio", "lgbm", "no_audio"),
        ("LGBM_no_bio", "lgbm", "no_bio"),
        ("MLP_compact", "mlp", "compact"),
        ("MLP_compact_plus", "mlp", "compact_plus"),   # MLP with emo2vec probs
        ("MLP_probs", "mlp", "probs_only"),
        ("SimpleAvg", "simple_avg", "all"),
    ]

    for exp_name, method, feat_mode in experiments:
        print(f"\n{'─'*50}")
        print(f"Running: {exp_name} (method={method}, features={feat_mode})")
        print(f"{'─'*50}")

        try:
            rA, rV = run_cv(data, method=method, feature_mode=feat_mode, n_splits=args.n_splits)
            sA = summarize_cv(rA)
            sV = summarize_cv(rV)

            all_results[exp_name] = {"A": sA, "V": sV}

            print(f"  Arousal — Acc: {sA['acc']}, F1: {sA['f1']}, UAR: {sA['uar']}")
            print(f"  Valence — Acc: {sV['acc']}, F1: {sV['f1']}, UAR: {sV['uar']}")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ── Print Comparison Table ──
    print(f"\n{'='*80}")
    print("COMPARISON TABLE — Binary Classification (6-fold GroupKFold by pair_id)")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'A_Acc':>10} {'A_F1':>10} {'A_UAR':>10} {'V_Acc':>10} {'V_F1':>10} {'V_UAR':>10}")
    print(f"{'─'*85}")

    for name, res in all_results.items():
        a = res.get("A", {})
        v = res.get("V", {})
        print(f"{name:<25} {a.get('acc','—'):>10} {a.get('f1','—'):>10} {a.get('uar','—'):>10} "
              f"{v.get('acc','—'):>10} {v.get('f1','—'):>10} {v.get('uar','—'):>10}")

    print(f"{'─'*85}")
    print(f"{'Park et al. (2020)':<25} {'~65%':>10} {'—':>10} {'—':>10} {'~60%':>10} {'—':>10} {'—':>10}")
    print(f"{'Alhussein (2024)':<25} {'~75%':>10} {'—':>10} {'—':>10} {'~70%':>10} {'—':>10} {'—':>10}")
    print(f"{'V4 VisionMER':<25} {'~55%':>10} {'—':>10} {'—':>10} {'~83%':>10} {'—':>10} {'—':>10}")

    # ── Save ──
    with open(out_dir / "fusion_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {out_dir / 'fusion_results.json'}")


if __name__ == "__main__":
    main()
