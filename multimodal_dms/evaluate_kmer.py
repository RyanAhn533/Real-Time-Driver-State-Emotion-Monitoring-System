#!/usr/bin/env python3
"""
K-MER Evaluation + 실증 Report
================================
Reads ablation results and generates comprehensive evaluation:
  1. Performance comparison table (8 experiments)
  2. Expert contribution (attention weights)
  3. Korean compound emotion distribution
  4. Robustness matrix (modality combinations)
  5. Dynamic α analysis
  6. KD checklist

Usage:
    python evaluate_kmer.py
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

RESULT_DIR = Path(__file__).parent / "results_kmer"
FEATURES_V2 = Path(__file__).parent / "features" / "kemocon_features_v2.npz"


def load_ablation_results():
    """Load ablation results from JSON."""
    path = RESULT_DIR / "ablation_results.json"
    if not path.exists():
        print(f"  ERROR: {path} not found. Run train_kmer.py first.")
        return None
    with open(path) as f:
        return json.load(f)


def report_performance_table(results):
    """(1) Performance comparison table."""
    print("\n" + "=" * 75)
    print("  (1) PERFORMANCE COMPARISON TABLE")
    print("=" * 75)
    print(f"{'#':>2}  {'Name':<30}  {'A_UAR':>8}  {'±std':>6}  {'vs #1':>7}")
    print("-" * 60)

    baseline_uar = results[0]["mean_A_UAR"] if results else 0

    for r in results:
        delta = r["mean_A_UAR"] - baseline_uar
        delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
        star = " ★" if "dynamic" in r["name"].lower() else ""
        print(f"{r['ablation_id']:>2}  {r['name']:<30}  "
              f"{r['mean_A_UAR']:>7.2f}%  ±{r['std_A_UAR']:.2f}  "
              f"{delta_str:>6}%{star}")

    # Find best
    best = max(results, key=lambda x: x["mean_A_UAR"])
    print(f"\n  Best: #{best['ablation_id']} {best['name']} = {best['mean_A_UAR']:.2f}%")

    # KPI check
    kpi_target = 60.5
    if best["mean_A_UAR"] >= kpi_target:
        print(f"  ✓ KPI PASS: A_UAR {best['mean_A_UAR']:.2f}% >= {kpi_target}%")
    else:
        print(f"  ✗ KPI MISS: A_UAR {best['mean_A_UAR']:.2f}% < {kpi_target}%")
        print(f"    Gap: {kpi_target - best['mean_A_UAR']:.2f}% to close")


def report_expert_contribution(results):
    """(2) Expert contribution analysis."""
    print("\n" + "=" * 75)
    print("  (2) EXPERT CONTRIBUTION ANALYSIS")
    print("=" * 75)

    # Compare with/without K-FER (ablation 5 vs 6)
    hybrid_all = None
    hybrid_no_kfer = None
    hybrid_no_bio = None

    for r in results:
        if r["ablation_id"] == 5:
            hybrid_all = r
        elif r["ablation_id"] == 6:
            hybrid_no_kfer = r
        elif r["ablation_id"] == 7:
            hybrid_no_bio = r

    if hybrid_all and hybrid_no_kfer:
        delta_kfer = hybrid_all["mean_A_UAR"] - hybrid_no_kfer["mean_A_UAR"]
        print(f"  K-FER contribution:  {delta_kfer:+.2f}% A_UAR")
        print(f"    (with K-FER: {hybrid_all['mean_A_UAR']:.2f}% vs "
              f"without: {hybrid_no_kfer['mean_A_UAR']:.2f}%)")

    if hybrid_all and hybrid_no_bio:
        delta_bio = hybrid_all["mean_A_UAR"] - hybrid_no_bio["mean_A_UAR"]
        print(f"  Bio contribution:    {delta_bio:+.2f}% A_UAR")
        print(f"    (with Bio: {hybrid_all['mean_A_UAR']:.2f}% vs "
              f"without: {hybrid_no_bio['mean_A_UAR']:.2f}%)")

    # LGBM 4-expert vs 5-expert
    lgbm4 = None
    lgbm5 = None
    for r in results:
        if r["ablation_id"] == 1:
            lgbm4 = r
        elif r["ablation_id"] == 2:
            lgbm5 = r

    if lgbm4 and lgbm5:
        delta_lgbm = lgbm5["mean_A_UAR"] - lgbm4["mean_A_UAR"]
        print(f"  K-FER in LGBM:       {delta_lgbm:+.2f}% A_UAR")
        print(f"    (5-expert: {lgbm5['mean_A_UAR']:.2f}% vs "
              f"4-expert: {lgbm4['mean_A_UAR']:.2f}%)")

    # Validity mask contribution
    neural_mask = None
    neural_no_mask = None
    for r in results:
        if r["ablation_id"] == 3:
            neural_mask = r
        elif r["ablation_id"] == 8:
            neural_no_mask = r

    if neural_mask and neural_no_mask:
        delta_mask = neural_mask["mean_A_UAR"] - neural_no_mask["mean_A_UAR"]
        print(f"  Validity mask:       {delta_mask:+.2f}% A_UAR")
        print(f"    (with mask: {neural_mask['mean_A_UAR']:.2f}% vs "
              f"without: {neural_no_mask['mean_A_UAR']:.2f}%)")


def report_compound_emotion():
    """(3) Korean compound emotion distribution."""
    print("\n" + "=" * 75)
    print("  (3) KOREAN COMPOUND EMOTION (13-class, inference-time)")
    print("=" * 75)

    if not FEATURES_V2.exists():
        print("  Features V2 not found. Skipping.")
        return

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from fusion.compound_emotion import CompoundEmotionMapper

    data = np.load(str(FEATURES_V2), allow_pickle=True)
    kfer_probs = data["kfer_probs"]
    kfer_valid = data["kfer_valid"]
    audeering_avd = data["audeering_avd"]
    audeering_valid = data["audeering_valid"]

    mapper = CompoundEmotionMapper()

    # Map valid segments
    valid = kfer_valid.astype(bool)
    kfer_top1 = kfer_probs[valid].argmax(axis=1)

    # Use audeering arousal as proxy (normalized to [0,1])
    arousal_raw = audeering_avd[valid, 0]
    arousal_min, arousal_max = arousal_raw.min(), arousal_raw.max()
    if arousal_max - arousal_min > 1e-6:
        arousal_norm = (arousal_raw - arousal_min) / (arousal_max - arousal_min)
    else:
        arousal_norm = np.full_like(arousal_raw, 0.5)

    compound_ids, compound_labels = mapper.map_batch(kfer_top1, arousal_norm)
    dist = mapper.get_distribution(compound_ids)

    print(f"  Valid segments: {valid.sum()}")
    print(f"\n  {'Label':<20}  {'Count':>6}  {'%':>6}  Bar")
    print(f"  {'-'*55}")
    total = sum(dist.values())
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {label:<20}  {count:>6}  {pct:>5.1f}%  {bar}")

    # Count unique labels
    n_unique = len(dist)
    print(f"\n  Unique compound labels used: {n_unique}/13")
    if n_unique >= 8:
        print(f"  ✓ Good diversity for showcase")
    else:
        print(f"  △ Limited diversity — arousal predictions may need tuning")


def report_dynamic_alpha(results):
    """(5) Dynamic α analysis."""
    print("\n" + "=" * 75)
    print("  (5) DYNAMIC α ANALYSIS")
    print("=" * 75)

    static_result = None
    dynamic_result = None

    for r in results:
        if r["ablation_id"] == 4:
            static_result = r
        elif r["ablation_id"] == 5:
            dynamic_result = r

    if static_result and dynamic_result:
        delta = dynamic_result["mean_A_UAR"] - static_result["mean_A_UAR"]
        print(f"  Static α=0.5:   {static_result['mean_A_UAR']:.2f}%")
        print(f"  Dynamic α:      {dynamic_result['mean_A_UAR']:.2f}%")
        print(f"  Improvement:    {delta:+.2f}%")

        # Fold-level alpha stats
        if dynamic_result.get("folds"):
            alphas = [f.get("alpha_mean", 0.5) for f in dynamic_result["folds"]]
            alpha_stds = [f.get("alpha_std", 0) for f in dynamic_result["folds"]]
            print(f"\n  Per-fold α statistics:")
            for i, f in enumerate(dynamic_result["folds"]):
                am = f.get("alpha_mean", 0.5)
                ast = f.get("alpha_std", 0)
                print(f"    Fold {i+1}: α={am:.3f} ± {ast:.3f}")
            print(f"    Overall: α={np.mean(alphas):.3f} ± {np.mean(alpha_stds):.3f}")

            if np.mean(alpha_stds) > 0.05:
                print(f"    ✓ α is dynamic (std > 0.05)")
            else:
                print(f"    △ α is near-static (std < 0.05)")


def report_kd_checklist():
    """(6) KD deployment checklist."""
    print("\n" + "=" * 75)
    print("  (6) KD DEPLOYMENT CHECKLIST")
    print("=" * 75)

    checks = [
        ("Cross-label KD mapping matrix M(8→7)",
         Path(__file__).parent / "kd" / "cross_label_kd.py"),
        ("Teacher cache module",
         Path(__file__).parent / "kd" / "teacher_cache.py"),
        ("KMERFusion model (teacher)",
         Path(__file__).parent / "fusion" / "kmer_fusion.py"),
        ("K-FER expert (student backbone)",
         Path(__file__).parent / "experts" / "kfer_expert.py"),
        ("V2 features NPZ",
         FEATURES_V2),
        ("Ablation results",
         RESULT_DIR / "ablation_results.json"),
    ]

    all_ok = True
    for name, path in checks:
        exists = path.exists()
        status = "✓" if exists else "✗"
        if not exists:
            all_ok = False
        print(f"  {status} {name}")
        if exists:
            size = path.stat().st_size
            if size > 1e6:
                print(f"      → {path.name} ({size/1e6:.1f} MB)")
            else:
                print(f"      → {path.name} ({size/1e3:.1f} KB)")

    print(f"\n  Student model spec (Jetson Orin):")
    print(f"    K-FER backbone: MobileViTv2-100 (~5M params)")
    print(f"    Bio MLP: 15→128d")
    print(f"    Concat(384+128=512) → ArousalHead(512→1)")
    print(f"    Total: ~5.5M params, ~30ms on Orin (estimated)")
    print(f"    KD: 0.5×CE + 0.3×MSE(repr) + 0.2×KL(soft)")

    if all_ok:
        print(f"\n  ✓ All KD components ready for deployment pipeline")
    else:
        print(f"\n  △ Some components missing — generate before deployment")


def main():
    print("=" * 75)
    print("  K-MER v3: EVALUATION + 실증 REPORT")
    print("  Korean Multimodal Emotion Recognition System")
    print("=" * 75)

    results = load_ablation_results()

    if results:
        # (1) Performance table
        report_performance_table(results)

        # (2) Expert contribution
        report_expert_contribution(results)

        # (5) Dynamic α
        report_dynamic_alpha(results)
    else:
        print("\n  No ablation results found. Run train_kmer.py first.")

    # (3) Compound emotion
    report_compound_emotion()

    # (4) Robustness analysis would require re-running with forced masks
    print("\n" + "=" * 75)
    print("  (4) ROBUSTNESS MATRIX")
    print("=" * 75)
    print("  (Requires model re-evaluation with forced modality dropout)")
    print("  Run with --stress-test flag for full robustness evaluation.")

    # (6) KD checklist
    report_kd_checklist()

    print("\n" + "=" * 75)
    print("  REPORT COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
