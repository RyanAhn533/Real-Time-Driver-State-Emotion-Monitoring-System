#!/usr/bin/env python3
"""
Phase 0: K-FER Domain Gap Verification
========================================
K-FER (AI Hub acted data, 7-class) → K-EmoCon (spontaneous) face crops에 적용.
Entropy 분포, 예측 class 분포, HSEmotion과의 일치율 분석.

판정 기준 (plan 명시):
  - K-FER entropy < 1.5 on >= 50% valid frames → 유효 (domain gap tolerable)
  - Neutral 80%+ → severe domain gap (model collapse)

Usage:
    python phase0_kfer_domain_gap.py [--device cuda] [--max-segments 0]
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

os.environ["GLOG_minloglevel"] = "3"

import numpy as np
import torch
from pathlib import Path
from collections import Counter

# ── Paths ──
PROJ_ROOT = Path("/home/ajy/Jetson_thor")
EMO_SYS = PROJ_ROOT / "emotion_system"
MM_DMS = PROJ_ROOT / "multimodal_dms"
DATA_ROOT = PROJ_ROOT / "data" / "precessed_data"
CKPT_PATH = EMO_SYS / "result" / "best.pth"
NPZ_PATH = MM_DMS / "features" / "kemocon_features.npz"
SEG_CSV = DATA_ROOT / "segments_index.csv"
OUTPUT_DIR = MM_DMS / "features"

# Add emotion_system to path for imports
sys.path.insert(0, str(EMO_SYS))


# ── AU Region Definitions (must match training: 8 regions) ──
AU_REGIONS = [
    ("forehead",    (69, 299, 9)),
    ("eyes_left",   159),
    ("eyes_right",  386),
    ("nose",        195),
    ("cheek_left",  186),
    ("cheek_right", 410),
    ("mouth",       13),
    ("chin",        18),
]

# K-FER 7-class labels (AI Hub Korean)
KFER_LABELS = ["angry", "anxious", "happy", "hurt", "neutral", "sad", "surprised"]

# HSEmotion 8-class labels (AffectNet)
HSE_LABELS = ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# HSEmotion → K-FER approximate mapping (for agreement check)
HSE_TO_KFER = {
    0: 0,   # Angry → angry
    1: 3,   # Contempt → hurt (approximate)
    2: 0,   # Disgust → angry (approximate)
    3: 1,   # Fear → anxious
    4: 2,   # Happy → happy
    5: 4,   # Neutral → neutral
    6: 5,   # Sad → sad
    7: 6,   # Surprise → surprised
}


def load_kfer_model(ckpt_path: str, device: str = "cuda"):
    """Load K-FER model from checkpoint."""
    from models.fer_model import AUFERModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    # Config is nested: config["model"] has model params
    model_cfg = config.get("model", config)
    aug_cfg = config.get("augmentation", config)

    img_size = aug_cfg.get("global_img_size", model_cfg.get("GLOBAL_IMG_SIZE", 224))
    d_emb = model_cfg.get("d_emb", 384)
    n_heads = model_cfg.get("n_heads", 8)
    n_layers = model_cfg.get("n_fusion_layers", 1)
    num_classes = model_cfg.get("num_classes", 7)
    backbone = model_cfg.get("backbone", "mobilevitv2_100")
    num_au = len(AU_REGIONS)  # 8

    # AI Hub Korean FER 7-class labels (sorted alphabetically as used in training)
    label2id = ckpt.get("label2id", None)
    if label2id is None or len(label2id) == 0:
        # Standard AI Hub labels (alphabetical order as in build_label_mapping)
        label2id = {lb: i for i, lb in enumerate(KFER_LABELS)}
    id2label = {v: k for k, v in label2id.items()}

    model = AUFERModel(
        backbone_name=backbone,
        pretrained=False,
        num_au=num_au,
        num_classes=num_classes,
        d_emb=d_emb,
        n_heads=n_heads,
        n_fusion_layers=n_layers,
        img_size=img_size,
    )

    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    # Get normalization from backbone
    norm_mean = torch.tensor(model.backbone.norm_mean, dtype=torch.float32).view(3, 1, 1).to(device)
    norm_std = torch.tensor(model.backbone.norm_std, dtype=torch.float32).view(3, 1, 1).to(device)

    print(f"[K-FER] Loaded: {num_classes} classes, {num_au} AU, d_emb={d_emb}, backbone={backbone}")
    print(f"[K-FER] label2id: {label2id}")
    print(f"[K-FER] Best metric (from ckpt): {ckpt.get('best_metric', 'N/A')}")

    return model, id2label, norm_mean, norm_std, img_size


def compute_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of probability distribution."""
    p = np.clip(probs, 1e-10, 1.0)
    return -np.sum(p * np.log(p))


@torch.no_grad()
def run_kfer_on_segment(model, faces_np, norm_mean, norm_std, img_size, device,
                        default_au_coords):
    """
    Run K-FER on face crops for one segment using default AU coords.

    Face crops are already aligned 224x224 faces, so fixed proportional AU
    positions are valid for domain gap checking. The backbone sees the entire
    face; AU RoI sampling at approximate positions still captures relevant
    facial regions.

    Args:
        faces_np: (T, 3, 224, 224) float32 [0-255]
        default_au_coords: (8, 2) pre-computed default AU positions

    Returns:
        dict with probs, entropy, top1_class, top1_conf, n_valid_frames
        or None if no valid frames
    """
    T = faces_np.shape[0]

    # Filter zero/empty frames
    valid_frames = []
    for t in range(T):
        frame = faces_np[t]
        if np.linalg.norm(frame) > 1.0:
            valid_frames.append(frame)

    if len(valid_frames) == 0:
        return None

    # Sample up to 4 evenly-spaced frames (match face_expert.py strategy)
    max_frames = min(4, len(valid_frames))
    if len(valid_frames) > max_frames:
        indices = np.linspace(0, len(valid_frames) - 1, max_frames, dtype=int)
        valid_frames = [valid_frames[i] for i in indices]

    # Batch inference for efficiency
    batch = np.stack(valid_frames, axis=0)  # (N, 3, 224, 224)
    img_tensor = torch.from_numpy(batch).float().to(device) / 255.0
    img_tensor = (img_tensor - norm_mean) / norm_std

    N = img_tensor.shape[0]
    au_tensor = torch.from_numpy(default_au_coords).float().unsqueeze(0).expand(N, -1, -1).to(device)

    logits = model(img_tensor, au_tensor)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (N, 7)

    # Average probabilities across frames
    avg_probs = np.mean(probs, axis=0)
    entropy = compute_entropy(avg_probs)
    top1_id = int(np.argmax(avg_probs))
    top1_conf = float(avg_probs[top1_id])

    return {
        "probs": avg_probs,            # (7,) averaged over frames
        "entropy": entropy,
        "top1_id": top1_id,
        "top1_conf": top1_conf,
        "n_valid_frames": len(valid_frames),
        "n_total_frames": T,
    }


def make_default_au_coords(img_size: int = 224) -> np.ndarray:
    """
    Default AU coords for aligned face crops (224x224).
    Based on typical MediaPipe FaceMesh proportions for frontal faces.
    8 regions: forehead, eyes_left, eyes_right, nose, cheek_left, cheek_right, mouth, chin.
    """
    s = img_size
    return np.array([
        [s * 0.50, s * 0.22],  # forehead (top center)
        [s * 0.35, s * 0.38],  # eyes_left
        [s * 0.65, s * 0.38],  # eyes_right
        [s * 0.50, s * 0.52],  # nose (center)
        [s * 0.22, s * 0.55],  # cheek_left
        [s * 0.78, s * 0.55],  # cheek_right
        [s * 0.50, s * 0.72],  # mouth
        [s * 0.50, s * 0.88],  # chin
    ], dtype=np.float32)


def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Phase 0: K-FER Domain Gap Check")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--max-segments", type=int, default=0,
                        help="Max segments to process (0=all)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[Phase 0] Device: {device}")

    # ── 1. Load K-FER model ──
    print("\n[1] Loading K-FER model...")
    model, id2label, norm_mean, norm_std, img_size = load_kfer_model(str(CKPT_PATH), device)

    # ── 2. Load existing HSEmotion features ──
    print("[2] Loading existing HSEmotion features...")
    npz = np.load(str(NPZ_PATH))
    hse_probs = npz["face_probs"]      # (3577, 8)
    hse_valid = npz["face_valid"]       # (3577,)
    pids = npz["pid"]                   # (3577,)
    seg_idxs = npz["seg_idx"]          # (3577,)
    label_a = npz["label_A_bin"]       # (3577,)
    label_v = npz["label_V_bin"]       # (3577,)
    N = len(pids)
    print(f"  Total segments: {N}")
    print(f"  HSE face_valid: {hse_valid.sum()} ({hse_valid.mean()*100:.1f}%)")

    # ── 3. Load segments index ──
    print("[3] Loading segments index...")
    seg_df = pd.read_csv(str(SEG_CSV))
    print(f"  CSV rows: {len(seg_df)}")

    # Build lookup: (pid, seg_idx) → video_face_path
    face_path_map = {}
    has_face_col = "has_video_face" in seg_df.columns
    for _, row in seg_df.iterrows():
        key = (int(row["pid"]), int(row["seg_idx"]))
        if has_face_col:
            face_path_map[key] = (str(row.get("video_face_path", "")),
                                   bool(row.get("has_video_face", False)))
        else:
            fp = str(row.get("video_face_path", ""))
            face_path_map[key] = (fp, len(fp) > 0)

    # ── 4. Prepare default AU coordinates ──
    print("[4] Using default AU coords for aligned face crops (no MediaPipe needed)...")
    default_au_coords = make_default_au_coords(img_size)
    print(f"  AU coords (8 regions): {default_au_coords.shape}")

    # ── 5. Run K-FER on K-EmoCon face crops ──
    print("[5] Running K-FER inference on K-EmoCon face crops...")

    results = []
    n_processed = 0
    n_skipped_no_face = 0
    n_skipped_no_file = 0
    n_failed = 0

    max_seg = args.max_segments if args.max_segments > 0 else N
    t0 = time.time()

    for i in range(min(N, max_seg)):
        pid = int(pids[i])
        seg = int(seg_idxs[i])
        key = (pid, seg)

        if key not in face_path_map:
            n_skipped_no_face += 1
            results.append(None)
            continue

        face_rel_path, has_face = face_path_map[key]
        if not has_face or not face_rel_path:
            n_skipped_no_face += 1
            results.append(None)
            continue

        face_npz_path = DATA_ROOT / face_rel_path
        if not face_npz_path.exists():
            n_skipped_no_file += 1
            results.append(None)
            continue

        try:
            face_data = np.load(str(face_npz_path))
            faces = face_data["faces"]  # (T, 3, 224, 224)

            result = run_kfer_on_segment(model, faces, norm_mean, norm_std, img_size, device,
                                                default_au_coords)
            results.append(result)

            if result is not None:
                n_processed += 1
            else:
                n_failed += 1

        except Exception as e:
            results.append(None)
            n_failed += 1
            if n_failed <= 5:
                print(f"  [WARN] seg {i} (pid={pid}, seg={seg}): {e}")

        if (i + 1) % 200 == 0 or i == min(N, max_seg) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Progress: {i+1}/{min(N, max_seg)} "
                  f"({n_processed} ok, {n_skipped_no_face} no_face, "
                  f"{n_skipped_no_file} no_file, {n_failed} fail) "
                  f"[{rate:.1f} seg/s]")

    elapsed = time.time() - t0
    print(f"\n[6] Inference complete. {n_processed} segments processed in {elapsed:.1f}s")

    # ── 6. Analyze results ──
    print("\n" + "="*70)
    print("         PHASE 0: K-FER DOMAIN GAP ANALYSIS")
    print("="*70)

    # Collect valid results
    valid_idx = []
    kfer_probs_list = []
    kfer_entropy_list = []
    kfer_top1_list = []
    kfer_conf_list = []

    for i, r in enumerate(results):
        if r is not None:
            valid_idx.append(i)
            kfer_probs_list.append(r["probs"])
            kfer_entropy_list.append(r["entropy"])
            kfer_top1_list.append(r["top1_id"])
            kfer_conf_list.append(r["top1_conf"])

    kfer_probs_all = np.array(kfer_probs_list)      # (M, 7)
    kfer_entropy_all = np.array(kfer_entropy_list)   # (M,)
    kfer_top1_all = np.array(kfer_top1_list)         # (M,)
    kfer_conf_all = np.array(kfer_conf_list)         # (M,)
    M = len(valid_idx)

    print(f"\n[A] Coverage:")
    print(f"  Total segments:     {N}")
    print(f"  K-FER processed:    {M} ({M/N*100:.1f}%)")
    print(f"  No face crop:       {n_skipped_no_face}")
    print(f"  File missing:       {n_skipped_no_file}")
    print(f"  Processing fail:    {n_failed}")
    print(f"  Note: Using default AU coords (aligned face crops)")

    # ── Entropy distribution ──
    print(f"\n[B] Entropy Distribution (max possible = {np.log(7):.3f} for 7-class):")
    print(f"  Mean:    {kfer_entropy_all.mean():.4f}")
    print(f"  Median:  {np.median(kfer_entropy_all):.4f}")
    print(f"  Std:     {kfer_entropy_all.std():.4f}")
    print(f"  Min:     {kfer_entropy_all.min():.4f}")
    print(f"  Max:     {kfer_entropy_all.max():.4f}")

    # Key metric: fraction with entropy < 1.5
    low_ent_frac = (kfer_entropy_all < 1.5).mean()
    mid_ent_frac = ((kfer_entropy_all >= 1.5) & (kfer_entropy_all < 1.8)).mean()
    high_ent_frac = (kfer_entropy_all >= 1.8).mean()

    print(f"\n  Entropy < 1.0 (confident):     {(kfer_entropy_all < 1.0).mean()*100:.1f}%")
    print(f"  Entropy < 1.5 (usable):        {low_ent_frac*100:.1f}%")
    print(f"  1.5 <= Entropy < 1.8 (noisy):  {mid_ent_frac*100:.1f}%")
    print(f"  Entropy >= 1.8 (near random):  {high_ent_frac*100:.1f}%")

    # Entropy histogram (text)
    print(f"\n  Entropy histogram:")
    bins = np.linspace(0, np.log(7) + 0.1, 11)
    hist, bin_edges = np.histogram(kfer_entropy_all, bins=bins)
    for j in range(len(hist)):
        bar = "#" * int(hist[j] / max(hist.max(), 1) * 40)
        print(f"    [{bin_edges[j]:.2f}-{bin_edges[j+1]:.2f}): {hist[j]:5d}  {bar}")

    # ── Class distribution ──
    print(f"\n[C] K-FER Top-1 Class Distribution:")
    class_counts = Counter(kfer_top1_all.tolist())
    for cls_id in range(7):
        cnt = class_counts.get(cls_id, 0)
        pct = cnt / M * 100 if M > 0 else 0
        bar = "#" * int(pct / 2)
        lbl = id2label.get(cls_id, f"cls_{cls_id}")
        print(f"  {cls_id}: {lbl:12s}  {cnt:5d} ({pct:5.1f}%)  {bar}")

    neutral_frac = class_counts.get(4, 0) / M * 100 if M > 0 else 0  # assuming neutral=4
    # Find which class is neutral
    neutral_id = None
    for k, v in id2label.items():
        if v == "neutral":
            neutral_id = k
            break
    if neutral_id is not None:
        neutral_frac = class_counts.get(neutral_id, 0) / M * 100 if M > 0 else 0
        print(f"\n  *** Neutral fraction: {neutral_frac:.1f}% "
              f"{'(SEVERE: model collapse!)' if neutral_frac > 80 else '(OK)' if neutral_frac < 50 else '(elevated, watch)'}")

    # ── Confidence distribution ──
    print(f"\n[D] Top-1 Confidence Distribution:")
    print(f"  Mean:    {kfer_conf_all.mean():.4f}")
    print(f"  Median:  {np.median(kfer_conf_all):.4f}")
    print(f"  >0.9:    {(kfer_conf_all > 0.9).mean()*100:.1f}%")
    print(f"  >0.7:    {(kfer_conf_all > 0.7).mean()*100:.1f}%")
    print(f"  >0.5:    {(kfer_conf_all > 0.5).mean()*100:.1f}%")
    print(f"  <0.3:    {(kfer_conf_all < 0.3).mean()*100:.1f}% (near random)")

    # ── HSEmotion vs K-FER agreement ──
    print(f"\n[E] HSEmotion ↔ K-FER Agreement (approximate mapping):")

    n_agree = 0
    n_compared = 0
    agree_by_class = Counter()
    disagree_by_class = Counter()

    for j, i in enumerate(valid_idx):
        if i < len(hse_valid) and hse_valid[i]:
            hse_top1 = int(np.argmax(hse_probs[i]))
            kfer_top1 = int(kfer_top1_all[j])

            # Map HSE to K-FER label space
            hse_mapped = HSE_TO_KFER.get(hse_top1, -1)

            n_compared += 1
            if hse_mapped == kfer_top1:
                n_agree += 1
                agree_by_class[kfer_top1] += 1
            else:
                disagree_by_class[(hse_mapped, kfer_top1)] += 1

    if n_compared > 0:
        agreement_rate = n_agree / n_compared * 100
        print(f"  Compared: {n_compared} segments")
        print(f"  Agreement: {n_agree} ({agreement_rate:.1f}%)")

        print(f"\n  Top disagreements (HSE→KFER mapped vs KFER predicted):")
        for (hse_m, kfer_p), cnt in disagree_by_class.most_common(10):
            hse_lbl = id2label.get(hse_m, f"id{hse_m}")
            kfer_lbl = id2label.get(kfer_p, f"id{kfer_p}")
            print(f"    HSE→{hse_lbl:12s} vs KFER→{kfer_lbl:12s}: {cnt} ({cnt/n_compared*100:.1f}%)")

    # ── Per-Arousal analysis ──
    print(f"\n[F] K-FER Entropy by Arousal Label:")
    for a_val in [0, 1]:
        mask = []
        for j, i in enumerate(valid_idx):
            if i < len(label_a):
                mask.append(label_a[i] == a_val)
            else:
                mask.append(False)
        mask = np.array(mask)
        if mask.sum() > 0:
            ent_a = kfer_entropy_all[mask]
            lbl = "low" if a_val == 0 else "high"
            print(f"  Arousal={lbl} ({mask.sum()} segs): "
                  f"entropy mean={ent_a.mean():.3f}, "
                  f"<1.5: {(ent_a<1.5).mean()*100:.1f}%, "
                  f"conf mean={kfer_conf_all[mask].mean():.3f}")

    # ── VERDICT ──
    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")

    pass_entropy = low_ent_frac >= 0.5
    pass_neutral = neutral_frac < 80 if neutral_id is not None else True
    pass_overall = pass_entropy and pass_neutral

    if pass_overall:
        print(f"  ✓ PASS: K-FER domain gap is TOLERABLE")
        print(f"    - Entropy < 1.5 on {low_ent_frac*100:.1f}% >= 50% ✓")
        if neutral_id is not None:
            print(f"    - Neutral fraction {neutral_frac:.1f}% < 80% ✓")
        print(f"    → K-FER features can be used in fusion with uncertainty weighting")
    else:
        print(f"  ✗ FAIL: K-FER domain gap is SEVERE")
        if not pass_entropy:
            print(f"    - Entropy < 1.5 on only {low_ent_frac*100:.1f}% < 50% ✗")
        if not pass_neutral:
            print(f"    - Neutral fraction {neutral_frac:.1f}% >= 80% (model collapse) ✗")
        print(f"    → Consider: (a) fine-tune K-FER on K-EmoCon, or (b) use only HSEmotion")

    # Policy recommendation for fusion
    print(f"\n  Recommended uncertainty policy:")
    print(f"    entropy < 1.0  → face token weight × 1.2 (boost)")
    print(f"    entropy < 1.5  → face token weight × 1.0 (normal)")
    print(f"    1.5 <= ent < 1.8 → face token weight × 0.5 (discount)")
    print(f"    entropy >= 1.8 → face token MASK (drop)")

    # ── 7. Save results ──
    print(f"\n[7] Saving results...")

    # Build full-size arrays (N,) with NaN for missing
    kfer_probs_full = np.full((N, 7), np.nan, dtype=np.float32)
    kfer_entropy_full = np.full(N, np.nan, dtype=np.float32)
    kfer_top1_full = np.full(N, -1, dtype=np.int32)
    kfer_conf_full = np.full(N, np.nan, dtype=np.float32)
    kfer_valid_full = np.zeros(N, dtype=bool)

    for j, i in enumerate(valid_idx):
        kfer_probs_full[i] = kfer_probs_list[j]
        kfer_entropy_full[i] = kfer_entropy_list[j]
        kfer_top1_full[i] = kfer_top1_list[j]
        kfer_conf_full[i] = kfer_conf_list[j]
        kfer_valid_full[i] = True

    out_path = OUTPUT_DIR / "phase0_kfer_domain_gap.npz"
    np.savez_compressed(str(out_path),
        kfer_probs=kfer_probs_full,          # (N, 7)
        kfer_entropy=kfer_entropy_full,      # (N,)
        kfer_top1=kfer_top1_full,            # (N,)
        kfer_confidence=kfer_conf_full,      # (N,)
        kfer_valid=kfer_valid_full,          # (N,)
        id2label=np.array(list(id2label.items()), dtype=object),
        # Metadata
        pass_entropy=np.array(pass_entropy),
        pass_neutral=np.array(pass_neutral),
        entropy_frac_below_1_5=np.array(low_ent_frac),
        neutral_frac=np.array(neutral_frac if neutral_id is not None else -1.0),
    )
    print(f"  Saved to: {out_path}")
    print(f"  Arrays: kfer_probs({kfer_probs_full.shape}), kfer_entropy({N}), kfer_valid({N})")

    print(f"\n[Phase 0 Complete] Total time: {time.time()-t0:.1f}s")
    return pass_overall


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
