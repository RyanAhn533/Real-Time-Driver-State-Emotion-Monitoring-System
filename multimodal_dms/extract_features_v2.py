#!/usr/bin/env python3
"""
Feature Extraction V2 — 5 Experts + Auxiliary Signals
======================================================
Extends the existing kemocon_features.npz with:
  1. K-FER expert:  7-class probs + entropy + quality
  2. FACS Aux:      PERCLOS + EAR + FACS geometric scores (if MediaPipe available)
  3. Bio V2:        per-channel quality scores
  4. Cross-modal:   agreement/consistency features

Reuses existing face(HSEmotion)/audio/bio features from kemocon_features.npz.
Only extracts NEW features (K-FER) and augments existing ones.

Output: kemocon_features_v2.npz

Usage:
    python extract_features_v2.py [--device cuda] [--reuse-phase0]
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Paths ──
PROJ_ROOT = Path("/home/ajy/Jetson_thor")
DATA_ROOT = PROJ_ROOT / "data" / "precessed_data"
MM_DMS = PROJ_ROOT / "multimodal_dms"
FEATURES_DIR = MM_DMS / "features"
EXISTING_NPZ = FEATURES_DIR / "kemocon_features.npz"
PHASE0_NPZ = FEATURES_DIR / "phase0_kfer_domain_gap.npz"
SEG_CSV = DATA_ROOT / "segments_index.csv"

# Token dimensions for fusion (reference)
TOKEN_DIMS = {
    "T1_kfer_probs":       7,   # K-FER 7-class
    "T2_kfer_meta":        2,   # quality + entropy
    "T3_face_stats":       3,   # max_conf, mean_conf, std_conf (from HSEmotion)
    "T4_emo2vec_probs":    9,   # emotion2vec 9-class
    "T5_audeering_avd":    3,   # arousal, valence, dominance
    "T6_audio_quality":    3,   # rms, voicing_ratio, snr_est (placeholder)
    "T7_bvp_features":     4,   # mean_hr, sdnn, rmssd, lf_hf
    "T8_eda_features":     5,   # mean_scl, std_scl, n_peaks, amp, auc
    "T9_hr_temp_features": 6,   # hr_mean/std/range + temp_mean/slope/range
    "T10_bio_quality":     3,   # bvp_valid, eda_valid, hr_valid
    "T11_perclos_ear":     2,   # perclos, ear_mean
    "T12_facs_scores":     6,   # geometric emotion indicators
    "T13_cross_modal":     3,   # face_audio_agree, av_consistency, entropy_gap
    "T14_validity_flags":  3,   # face_valid, audio_valid, bio_valid
}


def load_npz(path):
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def compute_cross_modal_features(
    kfer_probs, kfer_entropy, hse_probs, hse_valid,
    emo2vec_probs, emo2vec_valid, audeering_avd, audeering_valid
):
    """
    Compute cross-modal agreement features for T13 token.

    Returns:
        (N, 3) float32: [face_audio_agree, av_consistency, entropy_gap]
    """
    N = len(kfer_probs)
    cross = np.zeros((N, 3), dtype=np.float32)

    for i in range(N):
        # 1. Face-Audio agreement: cosine similarity of emotion distributions
        # Map K-FER probs and emo2vec probs to a common space (valence-like)
        # K-FER: angry(0), anxious(1), happy(2), hurt(3), neutral(4), sad(5), surprised(6)
        # Positive: happy, surprised → high valence
        # Negative: angry, anxious, hurt, sad → low valence
        if hse_valid[i] and emo2vec_valid[i]:
            kfer_p = kfer_probs[i]
            e2v_p = emo2vec_probs[i]
            # Simple valence proxy from K-FER
            kfer_valence = kfer_p[2] + 0.5 * kfer_p[6] - kfer_p[0] - kfer_p[1] - kfer_p[3] - kfer_p[5]
            # Simple valence proxy from emo2vec
            # emo2vec: angry(0),disgusted(1),fearful(2),happy(3),neutral(4),other(5),sad(6),surprised(7),unknown(8)
            e2v_valence = e2v_p[3] + 0.5 * e2v_p[7] - e2v_p[0] - e2v_p[1] - e2v_p[2] - e2v_p[6]
            # Agreement = 1 - |diff| / 2 (normalize to [0,1])
            diff = abs(kfer_valence - e2v_valence)
            cross[i, 0] = max(0, 1.0 - diff / 2.0)

        # 2. A/V consistency: audeering arousal vs emo2vec intensity
        if audeering_valid[i] and emo2vec_valid[i]:
            arousal = audeering_avd[i, 0]  # continuous arousal
            e2v_intensity = 1.0 - emo2vec_probs[i, 4]  # 1 - neutral prob
            cross[i, 1] = 1.0 - abs(arousal - e2v_intensity)

        # 3. Entropy gap: K-FER entropy vs HSEmotion entropy
        if hse_valid[i]:
            hse_ent = float(-np.sum(np.clip(hse_probs[i], 1e-10, 1) *
                                     np.log(np.clip(hse_probs[i], 1e-10, 1))))
            ent_gap = abs(kfer_entropy[i] - hse_ent)
            cross[i, 2] = max(0, 1.0 - ent_gap / 2.0)  # normalize

    return cross


def main():
    parser = argparse.ArgumentParser(description="Feature Extraction V2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-phase0", action="store_true",
                        help="Reuse K-FER results from phase0 NPZ")
    parser.add_argument("--skip-facs", action="store_true",
                        help="Skip FACS aux (if MediaPipe unavailable)")
    args = parser.parse_args()

    t0 = time.time()

    # ── 1. Load existing features ──
    print("[1] Loading existing features...")
    if not EXISTING_NPZ.exists():
        print(f"  ERROR: {EXISTING_NPZ} not found. Run extract_features.py first.")
        return

    existing = load_npz(str(EXISTING_NPZ))
    N = len(existing["pid"])
    print(f"  Loaded {N} segments from {EXISTING_NPZ.name}")

    # Existing arrays
    face_probs_hse = existing["face_probs"]      # (N, 8)
    face_embed = existing["face_embed"]           # (N, 1280)
    face_valid = existing["face_valid"]           # (N,)
    emo2vec_embed = existing["emo2vec_embed"]     # (N, 1024)
    emo2vec_probs = existing["emo2vec_probs"]     # (N, 9)
    emo2vec_valid = existing["emo2vec_valid"]     # (N,)
    audeering_avd = existing["audeering_avd"]     # (N, 3)
    audeering_valid = existing["audeering_valid"] # (N,)
    bio_features = existing["bio_features"]       # (N, 15)
    bio_valid = existing["bio_valid"]             # (N,)
    label_A = existing["label_A_bin"]             # (N,)
    label_V = existing["label_V_bin"]             # (N,)
    pids = existing["pid"]                        # (N,)
    pair_ids = existing["pair_id"]                # (N,)
    seg_idxs = existing["seg_idx"]               # (N,)

    # ── 2. K-FER features ──
    kfer_probs = np.zeros((N, 7), dtype=np.float32)
    kfer_entropy = np.full(N, np.log(7), dtype=np.float32)
    kfer_quality = np.zeros(N, dtype=np.float32)
    kfer_valid = np.zeros(N, dtype=bool)

    if args.reuse_phase0 and PHASE0_NPZ.exists():
        print("[2] Reusing K-FER features from Phase 0...")
        p0 = load_npz(str(PHASE0_NPZ))
        kfer_probs = p0["kfer_probs"].astype(np.float32)
        kfer_entropy = p0["kfer_entropy"].astype(np.float32)
        kfer_valid = p0["kfer_valid"].astype(bool)
        # Fix NaN values
        kfer_probs = np.nan_to_num(kfer_probs, nan=0.0)
        kfer_entropy = np.nan_to_num(kfer_entropy, nan=np.log(7))
        kfer_quality = np.where(kfer_valid,
                                 1.0 - kfer_entropy / np.log(7), 0.0).astype(np.float32)
        print(f"  K-FER valid: {kfer_valid.sum()} ({kfer_valid.mean()*100:.1f}%)")
    else:
        print("[2] Extracting K-FER features (fresh run)...")
        sys.path.insert(0, str(MM_DMS))
        from experts.kfer_expert import KFERExpert
        kfer_expert = KFERExpert(device=args.device)

        # Load segments index
        seg_df = pd.read_csv(str(SEG_CSV))
        face_path_map = {}
        for _, row in seg_df.iterrows():
            key = (int(row["pid"]), int(row["seg_idx"]))
            fp = str(row.get("video_face_path", ""))
            has = bool(row.get("has_video_face", False))
            face_path_map[key] = (fp, has)

        for i in tqdm(range(N), desc="K-FER"):
            pid = int(pids[i])
            seg = int(seg_idxs[i])
            key = (pid, seg)
            if key not in face_path_map:
                continue
            fp, has = face_path_map[key]
            if not has or not fp:
                continue
            face_npz = DATA_ROOT / fp
            if not face_npz.exists():
                continue
            try:
                data = np.load(str(face_npz))
                faces = data["faces"]
                result = kfer_expert.extract(faces)
                if result["valid"]:
                    kfer_probs[i] = result["probs"]
                    kfer_entropy[i] = result["entropy"]
                    kfer_quality[i] = result["quality"]
                    kfer_valid[i] = True
            except Exception:
                pass

        print(f"  K-FER valid: {kfer_valid.sum()} ({kfer_valid.mean()*100:.1f}%)")

    # ── 3. Bio V2 quality ──
    print("[3] Computing bio quality scores...")
    bvp_valid = np.zeros(N, dtype=bool)
    eda_valid = np.zeros(N, dtype=bool)
    hr_valid = np.zeros(N, dtype=bool)
    bio_quality = np.zeros(N, dtype=np.float32)

    for i in range(N):
        feats = bio_features[i]
        bvp_valid[i] = not np.allclose(feats[0:4], 0.0)
        eda_valid[i] = not np.allclose(feats[4:9], 0.0)
        hr_valid[i] = not np.allclose(feats[12:15], 0.0)
        n_v = sum([bvp_valid[i], eda_valid[i], hr_valid[i]])
        bio_quality[i] = n_v / 3.0

    print(f"  BVP valid: {bvp_valid.sum()} ({bvp_valid.mean()*100:.1f}%)")
    print(f"  EDA valid: {eda_valid.sum()} ({eda_valid.mean()*100:.1f}%)")
    print(f"  HR valid:  {hr_valid.sum()} ({hr_valid.mean()*100:.1f}%)")

    # ── 4. Face stats (from HSEmotion probs) ──
    print("[4] Computing face statistics...")
    face_stats = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        if face_valid[i]:
            probs = face_probs_hse[i]
            face_stats[i, 0] = float(np.max(probs))      # max_conf
            face_stats[i, 1] = float(np.mean(probs))      # mean_conf
            face_stats[i, 2] = float(np.std(probs))       # std_conf (spread)

    # ── 5. FACS Auxiliary (placeholder if no MediaPipe) ──
    print("[5] FACS Auxiliary features...")
    perclos = np.full(N, 0.5, dtype=np.float32)    # prior: unknown
    ear_mean = np.full(N, 0.25, dtype=np.float32)  # prior: normal
    ear_conf = np.zeros(N, dtype=np.float32)
    facs_scores = np.zeros((N, 6), dtype=np.float32)
    facs_valid = np.zeros(N, dtype=bool)

    if not args.skip_facs:
        try:
            sys.path.insert(0, str(MM_DMS))
            from experts.facs_aux import FACSAuxExpert
            facs_expert = FACSAuxExpert()

            seg_df = pd.read_csv(str(SEG_CSV))
            face_path_map = {}
            for _, row in seg_df.iterrows():
                key = (int(row["pid"]), int(row["seg_idx"]))
                fp = str(row.get("video_face_path", ""))
                has = bool(row.get("has_video_face", False))
                face_path_map[key] = (fp, has)

            for i in tqdm(range(N), desc="FACS"):
                pid, seg = int(pids[i]), int(seg_idxs[i])
                key = (pid, seg)
                if key not in face_path_map:
                    continue
                fp, has = face_path_map[key]
                if not has or not fp:
                    continue
                face_npz = DATA_ROOT / fp
                if not face_npz.exists():
                    continue
                try:
                    data = np.load(str(face_npz))
                    faces = data["faces"]
                    result = facs_expert.extract(faces)
                    perclos[i] = result["perclos"]
                    ear_mean[i] = result["ear_mean"]
                    ear_conf[i] = result["ear_conf"]
                    facs_scores[i] = result["facs_scores"]
                    facs_valid[i] = result["facs_valid"]
                except Exception:
                    pass

            print(f"  FACS valid: {facs_valid.sum()} ({facs_valid.mean()*100:.1f}%)")
        except Exception as e:
            print(f"  FACS extraction failed (MediaPipe?): {e}")
            print(f"  Using placeholder values (perclos=0.5, ear=0.25, quality=0)")
    else:
        print(f"  Skipped (using placeholders)")

    # ── 6. Audio quality placeholder ──
    print("[6] Audio quality features (placeholder)...")
    audio_quality = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        if emo2vec_valid[i] or audeering_valid[i]:
            audio_quality[i, 0] = 0.5  # rms placeholder
            audio_quality[i, 1] = 0.5  # voicing ratio placeholder
            audio_quality[i, 2] = 10.0  # snr placeholder

    # ── 7. Cross-modal agreement ──
    print("[7] Computing cross-modal agreement features...")
    cross_modal = compute_cross_modal_features(
        kfer_probs, kfer_entropy, face_probs_hse, face_valid,
        emo2vec_probs, emo2vec_valid, audeering_avd, audeering_valid
    )

    # ── 8. Validity flags ──
    print("[8] Computing validity flags...")
    validity_flags = np.zeros((N, 3), dtype=np.float32)
    validity_flags[:, 0] = face_valid.astype(np.float32)
    validity_flags[:, 1] = (emo2vec_valid | audeering_valid).astype(np.float32)
    validity_flags[:, 2] = bio_valid.astype(np.float32)

    # ── 9. Save V2 NPZ ──
    print("[9] Saving V2 features...")
    out_path = FEATURES_DIR / "kemocon_features_v2.npz"
    np.savez_compressed(
        str(out_path),
        # === Existing (preserved) ===
        face_probs_hse=face_probs_hse,       # (N, 8)
        face_embed=face_embed,               # (N, 1280)
        face_valid=face_valid,               # (N,)
        emo2vec_embed=emo2vec_embed,          # (N, 1024)
        emo2vec_probs=emo2vec_probs,          # (N, 9)
        emo2vec_valid=emo2vec_valid,          # (N,)
        audeering_avd=audeering_avd,          # (N, 3)
        audeering_valid=audeering_valid,      # (N,)
        bio_features=bio_features,            # (N, 15)
        bio_valid=bio_valid,                  # (N,)
        label_A_bin=label_A,                  # (N,)
        label_V_bin=label_V,                  # (N,)
        pid=pids,                             # (N,)
        pair_id=pair_ids,                     # (N,)
        seg_idx=seg_idxs,                    # (N,)
        # === NEW: K-FER ===
        kfer_probs=kfer_probs,               # (N, 7)
        kfer_entropy=kfer_entropy,           # (N,)
        kfer_quality=kfer_quality,           # (N,)
        kfer_valid=kfer_valid,               # (N,)
        # === NEW: Bio V2 quality ===
        bvp_valid=bvp_valid,                 # (N,)
        eda_valid=eda_valid,                 # (N,)
        hr_valid=hr_valid,                   # (N,)
        bio_quality=bio_quality,             # (N,)
        # === NEW: Face stats ===
        face_stats=face_stats,               # (N, 3)
        # === NEW: FACS Aux ===
        perclos=perclos,                     # (N,)
        ear_mean=ear_mean,                   # (N,)
        ear_conf=ear_conf,                   # (N,)
        facs_scores=facs_scores,             # (N, 6)
        facs_valid=facs_valid,               # (N,)
        # === NEW: Audio quality ===
        audio_quality=audio_quality,          # (N, 3)
        # === NEW: Cross-modal ===
        cross_modal=cross_modal,              # (N, 3)
        # === NEW: Validity flags ===
        validity_flags=validity_flags,        # (N, 3)
    )

    elapsed = time.time() - t0
    file_size = out_path.stat().st_size / 1e6

    print(f"\n{'='*60}")
    print(f"Feature Extraction V2 Complete!")
    print(f"{'='*60}")
    print(f"Output: {out_path} ({file_size:.1f} MB)")
    print(f"Time: {elapsed:.1f}s")
    print(f"\n15-Token Feature Summary (N={N}):")
    print(f"  T1  K-FER probs:       {kfer_probs.shape}  valid={kfer_valid.sum()}")
    print(f"  T2  K-FER meta:        entropy+quality     valid={kfer_valid.sum()}")
    print(f"  T3  Face stats:        {face_stats.shape}  valid={face_valid.sum()}")
    print(f"  T4  emo2vec probs:     {emo2vec_probs.shape}  valid={emo2vec_valid.sum()}")
    print(f"  T5  audeering AVD:     {audeering_avd.shape}  valid={audeering_valid.sum()}")
    print(f"  T6  Audio quality:     {audio_quality.shape}  (placeholder)")
    print(f"  T7  BVP features:      (N,4) from bio[0:4] valid={bvp_valid.sum()}")
    print(f"  T8  EDA features:      (N,5) from bio[4:9] valid={eda_valid.sum()}")
    print(f"  T9  HR+TEMP features:  (N,6) from bio[9:15] valid={hr_valid.sum()}")
    print(f"  T10 Bio quality:       (N,3) bvp/eda/hr valid")
    print(f"  T11 PERCLOS+EAR:       (N,2) valid={facs_valid.sum()}")
    print(f"  T12 FACS scores:       {facs_scores.shape}  valid={facs_valid.sum()}")
    print(f"  T13 Cross-modal:       {cross_modal.shape}")
    print(f"  T14 Validity flags:    {validity_flags.shape}")
    print(f"  T15 [CLS] token:       (learnable, in model)")


if __name__ == "__main__":
    main()
