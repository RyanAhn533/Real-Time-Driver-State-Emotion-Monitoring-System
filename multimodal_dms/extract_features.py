"""
Feature Extraction — Pre-trained Experts on K-EmoCon
=====================================================
4개 expert로 3,577 세그먼트 특징 추출 → NPZ 캐싱

Usage:
  python extract_features.py
  python extract_features.py --skip_face        # face expert 건너뛰기
  python extract_features.py --skip_audio       # 모든 audio expert 건너뛰기
  python extract_features.py --skip_emotion2vec # emotion2vec만 건너뛰기
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time


def load_npz(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/ajy/Jetson_thor/data/precessed_data")
    parser.add_argument("--out_dir", default="features")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_face", action="store_true")
    parser.add_argument("--skip_audio", action="store_true")
    parser.add_argument("--skip_emotion2vec", action="store_true")
    parser.add_argument("--skip_bio", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load index
    df = pd.read_csv(data_dir / "segments_index.csv")
    N = len(df)
    print(f"Total segments: {N}")

    # ── Initialize experts ──
    face_expert = None
    emo2vec_expert = None
    audeering_expert = None

    if not args.skip_face:
        print("\n[1/4] Loading Face Expert (HSEmotion)...")
        try:
            from experts.face_expert import FaceExpert
            face_expert = FaceExpert(device=args.device)
            print("  ✓ Face Expert ready")
        except Exception as e:
            print(f"  ✗ Face Expert failed: {e}")

    if not args.skip_audio and not args.skip_emotion2vec:
        print("\n[2/4] Loading Audio Expert (emotion2vec)...")
        try:
            from experts.audio_expert import Emotion2VecExpert
            emo2vec_expert = Emotion2VecExpert(device=args.device)
            print(f"  ✓ emotion2vec ready (available={emo2vec_expert.available})")
        except Exception as e:
            print(f"  ✗ emotion2vec failed: {e}")
    elif args.skip_emotion2vec:
        print("\n[2/4] Skipping emotion2vec (--skip_emotion2vec)")

    if not args.skip_audio:
        print("\n[3/4] Loading Audio Expert (audeering)...")
        try:
            from experts.audio_expert import AudeeringExpert
            audeering_expert = AudeeringExpert(device=args.device)
            print(f"  ✓ audeering ready (available={audeering_expert.available})")
        except Exception as e:
            print(f"  ✗ audeering failed: {e}")

    if not args.skip_bio:
        print("\n[4/4] Bio Expert (neurokit2) — no initialization needed")

    # ── Allocate arrays ──
    face_probs = np.zeros((N, 8), dtype=np.float32)
    face_embed = np.zeros((N, 1280), dtype=np.float32)
    face_valid = np.zeros(N, dtype=bool)

    emo2vec_embed = np.zeros((N, 1024), dtype=np.float32)
    emo2vec_probs = np.zeros((N, 9), dtype=np.float32)
    emo2vec_valid = np.zeros(N, dtype=bool)

    audeering_avd = np.zeros((N, 3), dtype=np.float32)
    audeering_valid = np.zeros(N, dtype=bool)

    bio_features = np.zeros((N, 15), dtype=np.float32)
    bio_valid = np.zeros(N, dtype=bool)

    labels_A = np.zeros(N, dtype=np.int32)
    labels_V = np.zeros(N, dtype=np.int32)
    pids = np.zeros(N, dtype=np.int32)
    pair_ids = np.zeros(N, dtype=np.int32)
    seg_idxs = np.zeros(N, dtype=np.int32)

    # ── Extract features ──
    print(f"\n{'='*60}")
    print("Extracting features...")
    print(f"{'='*60}")
    t0 = time.time()

    for i, row in tqdm(df.iterrows(), total=N, desc="Extracting"):
        pid = int(row["pid"])
        seg_idx = int(row["seg_idx"])
        pids[i] = pid
        pair_ids[i] = int(row.get("pair_id", -1))
        seg_idxs[i] = seg_idx
        labels_A[i] = int(row.get("label_A_binary", int(row["label_ext_A"] >= 3)))
        labels_V[i] = int(row.get("label_V_binary", int(row["label_ext_V"] >= 3)))

        # ── Face ──
        if face_expert is not None and row.get("has_video_face", 0) == 1:
            try:
                vf_path = data_dir / row["video_face_path"]
                vf_data = load_npz(vf_path)
                faces = vf_data.get("faces", None)
                if faces is not None and len(faces) > 0:
                    result = face_expert.extract(faces)
                    face_probs[i] = result["probs"]
                    face_embed[i] = result["embed"]
                    face_valid[i] = result["valid"]
            except Exception as e:
                pass

        # ── Audio (emotion2vec) ──
        if emo2vec_expert is not None and emo2vec_expert.available and row.get("has_audio", 0) == 1:
            try:
                aud_path = data_dir / row["audio_path"]
                aud_data = load_npz(aud_path)
                waveform = aud_data.get("audio", None)
                sr = int(aud_data.get("sr", 16000))
                if waveform is not None:
                    result = emo2vec_expert.extract(waveform, sr=sr)
                    emo2vec_embed[i] = result["embed"]
                    probs = result["probs"]
                    if len(probs) >= 9:
                        emo2vec_probs[i] = probs[:9]
                    else:
                        emo2vec_probs[i, :len(probs)] = probs
                    emo2vec_valid[i] = result["valid"]
            except Exception as e:
                pass

        # ── Audio (audeering) ──
        if audeering_expert is not None and audeering_expert.available and row.get("has_audio", 0) == 1:
            try:
                aud_path = data_dir / row["audio_path"]
                aud_data = load_npz(aud_path)
                waveform = aud_data.get("audio", None)
                sr = int(aud_data.get("sr", 16000))
                if waveform is not None:
                    result = audeering_expert.extract(waveform, sr=sr)
                    audeering_avd[i] = result["avd"]
                    audeering_valid[i] = result["valid"]
            except Exception as e:
                pass

        # ── Bio ──
        if not args.skip_bio:
            try:
                seg_path = data_dir / row["path"]
                seg_data = load_npz(seg_path)
                from experts.bio_expert import extract_bio_features
                bio_features[i] = extract_bio_features(seg_data)
                bio_valid[i] = True
            except Exception as e:
                pass

    elapsed = time.time() - t0

    # ── Save ──
    out_path = out_dir / "kemocon_features.npz"
    np.savez_compressed(
        out_path,
        face_probs=face_probs,
        face_embed=face_embed,
        face_valid=face_valid,
        emo2vec_embed=emo2vec_embed,
        emo2vec_probs=emo2vec_probs,
        emo2vec_valid=emo2vec_valid,
        audeering_avd=audeering_avd,
        audeering_valid=audeering_valid,
        bio_features=bio_features,
        bio_valid=bio_valid,
        label_A_bin=labels_A,
        label_V_bin=labels_V,
        pid=pids,
        pair_id=pair_ids,
        seg_idx=seg_idxs,
    )

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"\nModality coverage:")
    print(f"  Face (HSEmotion):  {face_valid.sum()}/{N} ({face_valid.mean()*100:.1f}%)")
    print(f"  emotion2vec:       {emo2vec_valid.sum()}/{N} ({emo2vec_valid.mean()*100:.1f}%)")
    print(f"  audeering:         {audeering_valid.sum()}/{N} ({audeering_valid.mean()*100:.1f}%)")
    print(f"  Bio features:      {bio_valid.sum()}/{N} ({bio_valid.mean()*100:.1f}%)")
    print(f"\nLabel distribution:")
    print(f"  A_binary: 0={np.sum(labels_A==0)}, 1={np.sum(labels_A==1)}")
    print(f"  V_binary: 0={np.sum(labels_V==0)}, 1={np.sum(labels_V==1)}")


if __name__ == "__main__":
    main()
