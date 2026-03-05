"""
Teacher Cache — Save/Load Teacher Predictions for KD
=====================================================
Stores per-segment teacher outputs for offline distillation:
  - fused_repr:     (64,)  L2-normalized CLS representation
  - arousal_logit:  (1,)   arousal prediction (pre-sigmoid)
  - valence_logit:  (1,)   valence prediction (pre-sigmoid)
  - kfer_probs:     (7,)   K-FER soft labels
  - hse_probs:      (8,)   HSEmotion soft labels (for cross-label KD)

Total: ~200 bytes per segment → 3577 × 200 ≈ 700KB
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class TeacherCache:
    """Manages teacher prediction cache for KD."""

    def __init__(self, n_segments: int, d_model: int = 64):
        self.n_segments = n_segments
        self.d_model = d_model

        # Storage
        self.fused_repr = np.zeros((n_segments, d_model), dtype=np.float16)
        self.arousal_logit = np.zeros((n_segments, 1), dtype=np.float32)
        self.valence_logit = np.zeros((n_segments, 1), dtype=np.float32)
        self.kfer_probs = np.zeros((n_segments, 7), dtype=np.float32)
        self.hse_probs = np.zeros((n_segments, 8), dtype=np.float32)
        self.valid = np.zeros(n_segments, dtype=bool)

    def store(self, idx: int,
              fused_repr: np.ndarray,
              arousal_logit: float,
              valence_logit: float = 0.0,
              kfer_probs: Optional[np.ndarray] = None,
              hse_probs: Optional[np.ndarray] = None):
        """Store teacher output for one segment."""
        # L2-normalize fused_repr
        norm = np.linalg.norm(fused_repr)
        if norm > 1e-8:
            fused_repr = fused_repr / norm

        self.fused_repr[idx] = fused_repr.astype(np.float16)
        self.arousal_logit[idx, 0] = arousal_logit
        self.valence_logit[idx, 0] = valence_logit
        if kfer_probs is not None:
            self.kfer_probs[idx] = kfer_probs
        if hse_probs is not None:
            self.hse_probs[idx] = hse_probs
        self.valid[idx] = True

    def save(self, path: str):
        """Save cache to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "fused_repr": self.fused_repr,
            "arousal_logit": self.arousal_logit,
            "valence_logit": self.valence_logit,
            "kfer_probs": self.kfer_probs,
            "hse_probs": self.hse_probs,
            "valid": self.valid,
            "n_segments": self.n_segments,
            "d_model": self.d_model,
        }, str(path))

        file_size = path.stat().st_size / 1024
        n_valid = self.valid.sum()
        print(f"[TeacherCache] Saved: {path} ({file_size:.1f} KB)")
        print(f"  Valid entries: {n_valid}/{self.n_segments} ({n_valid/self.n_segments*100:.1f}%)")

    @classmethod
    def load(cls, path: str) -> "TeacherCache":
        """Load cache from disk."""
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        cache = cls(data["n_segments"], data["d_model"])
        cache.fused_repr = data["fused_repr"]
        cache.arousal_logit = data["arousal_logit"]
        cache.valence_logit = data["valence_logit"]
        cache.kfer_probs = data["kfer_probs"]
        cache.hse_probs = data["hse_probs"]
        cache.valid = data["valid"]
        return cache

    def get_batch(self, indices: np.ndarray, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Get teacher cache for a batch of indices."""
        return {
            "fused_repr": torch.from_numpy(
                self.fused_repr[indices].astype(np.float32)
            ).to(device),
            "arousal_logit": torch.from_numpy(
                self.arousal_logit[indices]
            ).to(device),
            "valence_logit": torch.from_numpy(
                self.valence_logit[indices]
            ).to(device),
        }


def build_teacher_cache(model, dataset, device="cuda",
                        batch_size=64) -> TeacherCache:
    """
    Build teacher cache from a trained KMERFusion model.

    Args:
        model: trained KMERFusion model (eval mode)
        dataset: KMERDataset instance
        device: compute device

    Returns:
        TeacherCache with all segments populated
    """
    from fusion.kmer_fusion import build_valid_mask

    model.eval()
    cache = TeacherCache(dataset.N, model.d_model)

    tokens = dataset.get_standardized_tokens(np.arange(dataset.N))
    token_tensors = {k: torch.from_numpy(v).float() for k, v in tokens.items()}

    face_v = torch.from_numpy(dataset.face_valid)
    audio_v = torch.from_numpy(dataset.audio_valid)
    bio_v = torch.from_numpy(dataset.bio_valid)
    mask = build_valid_mask(face_v, audio_v, bio_v)

    with torch.no_grad():
        for start in range(0, dataset.N, batch_size):
            end = min(start + batch_size, dataset.N)
            batch_feat = {k: v[start:end].to(device) for k, v in token_tensors.items()}
            batch_mask = mask[start:end].to(device)

            outputs = model(batch_feat, valid_mask=batch_mask)

            for i in range(end - start):
                idx = start + i
                cache.store(
                    idx=idx,
                    fused_repr=outputs["fused_repr"][i].cpu().numpy(),
                    arousal_logit=outputs["arousal"][i, 0].cpu().item(),
                    valence_logit=outputs.get("valence", torch.zeros(end-start, 1))[i, 0].cpu().item(),
                    kfer_probs=dataset.tokens["kfer_probs"][idx],
                    hse_probs=dataset.tokens.get("face_probs_hse",
                              np.zeros((dataset.N, 8)))[idx] if "face_probs_hse" in dataset.tokens else None,
                )

    return cache
