"""
K-FER Expert — Korean FER Model (MobileViTv2 + AU RoI + CrossAttention)
========================================================================
AI Hub 7-class: angry, anxious, happy, hurt, neutral, sad, surprised.

Outputs per segment:
  - kfer_probs:    (7,)  emotion probability distribution
  - kfer_entropy:  float Shannon entropy (model uncertainty)
  - kfer_quality:  float quality score [0, 1] (inverse entropy, normalized)
  - kfer_valid:    bool

Quality / uncertainty policy (for fusion token weighting):
  entropy < 1.0  → weight × 1.2 (confident)
  entropy < 1.5  → weight × 1.0 (normal)
  1.5 ≤ ent < 1.8 → weight × 0.5 (noisy)
  entropy ≥ 1.8  → mask (drop)
"""

import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional

# ── Paths ──
EMO_SYS_ROOT = Path("/home/ajy/Jetson_thor/emotion_system")
CKPT_PATH = EMO_SYS_ROOT / "result" / "best.pth"

# K-FER 7 classes (alphabetical, matching training CSV)
KFER_LABELS = ["angry", "anxious", "happy", "hurt", "neutral", "sad", "surprised"]
NUM_CLASSES = 7

# AU Region Definitions (8 regions, must match training)
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
NUM_AU = len(AU_REGIONS)

# Entropy thresholds
MAX_ENTROPY = np.log(NUM_CLASSES)  # ≈ 1.946


def _make_default_au_coords(img_size: int = 224) -> np.ndarray:
    """Default AU coords for aligned 224×224 face crops."""
    s = img_size
    return np.array([
        [s * 0.50, s * 0.22],  # forehead
        [s * 0.35, s * 0.38],  # eyes_left
        [s * 0.65, s * 0.38],  # eyes_right
        [s * 0.50, s * 0.52],  # nose
        [s * 0.22, s * 0.55],  # cheek_left
        [s * 0.78, s * 0.55],  # cheek_right
        [s * 0.50, s * 0.72],  # mouth
        [s * 0.50, s * 0.88],  # chin
    ], dtype=np.float32)


def _compute_entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(p * np.log(p)))


def _entropy_to_quality(entropy: float) -> float:
    """Convert entropy to quality score [0, 1]. Lower entropy = higher quality."""
    return float(max(0.0, 1.0 - entropy / MAX_ENTROPY))


class KFERExpert:
    """
    K-FER Model wrapper for feature extraction.
    Loads AUFERModel from checkpoint, runs inference on face crops.
    Uses default AU coords for aligned face crops.
    """

    def __init__(self, checkpoint: str = None, device: str = "cuda"):
        self.device = device
        ckpt_path = checkpoint or str(CKPT_PATH)

        # Add emotion_system to path
        emo_sys_str = str(EMO_SYS_ROOT)
        if emo_sys_str not in sys.path:
            sys.path.insert(0, emo_sys_str)

        from models.fer_model import AUFERModel

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        model_cfg = config.get("model", config)
        aug_cfg = config.get("augmentation", config)

        self.img_size = aug_cfg.get("global_img_size", 224)
        d_emb = model_cfg.get("d_emb", 384)
        n_heads = model_cfg.get("n_heads", 8)
        n_layers = model_cfg.get("n_fusion_layers", 1)
        num_classes = model_cfg.get("num_classes", NUM_CLASSES)
        backbone = model_cfg.get("backbone", "mobilevitv2_100")

        self.model = AUFERModel(
            backbone_name=backbone, pretrained=False,
            num_au=NUM_AU, num_classes=num_classes,
            d_emb=d_emb, n_heads=n_heads, n_fusion_layers=n_layers,
            img_size=self.img_size,
        )
        self.model.load_state_dict(ckpt["model"])
        self.model = self.model.to(device).eval()

        # Normalization (from backbone)
        self.norm_mean = torch.tensor(
            self.model.backbone.norm_mean, dtype=torch.float32
        ).view(3, 1, 1).to(device)
        self.norm_std = torch.tensor(
            self.model.backbone.norm_std, dtype=torch.float32
        ).view(3, 1, 1).to(device)

        # Default AU coords
        self.default_au = _make_default_au_coords(self.img_size)

        # Label mapping
        self.id2label = {i: lb for i, lb in enumerate(KFER_LABELS)}
        self.label2id = {lb: i for i, lb in enumerate(KFER_LABELS)}

        print(f"[K-FER Expert] Loaded: {num_classes}cls, {NUM_AU}AU, "
              f"d_emb={d_emb}, backbone={backbone}")

    @torch.no_grad()
    def extract(self, faces_np: np.ndarray,
                au_coords: Optional[np.ndarray] = None) -> Dict:
        """
        Extract K-FER features from face crops.

        Args:
            faces_np: (T, 3, 224, 224) float32 [0-255]
            au_coords: (8, 2) optional AU coordinates. Uses default if None.

        Returns:
            {
                "probs":   (7,) float32 — emotion probabilities
                "entropy": float — Shannon entropy
                "quality": float — quality score [0, 1]
                "top1_id": int — predicted class index
                "top1_label": str — predicted class name
                "top1_conf": float — top-1 confidence
                "valid": bool
            }
        """
        empty_result = {
            "probs": np.zeros(NUM_CLASSES, dtype=np.float32),
            "entropy": float(MAX_ENTROPY),
            "quality": 0.0,
            "top1_id": -1,
            "top1_label": "unknown",
            "top1_conf": 0.0,
            "valid": False,
        }

        if faces_np is None or len(faces_np) == 0:
            return empty_result

        # Filter out zero/empty frames
        valid_frames = []
        for t in range(len(faces_np)):
            if np.linalg.norm(faces_np[t]) > 1.0:
                valid_frames.append(faces_np[t])

        if len(valid_frames) == 0:
            return empty_result

        # Sample up to 4 evenly-spaced frames
        max_frames = min(4, len(valid_frames))
        if len(valid_frames) > max_frames:
            indices = np.linspace(0, len(valid_frames) - 1, max_frames, dtype=int)
            valid_frames = [valid_frames[i] for i in indices]

        # Batch inference
        batch = np.stack(valid_frames, axis=0)  # (N, 3, 224, 224)
        img_tensor = torch.from_numpy(batch).float().to(self.device) / 255.0
        img_tensor = (img_tensor - self.norm_mean) / self.norm_std

        coords = au_coords if au_coords is not None else self.default_au
        N = img_tensor.shape[0]
        au_tensor = torch.from_numpy(coords).float().unsqueeze(0).expand(N, -1, -1).to(self.device)

        logits = self.model(img_tensor, au_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (N, 7)

        # Average over frames
        avg_probs = np.mean(probs, axis=0).astype(np.float32)
        entropy = _compute_entropy(avg_probs)
        quality = _entropy_to_quality(entropy)
        top1_id = int(np.argmax(avg_probs))
        top1_conf = float(avg_probs[top1_id])

        return {
            "probs": avg_probs,
            "entropy": entropy,
            "quality": quality,
            "top1_id": top1_id,
            "top1_label": self.id2label.get(top1_id, "unknown"),
            "top1_conf": top1_conf,
            "valid": True,
        }

    @torch.no_grad()
    def extract_with_features(self, faces_np: np.ndarray,
                               au_coords: Optional[np.ndarray] = None) -> Dict:
        """
        Like extract() but also returns global feature embedding (for KD).

        Returns additional:
            "embed": (384,) float32 — global feature (d_emb dimension)
        """
        result = self.extract(faces_np, au_coords)
        if not result["valid"]:
            result["embed"] = np.zeros(384, dtype=np.float32)
            return result

        # Re-run with return_features=True for embedding
        valid_frames = [f for f in faces_np if np.linalg.norm(f) > 1.0]
        max_frames = min(4, len(valid_frames))
        if len(valid_frames) > max_frames:
            indices = np.linspace(0, len(valid_frames) - 1, max_frames, dtype=int)
            valid_frames = [valid_frames[i] for i in indices]

        batch = np.stack(valid_frames, axis=0)
        img_tensor = torch.from_numpy(batch).float().to(self.device) / 255.0
        img_tensor = (img_tensor - self.norm_mean) / self.norm_std

        coords = au_coords if au_coords is not None else self.default_au
        N = img_tensor.shape[0]
        au_tensor = torch.from_numpy(coords).float().unsqueeze(0).expand(N, -1, -1).to(self.device)

        logits, global_feat = self.model(img_tensor, au_tensor, return_features=True)
        embed = global_feat.mean(dim=0).cpu().numpy().astype(np.float32)

        result["embed"] = embed
        return result
