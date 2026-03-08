"""
Compound Emotion Mapper — Inference-Time Rule
==============================================
Maps K-FER base emotion × predicted arousal level → 13 compound labels.
NOT a training target — pure post-processing for showcase.

Based on emotion_refiner.py REFINEMENT_RULES.
"""

import numpy as np
from typing import Tuple, Optional


# ── 13-class compound labels ──
COMPOUND_LABELS = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "positive_engaged",
    4: "excited",
    5: "sad",
    6: "depressed",
    7: "anxious",
    8: "stressed",
    9: "angry",
    10: "hurt",
    11: "surprised",
    12: "drowsy",
}

# K-FER 7-class labels
KFER_LABELS = ["angry", "anxious", "happy", "hurt", "neutral", "sad", "surprised"]

# (base_emotion, arousal_level) → compound_label_id
COMPOUND_RULES = {
    # neutral
    ("neutral", "low"):    1,    # calm
    ("neutral", "mid"):    0,    # neutral
    ("neutral", "high"):   0,    # neutral
    ("neutral", None):     0,
    # happy
    ("happy", "low"):      2,    # happy
    ("happy", "mid"):      3,    # positive_engaged
    ("happy", "high"):     4,    # excited
    ("happy", None):       2,
    # sad
    ("sad", "low"):        6,    # depressed
    ("sad", "mid"):        5,    # sad
    ("sad", "high"):       5,    # sad (agitated)
    ("sad", None):         5,
    # anxious
    ("anxious", "low"):    7,    # anxious
    ("anxious", "mid"):    7,    # anxious
    ("anxious", "high"):   8,    # stressed
    ("anxious", None):     7,
    # angry
    ("angry", "low"):      9,    # angry
    ("angry", "mid"):      9,    # angry
    ("angry", "high"):     9,    # angry (rage → merged)
    ("angry", None):       9,
    # hurt
    ("hurt", "low"):       10,   # hurt
    ("hurt", "mid"):       10,   # hurt
    ("hurt", "high"):      10,   # hurt
    ("hurt", None):        10,
    # surprised
    ("surprised", "low"):  11,
    ("surprised", "mid"):  11,
    ("surprised", "high"): 11,
    ("surprised", None):   11,
}


def discretize_arousal(arousal: float) -> str:
    """Arousal [0, 1] → low / mid / high."""
    if arousal < 0.33:
        return "low"
    elif arousal < 0.66:
        return "mid"
    return "high"


class CompoundEmotionMapper:
    """
    Maps K-FER prediction + arousal prediction → compound emotion.
    Inference-time only.
    """

    def __init__(self):
        self.id2label = COMPOUND_LABELS
        self.kfer_labels = KFER_LABELS

    def map_single(self,
                   kfer_top1_id: int,
                   arousal_pred: Optional[float] = None,
                   is_drowsy: bool = False) -> Tuple[int, str]:
        """
        Map single sample.

        Args:
            kfer_top1_id: K-FER top-1 class index (0-6)
            arousal_pred: predicted arousal [0, 1] or None
            is_drowsy: whether drowsy head predicts drowsy

        Returns:
            (compound_id, compound_label)
        """
        if is_drowsy:
            return 12, "drowsy"

        base_emotion = self.kfer_labels[kfer_top1_id]

        if arousal_pred is not None:
            arousal_level = discretize_arousal(arousal_pred)
        else:
            arousal_level = None

        key = (base_emotion, arousal_level)
        if key in COMPOUND_RULES:
            cid = COMPOUND_RULES[key]
            return cid, self.id2label[cid]

        # Fallback to None arousal
        key_none = (base_emotion, None)
        if key_none in COMPOUND_RULES:
            cid = COMPOUND_RULES[key_none]
            return cid, self.id2label[cid]

        return 0, "neutral"

    def map_batch(self,
                  kfer_top1_ids: np.ndarray,
                  arousal_preds: Optional[np.ndarray] = None,
                  drowsy_preds: Optional[np.ndarray] = None) -> Tuple[np.ndarray, list]:
        """
        Map batch of samples.

        Args:
            kfer_top1_ids: (B,) int
            arousal_preds: (B,) float [0, 1] or None
            drowsy_preds:  (B,) int {0=alert, 1=drowsy, 2=sleeping} or None

        Returns:
            compound_ids: (B,) int
            compound_labels: list of str
        """
        B = len(kfer_top1_ids)
        compound_ids = np.zeros(B, dtype=np.int32)
        compound_labels = []

        for i in range(B):
            a = float(arousal_preds[i]) if arousal_preds is not None else None
            is_drowsy = (drowsy_preds is not None and int(drowsy_preds[i]) >= 1)
            cid, clabel = self.map_single(int(kfer_top1_ids[i]), a, is_drowsy)
            compound_ids[i] = cid
            compound_labels.append(clabel)

        return compound_ids, compound_labels

    def get_distribution(self, compound_ids: np.ndarray) -> dict:
        """
        Get compound emotion distribution for reporting.

        Returns:
            dict of {label: count}
        """
        from collections import Counter
        counts = Counter(compound_ids.tolist())
        return {self.id2label.get(k, f"id_{k}"): v
                for k, v in sorted(counts.items())}
