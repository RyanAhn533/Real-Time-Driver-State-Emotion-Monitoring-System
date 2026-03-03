"""
Emotion Refiner: 7 Basic Emotions × Arousal/Valence → 10 Refined Labels
=========================================================================

기본 감정 7개:
  angry, disgust, fear, happy, neutral, sad, surprise

Arousal/Valence에 따라 확장:
  - angry + high_arousal → rage (격노)
  - angry + low_arousal  → irritated (짜증)
  - sad + low_arousal    → depressed (우울)
  - happy + high_arousal → excited (흥분/기쁨)
  - fear + high_arousal  → panic (공포)
  등등...

최종 10개 라벨 예시 (프로젝트 요구사항에 따라 조정):
  0: neutral
  1: happy
  2: excited       (happy + high arousal)
  3: sad
  4: depressed     (sad + low arousal)
  5: angry
  6: rage          (angry + high arousal)
  7: fear
  8: panic         (fear + high arousal)
  9: surprise
"""

import numpy as np
from typing import Dict, Tuple, Optional


# ── Label Mapping ──
REFINED_LABELS = {
    0: "neutral",
    1: "happy",
    2: "excited",
    3: "sad",
    4: "depressed",
    5: "angry",
    6: "rage",
    7: "fear",
    8: "panic",
    9: "surprise",
}

# FER 7-class → refined mapping rules
# (base_emotion, arousal_condition) → refined_label
REFINEMENT_RULES = {
    ("neutral", None):            0,
    ("happy",   "low"):           1,
    ("happy",   "mid"):           1,
    ("happy",   "high"):          2,   # excited
    ("sad",     "low"):           4,   # depressed
    ("sad",     "mid"):           3,
    ("sad",     "high"):          3,
    ("angry",   "low"):           5,
    ("angry",   "mid"):           5,
    ("angry",   "high"):          6,   # rage
    ("fear",    "low"):           7,
    ("fear",    "mid"):           7,
    ("fear",    "high"):          8,   # panic
    ("surprise", None):           9,
    ("disgust", None):            5,   # map to angry group (adjust as needed)
}


def discretize_arousal(arousal: float) -> str:
    """Arousal [0, 1] → low / mid / high"""
    if arousal < 0.33:
        return "low"
    elif arousal < 0.66:
        return "mid"
    return "high"


def refine_emotion(base_emotion: str, arousal: Optional[float] = None,
                   valence: Optional[float] = None) -> Tuple[int, str]:
    """
    FER base emotion + Arousal/Valence → refined label.

    Args:
        base_emotion: one of 7 basic emotions (lowercase)
        arousal: [0, 1] from audio/bio sensor (None if unavailable)
        valence: [0, 1] from audio/bio sensor (unused for now, reserved)

    Returns:
        (label_id, label_name)
    """
    base = base_emotion.lower()

    if arousal is None:
        # No A/V data: use direct mapping
        key = (base, None)
        if key in REFINEMENT_RULES:
            lid = REFINEMENT_RULES[key]
            return lid, REFINED_LABELS[lid]
        # Fallback: try mid arousal
        key = (base, "mid")
        if key in REFINEMENT_RULES:
            lid = REFINEMENT_RULES[key]
            return lid, REFINED_LABELS[lid]
        return 0, "neutral"

    arousal_level = discretize_arousal(arousal)

    key = (base, arousal_level)
    if key in REFINEMENT_RULES:
        lid = REFINEMENT_RULES[key]
        return lid, REFINED_LABELS[lid]

    # Fallback
    key = (base, None)
    if key in REFINEMENT_RULES:
        lid = REFINEMENT_RULES[key]
        return lid, REFINED_LABELS[lid]

    return 0, "neutral"


class EmotionRefiner:
    """
    Batch processing wrapper.
    FER predictions + A/V signals → refined emotion labels.
    """

    def __init__(self, fer_id2label: Dict[int, str]):
        self.fer_id2label = fer_id2label

    def refine_batch(self, fer_preds: np.ndarray,
                     arousals: Optional[np.ndarray] = None,
                     valences: Optional[np.ndarray] = None):
        """
        Args:
            fer_preds: [B] FER class predictions (int)
            arousals: [B] arousal values or None
            valences: [B] valence values or None

        Returns:
            refined_ids: [B] refined label ids
            refined_names: [B] refined label names
        """
        B = len(fer_preds)
        ids = []
        names = []

        for i in range(B):
            base = self.fer_id2label[int(fer_preds[i])]
            a = float(arousals[i]) if arousals is not None else None
            v = float(valences[i]) if valences is not None else None
            lid, lname = refine_emotion(base, a, v)
            ids.append(lid)
            names.append(lname)

        return np.array(ids), names
