"""
Emotion Refiner: FER 7-class × Arousal/Valence → 12 Refined Labels
=====================================================================

FER 7-class (AI Hub Korean dataset):
  angry, anxious, happy, hurt, neutral, sad, surprised

Arousal/Valence 기반 확장 → 최종 12개 클래스:
  0: neutral         — 기본
  1: calm             — neutral + low arousal
  2: happy            — 기본
  3: positive_engaged — happy + mid arousal (적극 참여)
  4: excited          — happy + high arousal
  5: sad              — 기본
  6: depressed        — sad + low arousal
  7: anxious          — 기본 (FER anxious)
  8: stressed         — anxious + high arousal
  9: angry            — 기본
  10: hurt            — 기본 (FER hurt, 상처/아픔)
  11: surprised       — 기본
  12: drowsy          — Agent에서 PerClos/bio로 판별 (여기선 미사용)

Note: drowsy(12)는 Agent/Gating에서 PerClos + arousal 기반으로 최종 결정.
      이 refiner에서는 FER + A/V만 다룸.
"""

import numpy as np
from typing import Dict, Tuple, Optional


# ── Final Label Mapping (12 classes, drowsy 포함 시 13) ──
REFINED_LABELS = {
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

# FER 7-class (AI Hub) × arousal → refined mapping
# (base_emotion, arousal_condition) → refined_label_id
REFINEMENT_RULES = {
    # neutral
    ("neutral", None):            0,
    ("neutral", "low"):           1,    # calm
    ("neutral", "mid"):           0,    # neutral
    ("neutral", "high"):          0,    # neutral (high arousal neutral = still neutral)
    # happy
    ("happy",   "low"):           2,    # happy
    ("happy",   "mid"):           3,    # positive engaged
    ("happy",   "high"):          4,    # excited
    ("happy",   None):            2,    # happy (no A/V)
    # sad
    ("sad",     "low"):           6,    # depressed
    ("sad",     "mid"):           5,    # sad
    ("sad",     "high"):          5,    # sad (high arousal sad = agitated sad)
    ("sad",     None):            5,    # sad (no A/V)
    # anxious (AI Hub specific)
    ("anxious", "low"):           7,    # anxious
    ("anxious", "mid"):           7,    # anxious
    ("anxious", "high"):          8,    # stressed
    ("anxious", None):            7,    # anxious (no A/V)
    # angry
    ("angry",   "low"):           9,    # angry
    ("angry",   "mid"):           9,    # angry
    ("angry",   "high"):          9,    # angry (rage merged into angry)
    ("angry",   None):            9,    # angry (no A/V)
    # hurt (AI Hub specific)
    ("hurt",    "low"):           10,   # hurt
    ("hurt",    "mid"):           10,   # hurt
    ("hurt",    "high"):          10,   # hurt
    ("hurt",    None):            10,   # hurt (no A/V)
    # surprised
    ("surprised", None):          11,
    ("surprised", "low"):         11,
    ("surprised", "mid"):         11,
    ("surprised", "high"):        11,
    # Legacy mappings (basic emotion names)
    ("surprise", None):           11,
    ("fear",    "low"):           7,    # map to anxious
    ("fear",    "mid"):           7,
    ("fear",    "high"):          8,    # map to stressed
    ("fear",    None):            7,
    ("disgust", None):            9,    # map to angry
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
