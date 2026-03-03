"""
Multimodal Fuser (Placeholder)
================================
최종 통합 모듈. 모든 modality를 합쳐 최종 출력 생성.

Current: FER only
Next:    FER + Audio A/V + Bio A/V → refined emotion
Future:  + External context (YOLO scene) → context-aware emotion

Context integration approach (planned):
  - External YOLO → scene labels → text description
  - Text → embedding (e.g., sentence-transformers)
  - Context embedding + emotion embedding → attention/concat → final output
"""

from typing import Dict, Optional, Any
from .emotion_refiner import EmotionRefiner
from .drowsiness_judge import DrowsinessJudge


class MultimodalFuser:
    """
    All modalities → final output.
    Incrementally built: start with FER only, add modalities later.
    """

    def __init__(self, fer_id2label: Dict[int, str],
                 enable_drowsiness: bool = True,
                 enable_av: bool = False,
                 enable_context: bool = False):

        self.emotion_refiner = EmotionRefiner(fer_id2label)
        self.drowsiness_judge = DrowsinessJudge() if enable_drowsiness else None
        self.enable_av = enable_av
        self.enable_context = enable_context

    def process(self,
                fer_pred: int,
                ear: Optional[float] = None,
                arousal: Optional[float] = None,
                valence: Optional[float] = None,
                context_embedding: Optional[Any] = None) -> Dict:
        """
        Single-frame processing.

        Returns dict with all available outputs:
          - base_emotion: FER 7-class prediction
          - refined_emotion: 10-class refined label (if A/V available)
          - drowsiness_level: 0/1/2 (if EAR available)
          - perclos: current PERCLOS value
          - arousal, valence: pass-through
        """
        result = {}

        # FER base emotion
        base_emotion = self.emotion_refiner.fer_id2label[fer_pred]
        result["base_emotion"] = base_emotion
        result["fer_pred_id"] = fer_pred

        # Refined emotion (if A/V available)
        if self.enable_av and arousal is not None:
            from .emotion_refiner import refine_emotion
            lid, lname = refine_emotion(base_emotion, arousal, valence)
            result["refined_emotion"] = lname
            result["refined_id"] = lid
        else:
            result["refined_emotion"] = base_emotion
            result["refined_id"] = fer_pred

        # Drowsiness
        if self.drowsiness_judge and ear is not None:
            self.drowsiness_judge.push_ear(ear)
            result["drowsiness_level"] = self.drowsiness_judge.judge(arousal)
            result["perclos"] = self.drowsiness_judge.get_perclos()

        # Pass-through
        result["arousal"] = arousal
        result["valence"] = valence

        # Context (future)
        if self.enable_context and context_embedding is not None:
            result["context"] = "TODO: attention merge with context embedding"

        return result
