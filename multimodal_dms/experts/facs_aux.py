"""
FACS Auxiliary Expert — EAR/PERCLOS + Geometric Emotion Scores
===============================================================
Extracts auxiliary features from face crops:
  1. EAR (Eye Aspect Ratio) per frame → PERCLOS over segment
  2. FACS-like geometric emotion indicators (landmark distances)

These are LOW-SNR auxiliary signals — used as additional tokens in fusion
with low weight. The main face signal comes from K-FER / HSEmotion probs.

MediaPipe is required for landmark extraction. If unavailable, returns
placeholder values with quality=0.

Outputs per segment:
  - perclos:     float — PERCLOS score [0, 1] (fraction eyes closed)
  - ear_mean:    float — mean EAR (both eyes average)
  - ear_conf:    float — EAR confidence [0, 1]
  - facs_scores: (6,)  — geometric emotion indicators
  - facs_valid:  bool  — whether landmarks were extracted
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

# EAR threshold for "eyes closed"
EAR_THRESHOLD = 0.21

# MediaPipe FaceMesh landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# FACS geometric feature landmarks
# These measure relative distances between facial keypoints
FACS_LANDMARKS = {
    # Mouth opening (inner lips)
    "mouth_open": {"top": 13, "bottom": 14},  # upper/lower lip
    # Brow raise (eyebrow to eye)
    "brow_left": {"brow": 70, "eye": 159},
    "brow_right": {"brow": 300, "eye": 386},
    # Lip corner (smile/frown)
    "lip_corner_left": {"corner": 61, "center": 0},
    "lip_corner_right": {"corner": 291, "center": 0},
    # Nose wrinkle (nasolabial fold)
    "nose_wrinkle": {"nose_tip": 1, "nose_bridge": 6},
}


def _compute_ear_from_landmarks(landmarks, eye_indices: List[int],
                                 w: float, h: float) -> float:
    """EAR from FaceMesh landmarks. Returns 0.0 if invalid."""
    try:
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
        def dist(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        v1 = dist(pts[1], pts[5])
        v2 = dist(pts[2], pts[4])
        hz = dist(pts[0], pts[3])
        if hz < 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * hz)
    except (IndexError, AttributeError):
        return 0.0


def _compute_facs_scores(landmarks, w: float, h: float) -> np.ndarray:
    """
    Compute 6 geometric FACS-like scores from landmarks.
    Returns: (6,) float32 normalized scores.
    """
    scores = np.zeros(6, dtype=np.float32)

    try:
        lm = landmarks

        # Normalize by face height (forehead to chin)
        face_h = np.sqrt(
            (lm[10].x * w - lm[152].x * w)**2 +
            (lm[10].y * h - lm[152].y * h)**2
        )
        if face_h < 1e-6:
            return scores

        # 0: Mouth openness
        mouth_open = abs(lm[13].y * h - lm[14].y * h) / face_h
        scores[0] = min(mouth_open * 10, 1.0)  # normalize

        # 1: Left brow raise
        brow_l = abs(lm[70].y * h - lm[159].y * h) / face_h
        scores[1] = min(brow_l * 5, 1.0)

        # 2: Right brow raise
        brow_r = abs(lm[300].y * h - lm[386].y * h) / face_h
        scores[2] = min(brow_r * 5, 1.0)

        # 3: Left lip corner height (relative to center)
        lip_l = (lm[0].y * h - lm[61].y * h) / face_h
        scores[3] = max(-1.0, min(lip_l * 5, 1.0))

        # 4: Right lip corner height
        lip_r = (lm[0].y * h - lm[291].y * h) / face_h
        scores[4] = max(-1.0, min(lip_r * 5, 1.0))

        # 5: Nose bridge compression (indicates wrinkle/frown)
        nose_comp = abs(lm[1].y * h - lm[6].y * h) / face_h
        scores[5] = min(nose_comp * 5, 1.0)

    except (IndexError, AttributeError):
        pass

    return scores


def compute_perclos(ear_sequence: List[float],
                    threshold: float = EAR_THRESHOLD) -> float:
    """PERCLOS: fraction of time eyes are closed in window."""
    if len(ear_sequence) == 0:
        return 0.0
    closed = sum(1 for ear in ear_sequence if ear < threshold)
    return closed / len(ear_sequence)


class FACSAuxExpert:
    """
    Extract EAR/PERCLOS and FACS geometric features from face crops.
    Requires MediaPipe FaceMesh for landmark extraction.
    """

    def __init__(self):
        self._fm = None
        self._available = None

    def _init_facemesh(self):
        """Lazy init MediaPipe FaceMesh. Returns True if available."""
        if self._available is not None:
            return self._available

        try:
            import mediapipe as mp
            self._fm = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3,
            )
            self._available = True
            print("[FACS Aux] MediaPipe FaceMesh initialized")
        except Exception as e:
            print(f"[FACS Aux] MediaPipe not available: {e}")
            self._available = False

        return self._available

    def extract(self, faces_np: np.ndarray) -> Dict:
        """
        Extract FACS auxiliary features from face crops.

        Args:
            faces_np: (T, 3, 224, 224) float32 [0-255]

        Returns:
            {
                "perclos":     float   — PERCLOS score [0, 1]
                "ear_mean":    float   — mean EAR (averaged over frames)
                "ear_conf":    float   — EAR confidence [0, 1]
                "facs_scores": (6,)    — geometric emotion indicators
                "facs_valid":  bool    — whether landmarks were extracted
            }
        """
        empty_result = {
            "perclos": 0.5,  # prior: unknown state
            "ear_mean": 0.25,  # prior: normal EAR
            "ear_conf": 0.0,
            "facs_scores": np.zeros(6, dtype=np.float32),
            "facs_valid": False,
        }

        if faces_np is None or len(faces_np) == 0:
            return empty_result

        # Check MediaPipe availability
        if not self._init_facemesh():
            return empty_result

        # Process frames
        ear_values = []
        all_facs = []
        n_landmark_ok = 0

        # Sample frames (process more for EAR temporal signal)
        T = len(faces_np)
        max_frames = min(T, 15)  # Up to 15 frames for PERCLOS
        if T > max_frames:
            indices = np.linspace(0, T - 1, max_frames, dtype=int)
        else:
            indices = np.arange(T)

        for t in indices:
            frame_chw = faces_np[t]
            if np.linalg.norm(frame_chw) < 1.0:
                continue

            # CHW → HWC uint8
            frame_hwc = frame_chw.transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
            h, w = frame_hwc.shape[:2]

            result = self._fm.process(frame_hwc)
            if not result.multi_face_landmarks:
                continue

            landmarks = result.multi_face_landmarks[0].landmark
            n_landmark_ok += 1

            # EAR
            ear_l = _compute_ear_from_landmarks(landmarks, LEFT_EYE, w, h)
            ear_r = _compute_ear_from_landmarks(landmarks, RIGHT_EYE, w, h)
            ear_avg = (ear_l + ear_r) / 2.0
            ear_values.append(ear_avg)

            # FACS geometric scores
            facs = _compute_facs_scores(landmarks, w, h)
            all_facs.append(facs)

        if n_landmark_ok == 0:
            return empty_result

        # Compute PERCLOS from EAR sequence
        perclos = compute_perclos(ear_values)
        ear_mean = float(np.mean(ear_values)) if ear_values else 0.25

        # EAR confidence: based on fraction of successful landmark extractions
        ear_conf = n_landmark_ok / max(len(indices), 1)

        # Average FACS scores
        facs_scores = np.mean(all_facs, axis=0).astype(np.float32) if all_facs else np.zeros(6, dtype=np.float32)

        return {
            "perclos": perclos,
            "ear_mean": ear_mean,
            "ear_conf": ear_conf,
            "facs_scores": facs_scores,
            "facs_valid": True,
        }

    def extract_precomputed(self, ear_sequence: List[float],
                             facs_scores: Optional[np.ndarray] = None) -> Dict:
        """
        Use pre-computed EAR values and FACS scores.
        Useful when MediaPipe was run in a different environment.
        """
        perclos = compute_perclos(ear_sequence) if ear_sequence else 0.5
        ear_mean = float(np.mean(ear_sequence)) if ear_sequence else 0.25
        ear_conf = 1.0 if ear_sequence else 0.0

        return {
            "perclos": perclos,
            "ear_mean": ear_mean,
            "ear_conf": ear_conf,
            "facs_scores": facs_scores if facs_scores is not None else np.zeros(6, dtype=np.float32),
            "facs_valid": facs_scores is not None,
        }
