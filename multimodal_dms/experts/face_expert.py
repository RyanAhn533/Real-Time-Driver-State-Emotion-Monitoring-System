"""
Face Expert — HSEmotion (frozen EfficientNet-B0, AffectNet 8-class)
"""
import numpy as np
import torch
from pathlib import Path


class FaceExpert:
    """
    HSEmotion wrapper.
    Input: face frames (T, 3, 224, 224) float32 [0~255]
    Output: 8-class probs, penultimate embedding (1280d)
    """

    CLASSES = ["Angry", "Contempt", "Disgust", "Fear",
               "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, model_name="enet_b0_8_best_afew", device="cuda"):
        import torch as _torch
        _orig_load = _torch.load
        def _patched_load(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_load(*a, **kw)
        _torch.load = _patched_load

        from hsemotion.facial_emotions import HSEmotionRecognizer
        self.device = device
        self.recognizer = HSEmotionRecognizer(model_name=model_name, device=device)
        _torch.load = _orig_load

    @torch.no_grad()
    def extract(self, faces_np: np.ndarray) -> dict:
        """
        faces_np: (T, 3, 224, 224) float32, range [0, 255]
        Returns: {probs: (8,), embed: (1280,)} averaged over T frames
        """
        if faces_np is None or len(faces_np) == 0:
            return {"probs": np.zeros(8, dtype=np.float32),
                    "embed": np.zeros(1280, dtype=np.float32),
                    "valid": False}

        # Filter out zero/empty frames
        frame_norms = np.linalg.norm(faces_np.reshape(len(faces_np), -1), axis=1)
        valid_idx = np.where(frame_norms > 1.0)[0]

        if len(valid_idx) == 0:
            return {"probs": np.zeros(8, dtype=np.float32),
                    "embed": np.zeros(1280, dtype=np.float32),
                    "valid": False}

        # Sample up to 4 evenly-spaced frames
        if len(valid_idx) > 4:
            sample_idx = valid_idx[np.linspace(0, len(valid_idx) - 1, 4, dtype=int)]
        else:
            sample_idx = valid_idx

        all_probs = []
        all_embeds = []

        for idx in sample_idx:
            frame = faces_np[idx]  # (3, 224, 224)
            # Convert to HWC uint8 for HSEmotion API
            frame_hwc = (frame.transpose(1, 2, 0)).clip(0, 255).astype(np.uint8)

            # extract_features returns (1, D) raw features
            feat = self.recognizer.extract_features(frame_hwc)  # (1, D)
            all_embeds.append(feat[0])

            # predict_emotions returns (label, scores)
            _, scores = self.recognizer.predict_emotions(frame_hwc, logits=False)
            all_probs.append(scores[:8])  # first 8 = emotion probs

        avg_probs = np.mean(all_probs, axis=0).astype(np.float32)
        avg_embed = np.mean(all_embeds, axis=0).astype(np.float32)

        return {"probs": avg_probs,
                "embed": avg_embed,
                "valid": True}
