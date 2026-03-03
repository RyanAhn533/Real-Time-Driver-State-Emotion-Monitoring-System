"""
Real-time FER Inference (No CSV needed)
=========================================
검증/추론 시에는 CSV 없이 이미지 1장 → FaceMesh → AU 좌표 → 모델 → 감정.

사용법:
  inferencer = FERInferencer("best.pth")
  result = inferencer.predict(image_bgr)
  # result = {"emotion": "happy", "confidence": 0.92, "logits": [...]}
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image
from torchvision.transforms import functional as TF


# AU Region definitions (must match training)
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

# EAR landmarks
LEFT_EYE_IDX =  [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


class FERInferencer:
    """
    단일 이미지 FER 추론기.
    CSV 없이 직접 FaceMesh → AU 좌표 → 모델 추론.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = ckpt.get("config", {})
        self.label2id = ckpt.get("label2id", {})
        self.id2label = {v: k for k, v in self.label2id.items()}

        # Build model
        from models.fer_model import AUFERModel
        model_cfg = self.config if isinstance(self.config, dict) else {}

        self.img_size = model_cfg.get("GLOBAL_IMG_SIZE",
                        model_cfg.get("global_img_size", 224))
        d_emb = model_cfg.get("D_EMB", model_cfg.get("d_emb", 384))
        n_heads = model_cfg.get("N_HEAD", model_cfg.get("n_heads", 8))
        n_layers = model_cfg.get("N_LAYERS", model_cfg.get("n_fusion_layers", 1))
        num_classes = len(self.label2id)
        num_au = len(AU_REGIONS)

        self.model = AUFERModel(
            backbone_name=model_cfg.get("backbone", "mobilevitv2_100"),
            pretrained=False,
            num_au=num_au,
            num_classes=num_classes,
            d_emb=d_emb,
            n_heads=n_heads,
            n_fusion_layers=n_layers,
            img_size=self.img_size,
        ).to(self.device)

        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # Normalization
        self.mean = torch.tensor(
            self.model.backbone.norm_mean, dtype=torch.float32
        ).view(3, 1, 1).to(self.device)
        self.std = torch.tensor(
            self.model.backbone.norm_std, dtype=torch.float32
        ).view(3, 1, 1).to(self.device)

        # FaceMesh (lazy init)
        self._fm = None

    def _get_facemesh(self):
        if self._fm is None:
            import mediapipe as mp
            self._fm = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,  # video mode for speed
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        return self._fm

    def _extract_au_coords(self, rgb_img: np.ndarray) -> Optional[np.ndarray]:
        """
        RGB image → AU center coordinates (img_size 기준).
        Returns: [K, 2] numpy array or None if no face.
        """
        fm = self._get_facemesh()
        h, w = rgb_img.shape[:2]
        result = fm.process(rgb_img)

        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        coords = []

        for name, idx in AU_REGIONS:
            if isinstance(idx, tuple):
                xs = [landmarks[i].x for i in idx]
                ys = [landmarks[i].y for i in idx]
                cx = float(np.mean(xs)) * self.img_size
                cy = float(np.mean(ys)) * self.img_size
            else:
                cx = landmarks[idx].x * self.img_size
                cy = landmarks[idx].y * self.img_size
            coords.append([cx, cy])

        return np.array(coords, dtype=np.float32)

    def _extract_ear(self, rgb_img: np.ndarray) -> Tuple[float, float]:
        """EAR 추출 (졸음감지용). FaceMesh 이미 돌렸으면 캐시 사용."""
        fm = self._get_facemesh()
        h, w = rgb_img.shape[:2]
        result = fm.process(rgb_img)

        if not result.multi_face_landmarks:
            return 0.0, 0.0

        lm = result.multi_face_landmarks[0].landmark

        def _ear(indices):
            pts = [(lm[i].x * w, lm[i].y * h) for i in indices]
            def dist(a, b):
                return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
            v1 = dist(pts[1], pts[5])
            v2 = dist(pts[2], pts[4])
            hz = dist(pts[0], pts[3])
            return (v1 + v2) / (2.0 * hz) if hz > 1e-6 else 0.0

        return _ear(LEFT_EYE_IDX), _ear(RIGHT_EYE_IDX)

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> Optional[Dict]:
        """
        BGR image (OpenCV format) → emotion prediction.

        Args:
            image_bgr: [H, W, 3] BGR numpy array (from cv2 or RealSense)

        Returns:
            dict with emotion, confidence, logits, au_coords, ear
            or None if no face detected
        """
        # BGR → RGB, resize
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (self.img_size, self.img_size))

        # AU extraction
        au_coords = self._extract_au_coords(rgb_resized)
        if au_coords is None:
            return None

        # Image → tensor
        img_pil = Image.fromarray(rgb_resized)
        img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(self.device)
        img_tensor = (img_tensor - self.mean) / self.std

        # AU coords → tensor
        au_tensor = torch.tensor(au_coords, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forward
        logits = self.model(img_tensor, au_tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = probs.argmax().item()
        confidence = probs[pred_id].item()

        return {
            "emotion": self.id2label[pred_id],
            "emotion_id": pred_id,
            "confidence": confidence,
            "probs": {self.id2label[i]: round(probs[i].item(), 4)
                      for i in range(len(self.id2label))},
            "au_coords": au_coords,
        }

    @torch.no_grad()
    def get_expr_magnitude(self, image_bgr: np.ndarray) -> float:
        """
        Backbone only → expression magnitude (peak selection용).
        Full predict보다 빠름 (attention 안 돌림).
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (self.img_size, self.img_size))

        img_pil = Image.fromarray(rgb_resized)
        img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(self.device)
        img_tensor = (img_tensor - self.mean) / self.std

        mag = self.model.get_expr_magnitude(img_tensor)
        return mag.item()
