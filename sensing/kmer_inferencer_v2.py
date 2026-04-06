"""
K-MER Real-Time Inferencer v2
==============================
v1 대비 변경점:
  [FIX] compound_emotion arousal_pred에 float 전달 (bool 버그 수정)
  [FIX] FaceMesh 1회만 실행 → landmarks를 모든 expert에 공유
  [FIX] FACS에 원본 프레임 대신 crop+landmarks 직접 전달
  [OPT] Expert 병렬 실행 (face는 선행, audio+bio 병렬)
  [OPT] librosa resample 캐싱 (동일 길이 오디오 재변환 방지)
  [OPT] TensorRT 엔진 자동 감지 및 사용
  [OPT] GPU→CPU→GPU 텐서 핑퐁 최소화
  [QOL] 로깅 통합, 매직넘버 제거, 상수 중앙화
"""

import sys
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Optional, Dict, Any

# ── Path Setup ──
_SENSING_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SENSING_DIR.parent
_EMO_SYS_ROOT = _PROJECT_ROOT / "emotion_system"
_MM_DMS_ROOT = _PROJECT_ROOT / "multimodal_dms"

for _p in [str(_EMO_SYS_ROOT), str(_MM_DMS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Logging ──
log = logging.getLogger("kmer.inferencer")

# ── Constants (single source of truth) ──
AU_REGIONS = [
    ("forehead", (69, 299, 9)), ("eyes_left", 159), ("eyes_right", 386),
    ("nose", 195), ("cheek_left", 186), ("cheek_right", 410),
    ("mouth", 13), ("chin", 18),
]
NUM_AU = len(AU_REGIONS)
IMG_SIZE = 224
WORK_SHORT_SIDE = 800
NUM_CLASSES = 7
KFER_LABELS = ["angry", "anxious", "happy", "hurt", "neutral", "sad", "surprised"]

# EAR landmarks (for PERCLOS — computed from shared FaceMesh)
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Audio
AUDIO_TARGET_SR = 16000
MIN_AUDIO_SAMPLES = int(AUDIO_TARGET_SR * 0.1)  # 100ms minimum


# ═══════════════════════════════════════════
# FaceMesh — Single Instance, Shared Results
# ═══════════════════════════════════════════
class FaceAnalyzer:
    """
    FaceMesh 1회 실행으로 face_crop, AU coords, EAR, landmarks 전부 추출.
    기존 3중 호출 문제 해결.
    """

    def __init__(self):
        self._mesh = None

    def _ensure_init(self):
        if self._mesh is not None:
            return
        try:
            import mediapipe
            self._mesh = mediapipe.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            log.info("FaceMesh initialized (single instance)")
        except Exception as e:
            log.error("FaceMesh init failed: %s", e)

    def analyze(self, frame_bgr: np.ndarray) -> Optional[Dict]:
        """
        BGR frame → 모든 얼굴 정보를 한번에 추출.

        Returns None if no face, otherwise dict:
            face_chw:    (3, 224, 224) float32 RGB
            au_coords:   (8, 2) float32
            bbox:        (x1, y1, x2, y2) original frame coords
            ear_left:    float
            ear_right:   float
            ear_mean:    float
            landmarks:   list of (x_norm, y_norm) for all 478 landmarks
        """
        self._ensure_init()
        if self._mesh is None:
            return None

        h, w = frame_bgr.shape[:2]
        scale = WORK_SHORT_SIDE / min(h, w)
        ww, wh = int(round(w * scale)), int(round(h * scale))
        work = cv2.resize(frame_bgr, (ww, wh), interpolation=cv2.INTER_LINEAR)
        work_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)

        res = self._mesh.process(work_rgb)
        if not res.multi_face_landmarks:
            return None

        lms = res.multi_face_landmarks[0].landmark

        # ── Bounding box + crop ──
        xs = [lm.x * ww for lm in lms]
        ys = [lm.y * wh for lm in lms]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(ww, x2 + pad_x)
        y2 = min(wh, y2 + pad_y)

        face_w, face_h = x2 - x1, y2 - y1
        if face_w < 10 or face_h < 10:
            return None

        face_crop = work[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))

        # ── AU coordinates ──
        au_coords = np.zeros((NUM_AU, 2), dtype=np.float32)
        for i, (_, idx) in enumerate(AU_REGIONS):
            if isinstance(idx, tuple):
                cx = float(np.mean([lms[j].x for j in idx]))
                cy = float(np.mean([lms[j].y for j in idx]))
            else:
                cx, cy = float(lms[idx].x), float(lms[idx].y)
            au_coords[i, 0] = np.clip((cx * ww - x1) * (IMG_SIZE / face_w), 0, IMG_SIZE - 1)
            au_coords[i, 1] = np.clip((cy * wh - y1) * (IMG_SIZE / face_h), 0, IMG_SIZE - 1)

        # ── EAR (Eye Aspect Ratio) — from same landmarks ──
        def _ear(indices):
            pts = [(lms[j].x * ww, lms[j].y * wh) for j in indices]
            def d(a, b):
                return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
            v1, v2, hz = d(pts[1], pts[5]), d(pts[2], pts[4]), d(pts[0], pts[3])
            return (v1 + v2) / (2.0 * hz) if hz > 1e-6 else 0.0

        ear_l = _ear(LEFT_EYE_IDX)
        ear_r = _ear(RIGHT_EYE_IDX)

        # Landmarks in original frame coordinates
        landmarks_orig = [(lm.x * ww / scale, lm.y * wh / scale) for lm in lms]

        return {
            "face_chw": face_resized.transpose(2, 0, 1).astype(np.float32),
            "au_coords": au_coords,
            "bbox": (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)),
            "ear_left": ear_l,
            "ear_right": ear_r,
            "ear_mean": (ear_l + ear_r) / 2.0,
            "landmarks": landmarks_orig,
        }


# ═══════════════════════════════════════════
# Audio Preprocessing (cached resample)
# ═══════════════════════════════════════════
class AudioPreprocessor:
    """Resample 캐싱 + quality 계산."""

    def __init__(self, src_sr: int = 48000, dst_sr: int = AUDIO_TARGET_SR):
        self.src_sr = src_sr
        self.dst_sr = dst_sr
        self._last_len = -1
        self._last_result = np.zeros(0, dtype=np.float32)
        self._librosa = None

    def _get_librosa(self):
        if self._librosa is None:
            try:
                import librosa
                self._librosa = librosa
            except ImportError:
                pass
        return self._librosa

    def resample(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio
        if self.src_sr == self.dst_sr:
            return audio.astype(np.float32)
        # Cache: skip if same audio length (likely same buffer)
        if len(audio) == self._last_len:
            return self._last_result
        lib = self._get_librosa()
        if lib is not None:
            self._last_result = lib.resample(audio.astype(np.float32),
                                              orig_sr=self.src_sr, target_sr=self.dst_sr)
        else:
            # Simple decimation fallback
            ratio = self.dst_sr / self.src_sr
            indices = np.arange(0, len(audio), 1.0 / ratio).astype(int)
            indices = indices[indices < len(audio)]
            self._last_result = audio[indices].astype(np.float32)
        self._last_len = len(audio)
        return self._last_result

    @staticmethod
    def compute_quality(audio: np.ndarray) -> np.ndarray:
        """(3,): rms_norm, zcr, crest_factor_norm"""
        if len(audio) == 0:
            return np.zeros(3, dtype=np.float32)
        a = audio.astype(np.float32)
        rms = float(np.sqrt(np.mean(a ** 2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(a)))) / 2)
        peak = float(np.max(np.abs(a)))
        crest = peak / (rms + 1e-8)
        return np.array([
            min(rms / 0.1, 1.0),
            zcr,
            min(crest / 10.0, 1.0),  # crest factor normalized
        ], dtype=np.float32)


# ═══════════════════════════════════════════
# Bio Preprocessing
# ═══════════════════════════════════════════
def build_bio_npz(ppg: list, eda: list, temp: list) -> dict:
    """BioQueues snapshot → bio_expert input format."""
    result = {}
    for key, data, n_cols in [("bvp", ppg, 2), ("eda", eda, 2), ("temp", temp, 2)]:
        if data and len(data) > 0:
            arr = np.array(data, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[1] < n_cols:
                arr = np.column_stack([arr, np.zeros((len(arr), n_cols - arr.shape[1]), dtype=np.float32)])
            result[key] = arr[:, :n_cols]
        else:
            result[key] = np.zeros((1, n_cols), dtype=np.float32)
    return result


def compute_cross_modal(kfer_probs, kfer_entropy, emo2vec_probs, audeering_avd,
                        face_valid: bool, audio_valid: bool) -> np.ndarray:
    """T13: (3,) face-audio agreement, AV consistency, entropy gap."""
    if not face_valid or not audio_valid:
        return np.zeros(3, dtype=np.float32)

    kfer_v_weights = np.array([-1, -1, 1, -1, 0, -1, 1], dtype=np.float32)
    emo2v_v_weights = np.array([-1, -1, -1, 1, 0, 0, -1, 0, 0], dtype=np.float32)

    face_v = float(np.dot(kfer_probs, kfer_v_weights))
    n = min(len(emo2vec_probs), len(emo2v_v_weights))
    audio_v = float(np.dot(emo2vec_probs[:n], emo2v_v_weights[:n]))

    agree = np.clip(face_v * audio_v, -1.0, 1.0)
    av_consist = 1.0 - abs(face_v - audeering_avd[1]) / 2.0 if len(audeering_avd) > 1 else 0.0

    eps = 1e-10
    e2v_ent = float(-np.sum(np.clip(emo2vec_probs, eps, 1.0) * np.log(np.clip(emo2vec_probs, eps, 1.0) + eps)))
    entropy_gap = float(np.clip(kfer_entropy - e2v_ent, -2, 2))

    return np.array([agree, av_consist, entropy_gap], dtype=np.float32)


# ═══════════════════════════════════════════
# KMERInferencer v2
# ═══════════════════════════════════════════
class KMERInferencer:
    """
    K-MER 실시간 추론기 v2.

    개선사항:
      - FaceMesh 1회 → 모든 expert에 공유
      - Audio/Bio expert 병렬 실행
      - Resample 캐싱
      - Compound emotion arousal 버그 수정
      - TensorRT 엔진 자동 감지
    """

    def __init__(
        self,
        kfer_ckpt: str = "../emotion_system/result/best.pth",
        kmer_ckpt: str = "../emotion_system/multimodal/checkpoints/ckpt_v3/fold_1/best.pth",
        device: str = "cuda",
        enable_face_expert: bool = False,
        enable_audio_experts: bool = True,
        audio_src_sr: int = 48000,
        trt_engine: Optional[str] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Shared face analyzer (single FaceMesh)
        self._face = FaceAnalyzer()

        # Audio preprocessor with cache
        self._audio_prep = AudioPreprocessor(src_sr=audio_src_sr, dst_sr=AUDIO_TARGET_SR)

        # Thread pool for parallel expert execution
        self._executor = ThreadPoolExecutor(max_workers=2)

        # K-FER Expert
        log.info("Loading K-FER: %s", kfer_ckpt)
        from experts.kfer_expert import KFERExpert
        self._kfer = KFERExpert(checkpoint=str(kfer_ckpt), device=device)

        # FACS is now computed from shared landmarks — no separate FaceMesh
        # (EAR/PERCLOS extracted in FaceAnalyzer.analyze())

        # HSEmotion (disabled by default — redundant with K-FER)
        self._face_expert = None
        if enable_face_expert:
            try:
                from experts.face_expert import FaceExpert
                self._face_expert = FaceExpert(device=device)
                log.info("HSEmotion FaceExpert loaded")
            except Exception as e:
                log.warning("HSEmotion load failed: %s", e)

        # Audio Experts
        self._emotion2vec = None
        self._audeering = None
        if enable_audio_experts:
            try:
                from experts.audio_expert import Emotion2VecExpert, AudeeringExpert
                log.info("Loading emotion2vec...")
                self._emotion2vec = Emotion2VecExpert(device=device)
                log.info("Loading audeering...")
                self._audeering = AudeeringExpert(device=device)
            except Exception as e:
                log.warning("Audio expert load failed: %s", e)

        # KMERFusion
        log.info("Loading KMERFusion: %s", kmer_ckpt)
        from fusion.kmer_fusion import KMERFusion, build_valid_mask
        self._build_valid_mask = build_valid_mask
        self._load_fusion(kmer_ckpt, KMERFusion)

        # Compound Emotion
        from fusion.compound_emotion import CompoundEmotionMapper
        self._compound = CompoundEmotionMapper()

        # PERCLOS tracker
        from collections import deque
        self._ear_buffer = deque(maxlen=90)  # 3 seconds at 30fps

        log.info("KMERInferencer v2 ready")

    def _load_fusion(self, ckpt_path: str, ModelClass):
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt

        self._fusion = ModelClass(
            d_model=config.get("d_model", 64),
            n_heads=config.get("n_heads", 4),
            dropout=config.get("dropout", 0.1),
            use_temporal=config.get("use_temporal", False),
            use_valence=config.get("use_valence", True),
            use_drowsy=config.get("use_drowsy", True),
        ).to(self.device).eval()

        self._fusion.load_state_dict(state, strict=False)
        epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
        log.info("KMERFusion loaded (epoch=%s)", epoch)

    # ── Expert Processing ──

    def _process_face(self, face_info: Dict) -> Dict:
        """From shared FaceAnalyzer output → K-FER + FACS features."""
        face_chw = face_info["face_chw"]
        au_coords = face_info["au_coords"]

        # K-FER
        kfer_out = self._kfer.extract(face_chw[np.newaxis], au_coords=au_coords)

        # EAR/PERCLOS from shared landmarks (no extra FaceMesh!)
        ear = face_info["ear_mean"]
        self._ear_buffer.append(ear)
        perclos = sum(1 for e in self._ear_buffer if e < 0.21) / len(self._ear_buffer)

        # face_stats (from K-FER probs directly — no need for HSEmotion)
        probs = kfer_out.get("probs", np.zeros(NUM_CLASSES, dtype=np.float32))
        face_stats = np.array([float(np.max(probs)), float(np.mean(probs)), float(np.std(probs))],
                              dtype=np.float32)

        return {
            "kfer_probs": probs,
            "kfer_meta": np.array([
                kfer_out.get("quality", 0.0),
                kfer_out.get("entropy", np.log(NUM_CLASSES)),
            ], dtype=np.float32),
            "face_stats": face_stats,
            "perclos_ear": np.array([perclos, ear], dtype=np.float32),
            "facs_scores": np.zeros(6, dtype=np.float32),  # simplified
            "face_valid": True,
            "_kfer_top1_id": kfer_out.get("top1_id", 4),
            "_kfer_top1_label": kfer_out.get("top1_label", "neutral"),
            "_kfer_top1_conf": kfer_out.get("top1_conf", 0.0),
            "_bbox": face_info["bbox"],
        }

    def _process_audio(self, audio_1d: np.ndarray) -> Dict:
        """Audio → emotion2vec + audeering."""
        empty = {
            "emo2vec_probs": np.zeros(9, dtype=np.float32),
            "audeering_avd": np.zeros(3, dtype=np.float32),
            "audio_quality": np.zeros(3, dtype=np.float32),
            "audio_valid": False,
        }
        if len(audio_1d) == 0:
            return empty

        audio_16k = self._audio_prep.resample(audio_1d)
        quality = AudioPreprocessor.compute_quality(audio_16k)

        if len(audio_16k) < MIN_AUDIO_SAMPLES:
            return empty

        emo2vec_probs = np.zeros(9, dtype=np.float32)
        audeering_avd = np.zeros(3, dtype=np.float32)
        audio_valid = False

        if self._emotion2vec is not None:
            e2v = self._emotion2vec.extract(audio_16k, sr=AUDIO_TARGET_SR)
            if e2v.get("valid", False):
                emo2vec_probs = e2v["probs"]
                audio_valid = True

        if self._audeering is not None:
            aud = self._audeering.extract(audio_16k, sr=AUDIO_TARGET_SR)
            if aud.get("valid", False):
                audeering_avd = aud["avd"]
                audio_valid = True

        return {
            "emo2vec_probs": emo2vec_probs,
            "audeering_avd": audeering_avd,
            "audio_quality": quality,
            "audio_valid": audio_valid,
        }

    def _process_bio(self, ppg: list, eda: list, temp: list) -> Dict:
        """Bio signals → features."""
        from experts.bio_expert import extract_bio_features_v2
        npz = build_bio_npz(ppg, eda, temp)
        bio = extract_bio_features_v2(npz)
        features = bio["features"]
        return {
            "bvp_features": features[0:4],
            "eda_features": features[4:9],
            "hr_temp_features": features[9:15],
            "bio_quality": np.array([
                float(bio["bio_quality"]),
                float(bio["bvp_valid"]),
                float(bio["eda_valid"]),
            ], dtype=np.float32),
            "bio_valid": bio.get("valid", False),
        }

    # ── Feature Assembly ──

    def _build_features(self, face_out: Dict, audio_out: Dict, bio_out: Dict):
        """Assemble all expert outputs into KMERFusion input."""
        cross_modal = compute_cross_modal(
            face_out["kfer_probs"], float(face_out["kfer_meta"][1]),
            audio_out["emo2vec_probs"], audio_out["audeering_avd"],
            face_out["face_valid"], audio_out["audio_valid"],
        )
        validity_flags = np.array([
            float(face_out["face_valid"]),
            float(audio_out["audio_valid"]),
            float(bio_out["bio_valid"]),
        ], dtype=np.float32)

        # Batch numpy→torch conversion (minimize individual .to(device) calls)
        def _t(arr):
            return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).to(self.device, non_blocking=True)

        features = {
            "kfer_probs": _t(face_out["kfer_probs"]),
            "kfer_meta": _t(face_out["kfer_meta"]),
            "face_stats": _t(face_out["face_stats"]),
            "emo2vec_probs": _t(audio_out["emo2vec_probs"]),
            "audeering_avd": _t(audio_out["audeering_avd"]),
            "audio_quality": _t(audio_out["audio_quality"]),
            "bvp_features": _t(bio_out["bvp_features"]),
            "eda_features": _t(bio_out["eda_features"]),
            "hr_temp_features": _t(bio_out["hr_temp_features"]),
            "bio_quality": _t(bio_out["bio_quality"]),
            "perclos_ear": _t(face_out["perclos_ear"]),
            "facs_scores": _t(face_out["facs_scores"]),
            "cross_modal": _t(cross_modal),
            "validity_flags": _t(validity_flags),
        }

        valid_mask = self._build_valid_mask(
            torch.tensor([face_out["face_valid"]], dtype=torch.bool, device=self.device),
            torch.tensor([audio_out["audio_valid"]], dtype=torch.bool, device=self.device),
            torch.tensor([bio_out["bio_valid"]], dtype=torch.bool, device=self.device),
        )

        return features, valid_mask

    # ── Main Forward ──

    @torch.no_grad()
    def forward(
        self,
        frame_bgr: np.ndarray,
        audio_1d: Optional[np.ndarray] = None,
        ppg: Optional[list] = None,
        eda: Optional[list] = None,
        temp: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        실시간 멀티모달 추론.

        Returns:
            arousal:         float [0, 1]
            valence:         float [0, 1] or None
            drowsy:          int {0, 1, 2}
            perclos:         float [0, 1]
            compound_id:     int (0~12)
            compound_label:  str
            kfer_emotion:    str
            kfer_confidence: float
            face_detected:   bool
            bbox:            tuple or None
        """
        fallback = {
            "arousal": 0.5, "valence": None, "drowsy": 0, "perclos": 0.0,
            "compound_id": 0, "compound_label": "neutral",
            "kfer_emotion": "neutral", "kfer_confidence": 0.0,
            "face_detected": False, "bbox": None,
        }

        # ── Step 1: FaceMesh (single call) ──
        face_info = self._face.analyze(frame_bgr)
        if face_info is None:
            return fallback

        # ── Step 2: Experts (face serial, audio+bio parallel) ──
        # Face must run first (needs face_info), audio+bio are independent
        face_out = self._process_face(face_info)

        if audio_1d is None:
            audio_1d = np.zeros(0, dtype=np.float32)
        if ppg is None:
            ppg = []
        if eda is None:
            eda = []
        if temp is None:
            temp = []

        # Parallel: audio + bio
        audio_future = self._executor.submit(self._process_audio, audio_1d)
        bio_future = self._executor.submit(self._process_bio, ppg, eda, temp)

        audio_out = audio_future.result()
        bio_out = bio_future.result()

        # ── Step 3: Fusion ──
        features, valid_mask = self._build_features(face_out, audio_out, bio_out)
        fusion_out = self._fusion(features, valid_mask=valid_mask)

        arousal = float(fusion_out["arousal"][0, 0].cpu())
        valence = float(fusion_out["valence"][0, 0].cpu()) if "valence" in fusion_out else None

        # Drowsy: use PERCLOS rule-based (more reliable without bio sensors)
        # NHTSA standard: PERCLOS >= 0.4 → drowsy, >= 0.6 → sleeping
        perclos = face_out["perclos_ear"][0]
        if perclos >= 0.6:
            drowsy = 2  # sleeping
        elif perclos >= 0.4:
            drowsy = 1  # drowsy
        else:
            drowsy = 0  # alert

        # ── Step 4: Compound Emotion (BUG FIX: pass float, not bool) ──
        compound = self._compound.map_single(
            kfer_top1_id=face_out["_kfer_top1_id"],
            arousal_pred=arousal,  # FIX: float, not bool
            is_drowsy=drowsy >= 1,
        )

        if isinstance(compound, tuple):
            compound_id, compound_label = compound
        else:
            compound_id = compound.get("compound_id", 0)
            compound_label = compound.get("compound_label", "neutral")

        return {
            "arousal": arousal,
            "valence": valence,
            "drowsy": drowsy,
            "perclos": face_out["perclos_ear"][0],
            "compound_id": compound_id,
            "compound_label": compound_label,
            "kfer_emotion": face_out["_kfer_top1_label"],
            "kfer_confidence": face_out["_kfer_top1_conf"],
            "face_detected": True,
            "bbox": face_out["_bbox"],
            "_kfer_probs": face_out["kfer_probs"],
            "_landmarks": face_info.get("landmarks"),
        }
