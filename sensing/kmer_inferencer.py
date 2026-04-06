"""
K-MER Real-Time Inferencer
==========================
multimodal_dms의 KMERFusion 모델을 실시간 센싱 데이터로 추론.

입력: 카메라 프레임 (BGR) + 오디오 (48kHz mono) + 생체신호 (PPG/EDA/TEMP)
출력: arousal, valence, drowsy, compound_emotion (13-class)

Usage:
    from kmer_inferencer import KMERInferencer

    infer = KMERInferencer(
        kfer_ckpt="emotion_system/result/best.pth",
        kmer_ckpt="multimodal_dms/results_kmer/best_model.pth",
        device="cuda",
    )
    result = infer.forward(frame_bgr, audio_1d, ppg, eda, temp)
"""

import sys
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

_SENSING_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SENSING_DIR.parent
_EMO_SYS_ROOT = _PROJECT_ROOT / "emotion_system"
_MM_DMS_ROOT = _PROJECT_ROOT / "multimodal_dms"

for _p in [str(_EMO_SYS_ROOT), str(_MM_DMS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_AU_REGIONS = [
    ("forehead", (69, 299, 9)),
    ("eyes_left", 159),
    ("eyes_right", 386),
    ("nose", 195),
    ("cheek_left", 186),
    ("cheek_right", 410),
    ("mouth", 13),
    ("chin", 18),
]
_WORK_SHORT = 800
_IMG_SIZE = 224


def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int = 16000) -> np.ndarray:
    """PCM float32 mono -> 16kHz (librosa). unavailable 시 원본 반환."""
    if len(audio) == 0:
        return audio
    try:
        import librosa
        return librosa.resample(audio.astype(np.float32), orig_sr=src_sr, target_sr=dst_sr)
    except ImportError:
        return audio


def _compute_audio_quality(audio: np.ndarray) -> np.ndarray:
    """T6 audio_quality (3,): rms, zero_crossing_rate, snr_est. [0,1] 범위."""
    if len(audio) == 0:
        return np.zeros(3, dtype=np.float32)
    audio = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    rms = min(rms / 0.1, 1.0)
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)
    snr = float(np.max(np.abs(audio)) / (rms + 1e-8))
    snr = min(snr / 0.1, 1.0)
    return np.array([rms, zcr, snr], dtype=np.float32)


def _build_bio_npz(ppg: list, eda: list, temp: list) -> dict:
    """BioQueues.snapshot() -> bio_expert 입력 포맷. 각 key: (N, 2)."""
    result = {}
    if ppg:
        arr = np.array(ppg, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] >= 2:
            result["bvp"] = arr[:, 1:3]
        else:
            result["bvp"] = np.column_stack([arr[:, 0], np.zeros(len(arr), dtype=np.float32)])
    else:
        result["bvp"] = np.array([[0.0, 0.0]], dtype=np.float32)
    if eda:
        arr = np.array(eda, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] >= 2:
            result["eda"] = arr[:, -2:]
        else:
            result["eda"] = np.column_stack([arr[:, 0], np.zeros(len(arr), dtype=np.float32)])
    else:
        result["eda"] = np.array([[0.0, 0.0]], dtype=np.float32)
    if temp:
        arr = np.array(temp, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] >= 2:
            result["temp"] = arr[:, -2:]
        else:
            result["temp"] = np.column_stack([arr[:, 0], np.zeros(len(arr), dtype=np.float32)])
    else:
        result["temp"] = np.array([[0.0, 0.0]], dtype=np.float32)
    return result


def _compute_cross_modal(kfer_probs, kfer_entropy, emo2vec_probs, audeering_avd,
                         face_valid: bool, audio_valid: bool) -> np.ndarray:
    """T13 cross_modal (3,): face_audio_agree, av_consistency, entropy_gap."""
    kfer_probs = np.array(kfer_probs, dtype=np.float32)
    emo2vec_probs = np.array(emo2vec_probs, dtype=np.float32)
    audeering_avd = np.array(audeering_avd, dtype=np.float32)

    if not face_valid or not audio_valid:
        return np.zeros(3, dtype=np.float32)

    kfer_valence = np.array([-1, -1, 1, -1, 0, -1, 1], dtype=np.float32)
    emo2vec_valence = np.array([-1, -1, -1, 1, 0, 0, -1, 0, 0], dtype=np.float32)
    face_v = float(np.dot(kfer_probs, kfer_valence))
    audio_v = float(np.dot(emo2vec_probs[:len(emo2vec_valence)], emo2vec_valence))
    agree = np.clip(face_v * audio_v, -1.0, 1.0)

    av_consist = 1.0 - abs(face_v - audeering_avd[1]) / 2.0 if len(audeering_avd) > 1 else 0.0

    emo2vec_ent = float(-np.sum(np.clip(emo2vec_probs, 1e-10, 1.0) * np.log(np.clip(emo2vec_probs, 1e-10, 1.0) + 1e-8)))
    entropy_gap = float(np.clip(kfer_entropy - emo2vec_ent, -2, 2))

    return np.array([agree, av_consist, entropy_gap], dtype=np.float32)


class _FaceDetector:
    """MediaPipe FaceMesh 기반 얼굴 crop + AU 좌표 추출."""

    def __init__(self):
        self._mesh = None

    def _init(self):
        if self._mesh is not None:
            return
        try:
            import mediapipe
            self._mesh = mediapipe.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            print("[FaceDetector] MediaPipe init failed: " + str(e))

    def detect(self, frame_bgr: np.ndarray):
        """BGR -> (face_crop_chw, au_coords, bbox) or (None, None, None)."""
        self._init()
        if self._mesh is None:
            return None, None, None

        h, w = frame_bgr.shape[:2]
        scale = _WORK_SHORT / min(h, w)
        ww, wh = int(round(w * scale)), int(round(h * scale))
        work = cv2.resize(frame_bgr, (ww, wh), interpolation=cv2.INTER_LINEAR)
        work_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)

        res = self._mesh.process(work_rgb)
        if not res.multi_face_landmarks:
            return None, None, None

        lms = res.multi_face_landmarks[0].landmark
        xs = [lm.x * ww for lm in lms]
        ys = [lm.y * wh for lm in lms]
        x1_w, x2_w = int(min(xs)), int(max(xs))
        y1_w, y2_w = int(min(ys)), int(max(ys))

        pad_x = int((x2_w - x1_w) * 0.2)
        pad_y = int((y2_w - y1_w) * 0.2)
        x1_w = max(0, x1_w - pad_x)
        y1_w = max(0, y1_w - pad_y)
        x2_w = min(ww, x2_w + pad_x)
        y2_w = min(wh, y2_w + pad_y)

        face_w = x2_w - x1_w
        face_h = y2_w - y1_w
        if face_w < 10 or face_h < 10:
            return None, None, None

        face_crop = work[y1_w:y2_w, x1_w:x2_w]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (_IMG_SIZE, _IMG_SIZE))

        coords = []
        for _, idx in _AU_REGIONS:
            if isinstance(idx, tuple):
                cx = float(np.mean([lms[i].x for i in idx]))
                cy = float(np.mean([lms[i].y for i in idx]))
            else:
                cx = float(lms[idx].x)
                cy = float(lms[idx].y)
            cx_crop = np.clip((cx * ww - x1_w) * (_IMG_SIZE / face_w), 0, _IMG_SIZE - 1)
            cy_crop = np.clip((cy * wh - y1_w) * (_IMG_SIZE / face_h), 0, _IMG_SIZE - 1)
            coords.append([cx_crop, cy_crop])

        au_coords = np.array(coords, dtype=np.float32)
        face_chw = face_resized.transpose(2, 0, 1).astype(np.float32)
        bbox = (int(x1_w / scale), int(y1_w / scale), int(x2_w / scale), int(y2_w / scale))

        return face_chw, au_coords, bbox


class KMERInferencer:
    """K-MER 실시간 추론기."""

    def __init__(
        self,
        kfer_ckpt: str = "../emotion_system/result/best.pth",
        kmer_ckpt: str = "../multimodal_dms/results_kmer/best_model.pth",
        device: str = "cuda",
        enable_face_expert: bool = False,
        enable_audio_experts: bool = True,
        audio_src_sr: int = 48000,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.audio_src_sr = audio_src_sr

        self._face_detector = _FaceDetector()

        print("[KMERInferencer] K-FER loading: " + str(kfer_ckpt))
        from experts.kfer_expert import KFERExpert
        self._kfer = KFERExpert(checkpoint=str(kfer_ckpt), device=device)

        from experts.facs_aux import FACSAuxExpert
        self._facs = FACSAuxExpert()

        self._face_expert = None
        if enable_face_expert:
            try:
                from experts.face_expert import FaceExpert
                self._face_expert = FaceExpert(device=device)
                print("[KMERInferencer] HSEmotion FaceExpert loaded")
            except Exception as e:
                print("[KMERInferencer] HSEmotion load failed (skip): " + str(e))

        self._emotion2vec = None
        self._audeering = None
        if enable_audio_experts:
            try:
                from experts.audio_expert import Emotion2VecExpert, AudeeringExpert
                print("[KMERInferencer] Loading emotion2vec...")
                self._emotion2vec = Emotion2VecExpert(device=device)
                print("[KMERInferencer] Loading audeering...")
                self._audeering = AudeeringExpert(device=device)
            except Exception as e:
                print("[KMERInferencer] Audio expert load failed (skip): " + str(e))

        print("[KMERInferencer] Loading KMERFusion: " + str(kmer_ckpt))
        from fusion.kmer_fusion import KMERFusion, build_valid_mask
        self._build_valid_mask = build_valid_mask
        self._load_kmer(kmer_ckpt, KMERFusion)

        from fusion.compound_emotion import CompoundEmotionMapper
        self._compound_mapper = CompoundEmotionMapper()

        print("[KMERInferencer] Init complete")

    def _load_kmer(self, ckpt_path: str, ModelClass):
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict):
            config = ckpt.get("config", {})
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        else:
            config = {}
            state = ckpt

        self._fusion = ModelClass(
            d_model=config.get("d_model", 64),
            n_heads=config.get("n_heads", 4),
            dropout=config.get("dropout", 0.1),
            use_temporal=config.get("use_temporal", False),
            use_valence=config.get("use_valence", True),
            use_drowsy=config.get("use_drowsy", True),
        ).to(self.device)

        self._fusion.load_state_dict(state, strict=False)
        self._fusion.eval()
        epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
        print("[KMERInferencer] KMERFusion loaded (epoch=" + str(epoch) + ")")

    def _process_face(self, frame_bgr: np.ndarray) -> dict:
        empty = {
            "kfer_probs": np.zeros(7, dtype=np.float32),
            "kfer_meta": np.array([0.0, np.log(7)], dtype=np.float32),
            "face_stats": np.array([0.5, 0.25, 0.0], dtype=np.float32),
            "perclos_ear": np.zeros(6, dtype=np.float32),
            "facs_scores": np.zeros(6, dtype=np.float32),
            "face_valid": False,
        }

        face_chw, au_coords, bbox = self._face_detector.detect(frame_bgr)
        if face_chw is None:
            return empty

        kfer_out = self._kfer.extract(face_chw[np.newaxis], au_coords=au_coords)

        # FACS needs face crop in CHW format for EAR/PERCLOS
        # Use the already-detected face_chw (224x224 RGB CHW)
        # But also pass original frame at higher res for better landmark detection
        h, w = frame_bgr.shape[:2]
        frame_rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Use larger size to preserve facial details for FACS
        facs_size = max(min(h, w), _IMG_SIZE)
        frame_rgb_resized = cv2.resize(frame_rgb_full, (facs_size, facs_size))
        frame_chw_facs = frame_rgb_resized.transpose(2, 0, 1).astype(np.float32)
        facs_out = self._facs.extract(frame_chw_facs[np.newaxis])

        face_stats = np.array([0.5, 0.25, 0.0], dtype=np.float32)
        if self._face_expert is not None:
            hse_out = self._face_expert.extract(face_chw[np.newaxis])
            if hse_out.get("valid", False):
                probs = hse_out["probs"]
                face_stats = np.array([float(np.max(probs)), float(np.mean(probs)), float(np.std(probs))], dtype=np.float32)

        return {
            "kfer_probs": kfer_out.get("probs", empty["kfer_probs"]),
            "kfer_meta": np.array([kfer_out.get("quality", 0.0), kfer_out.get("entropy", np.log(7))], dtype=np.float32),
            "face_stats": face_stats,
            "perclos_ear": np.array([facs_out.get("perclos", 0.0), facs_out.get("ear_mean", 0.0)] + list(facs_out.get("facs_scores", np.zeros(4)))[:4], dtype=np.float32),
            "facs_scores": facs_out.get("facs_scores", np.zeros(6, dtype=np.float32))[:6],
            "face_valid": True,
            "_kfer_top1_id": kfer_out.get("top1_id", 4),
            "_kfer_top1_label": kfer_out.get("top1_label", "neutral"),
            "_kfer_top1_conf": kfer_out.get("top1_conf", 0.0),
            "_bbox": bbox,
        }

    def _process_audio(self, audio_1d: np.ndarray) -> dict:
        empty = {
            "emo2vec_probs": np.zeros(9, dtype=np.float32),
            "audeering_avd": np.zeros(3, dtype=np.float32),
            "audio_quality": np.zeros(3, dtype=np.float32),
            "audio_valid": False,
        }
        if len(audio_1d) == 0:
            return empty

        audio_16k = _resample_audio(audio_1d, self.audio_src_sr, 16000)
        quality = _compute_audio_quality(audio_16k)

        if len(audio_16k) < 0.01 * 16000:
            return empty

        emo2vec_probs = np.zeros(9, dtype=np.float32)
        audeering_avd = np.zeros(3, dtype=np.float32)
        audio_valid = False

        if self._emotion2vec is not None:
            e2v = self._emotion2vec.extract(audio_16k, sr=16000)
            if e2v.get("valid", False):
                emo2vec_probs = e2v["probs"]
                audio_valid = True

        if self._audeering is not None:
            aud = self._audeering.extract(audio_16k, sr=16000)
            if aud.get("valid", False):
                audeering_avd = aud["avd"]
                audio_valid = True

        return {
            "emo2vec_probs": emo2vec_probs,
            "audeering_avd": audeering_avd,
            "audio_quality": quality,
            "audio_valid": audio_valid,
        }

    def _process_bio(self, ppg: list, eda: list, temp: list) -> dict:
        from experts.bio_expert import extract_bio_features_v2
        npz = _build_bio_npz(ppg, eda, temp)
        bio = extract_bio_features_v2(npz)
        features = bio["features"]
        return {
            "bvp_features": features[0:4],
            "eda_features": features[4:9],
            "hr_temp_features": features[9:15],
            "bio_quality": np.array([float(bio["bio_quality"]), float(bio["bvp_valid"]), float(bio["eda_valid"])], dtype=np.float32),
            "bio_valid": bio.get("valid", False),
        }

    def _build_feature_dict(self, face_out: dict, audio_out: dict, bio_out: dict):
        cross_modal = _compute_cross_modal(
            face_out["kfer_probs"], float(face_out["kfer_meta"][1]),
            audio_out["emo2vec_probs"], audio_out["audeering_avd"],
            face_out["face_valid"], audio_out["audio_valid"],
        )
        validity_flags = np.array([float(face_out["face_valid"]), float(audio_out["audio_valid"]), float(bio_out["bio_valid"])], dtype=np.float32)

        def _t(arr):
            return torch.from_numpy(arr).unsqueeze(0).to(self.device)

        features = {}
        for key in ("kfer_probs", "kfer_meta", "face_stats"):
            features[key] = _t(face_out[key])
        for key in ("emo2vec_probs", "audeering_avd", "audio_quality"):
            features[key] = _t(audio_out[key])
        for key in ("bvp_features", "eda_features", "hr_temp_features", "bio_quality"):
            features[key] = _t(bio_out[key])
        features["perclos_ear"] = _t(face_out["perclos_ear"][:2])  # T11: (2,)
        features["facs_scores"] = _t(face_out["facs_scores"])
        features["cross_modal"] = _t(cross_modal)
        features["validity_flags"] = _t(validity_flags)

        valid_mask = self._build_valid_mask(
            torch.tensor([face_out["face_valid"]], dtype=torch.bool, device=self.device),
            torch.tensor([audio_out["audio_valid"]], dtype=torch.bool, device=self.device),
            torch.tensor([bio_out["bio_valid"]], dtype=torch.bool, device=self.device),
        )

        return features, valid_mask

    @torch.no_grad()
    def forward(
        self,
        frame_bgr: np.ndarray,
        audio_1d: Optional[np.ndarray] = None,
        ppg: Optional[list] = None,
        eda: Optional[list] = None,
        temp: Optional[list] = None,
    ) -> Dict[str, Any]:
        fallback = {
            "arousal": 0.5, "valence": None, "drowsy": 0,
            "compound_id": 0, "compound_label": "neutral",
            "kfer_emotion": "neutral", "kfer_confidence": 0.0,
            "face_detected": False,
        }

        if audio_1d is None:
            audio_1d = np.zeros(0, dtype=np.float32)
        if ppg is None:
            ppg = []
        if eda is None:
            eda = []
        if temp is None:
            temp = []

        face_out = self._process_face(frame_bgr)
        if not face_out["face_valid"]:
            return fallback

        audio_out = self._process_audio(audio_1d)
        bio_out = self._process_bio(ppg, eda, temp)

        features, valid_mask = self._build_feature_dict(face_out, audio_out, bio_out)
        fusion_out = self._fusion(features, valid_mask=valid_mask)

        arousal = float(fusion_out["arousal"][0, 0].cpu())
        valence = float(fusion_out["valence"][0, 0].cpu()) if "valence" in fusion_out else None
        drowsy = int(torch.argmax(fusion_out["drowsy"][0]).item()) if "drowsy" in fusion_out else 0

        compound = self._compound_mapper.map_single(
            kfer_top1_id=face_out.get("_kfer_top1_id", 4),
            arousal_pred=arousal > 0.5,
            is_drowsy=drowsy >= 1,
        )

        # compound can be tuple (id, label) or dict
        if isinstance(compound, tuple):
            compound_id, compound_label = compound
        else:
            compound_id = compound.get("compound_id", 0)
            compound_label = compound.get("compound_label", "neutral")

        return {
            "arousal": arousal,
            "valence": valence,
            "drowsy": drowsy,
            "compound_id": compound_id,
            "compound_label": compound_label,
            "kfer_emotion": face_out.get("_kfer_top1_label", "neutral"),
            "kfer_confidence": face_out.get("_kfer_top1_conf", 0.0),
            "face_detected": True,
        }
