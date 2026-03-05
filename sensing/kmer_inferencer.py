"""
K-MER Real-Time Inferencer
==========================
multimodal_dms의 KMERFusion 모델을 실시간 센싱 데이터로 추론.

입력: 카메라 프레임 (BGR) + 오디오 (48kHz mono) + 생체신호 (PPG/EDA/TEMP)
출력: arousal, valence, drowsy, compound_emotion (13-class)

Pipeline:
  frame_bgr  → MediaPipe FaceMesh → face crop → KFERExpert  → kfer_probs/meta
                                               → FACSAuxExpert → perclos/ear/facs
  audio_1d   → 16kHz resample     → emotion2vec → emo2vec_probs
                                   → audeering   → avd
  ppg/eda/temp → bio feature extraction         → bvp/eda/hr_temp features
  (all features) → KMERFusion → arousal, valence, drowsy
                → CompoundEmotionMapper → compound_emotion (13-class)

사용법:
    from kmer_inferencer import KMERInferencer

    infer = KMERInferencer(
        kfer_ckpt="emotion_system/result/best.pth",
        kmer_ckpt="multimodal_dms/results_kmer/best_model.pth",
        device="cuda",
    )
    result = infer.forward(frame_bgr, audio_1d, ppg, eda, temp)
    # result: {arousal, valence, drowsy, compound_id, compound_label,
    #          kfer_emotion, kfer_confidence, face_detected}
"""

import sys
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# ── Project paths ────────────────────────────────────────────────────────────
_SENSING_DIR    = Path(__file__).resolve().parent
_PROJECT_ROOT   = _SENSING_DIR.parent
_EMO_SYS_ROOT   = _PROJECT_ROOT / "emotion_system"
_MM_DMS_ROOT    = _PROJECT_ROOT / "multimodal_dms"

for _p in [str(_EMO_SYS_ROOT), str(_MM_DMS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── MediaPipe face landmark indices (au_extractor.py 기준) ──────────────────
_AU_REGIONS = [
    ("forehead",    (69, 299, 9)),
    ("eyes_left",   159),
    ("eyes_right",  386),
    ("nose",        195),
    ("cheek_left",  186),
    ("cheek_right", 410),
    ("mouth",       13),
    ("chin",        18),
]
_WORK_SHORT = 800   # au_extractor.py 와 동일
_IMG_SIZE   = 224


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int = 16000) -> np.ndarray:
    """PCM float32 mono → 16kHz (librosa). unavailable 시 원본 반환."""
    if src_sr == dst_sr or audio is None or len(audio) == 0:
        return audio
    try:
        import librosa
        return librosa.resample(audio.astype(np.float32), orig_sr=src_sr, target_sr=dst_sr)
    except ImportError:
        return audio


def _compute_audio_quality(audio: np.ndarray) -> np.ndarray:
    """
    T6 audio_quality (3,): rms, zero_crossing_rate, snr_est
    모두 [0, 1] 범위로 정규화.
    """
    if audio is None or len(audio) == 0:
        return np.zeros(3, dtype=np.float32)
    audio = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    rms_norm = min(rms / 0.1, 1.0)   # 0.1 RMS = full scale 가정

    # ZCR (voicing proxy)
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0))

    # Simple SNR estimate (peak / rms)
    peak = float(np.max(np.abs(audio)) + 1e-8)
    snr_est = min(rms / peak, 1.0)

    return np.array([rms_norm, zcr, snr_est], dtype=np.float32)


def _build_bio_npz(ppg: list, eda: list, temp: list) -> dict:
    """
    BioQueues.snapshot() 포맷 → bio_expert.extract_bio_features_v2() 입력 포맷.
    각 key: (N, 2) shape ([:, 0] 이 실제 신호 값).
    """
    npz = {}
    if ppg:
        arr = np.array([(x[1], x[2]) for x in ppg], dtype=np.float32)  # (N, 2)
        npz["bvp"] = arr
    if eda:
        arr = np.array([(x[1], 0.0) for x in eda], dtype=np.float32)
        npz["eda"] = arr
    if temp:
        arr = np.array([(x[1], 0.0) for x in temp], dtype=np.float32)
        npz["temp"] = arr
    return npz


def _compute_cross_modal(kfer_probs: np.ndarray, kfer_entropy: float,
                          emo2vec_probs: np.ndarray, audeering_avd: np.ndarray,
                          face_valid: bool, audio_valid: bool) -> np.ndarray:
    """
    T13 cross_modal (3,): face_audio_agree, av_consistency, entropy_gap.
    """
    # K-FER 감정의 valence 부호 (+: happy, surprised / -: angry, anxious, hurt, sad)
    KFER_VALENCE = np.array([-1, -1, +1, -1, 0, -1, +1], dtype=np.float32)  # 7-class
    # emotion2vec valence 부호 (angry/disgust/fear/sad=-1, happy=+1, neutral=0, ...)
    EMO2VEC_VALENCE = np.array([-1, -1, -1, +1, 0, 0, -1, 0, 0], dtype=np.float32)  # 9-class

    cross = np.zeros(3, dtype=np.float32)

    if face_valid and audio_valid:
        kfer_v = float(np.dot(kfer_probs, KFER_VALENCE))
        e2v_v  = float(np.dot(emo2vec_probs, EMO2VEC_VALENCE))
        # 코사인 유사도 [-1, 1] → [0, 1]
        cross[0] = (kfer_v * e2v_v + 1.0) / 2.0

        # AV consistency: audeering arousal 과 valence 의 일치도
        if audeering_avd is not None:
            a, v = float(audeering_avd[0]), float(audeering_avd[1])
            cross[1] = 1.0 - abs(a - v)

    # Entropy gap: K-FER entropy vs emotion2vec entropy (정규화)
    e2v_ent = float(-np.sum(np.clip(emo2vec_probs, 1e-10, 1) * np.log(np.clip(emo2vec_probs, 1e-10, 1))))
    max_ent = np.log(max(len(kfer_probs), len(emo2vec_probs)))
    cross[2] = float(abs(kfer_entropy - e2v_ent) / (max_ent + 1e-8))

    return cross


# ─────────────────────────────────────────────────────────────────────────────
# Face detection / crop (MediaPipe FaceMesh)
# ─────────────────────────────────────────────────────────────────────────────

class _FaceDetector:
    """MediaPipe FaceMesh 기반 얼굴 crop + AU 좌표 추출."""

    def __init__(self):
        self._mesh = None

    def _init(self):
        if self._mesh is not None:
            return True
        try:
            import mediapipe as mp
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return True
        except Exception as e:
            print(f"[FaceDetector] MediaPipe 초기화 실패: {e}")
            return False

    def detect(self, frame_bgr: np.ndarray):
        """
        BGR 프레임 → face_crop_chw, au_coords, bbox 반환.
        실패 시 (None, None, None).

        face_crop_chw: (3, IMG_SIZE, IMG_SIZE) float32 [0-255] RGB
        au_coords    : (8, 2) float32 IMG_SIZE 스케일 좌표
        bbox         : (x1, y1, x2, y2) 원본 해상도 기준
        """
        if not self._init():
            return None, None, None

        h, w = frame_bgr.shape[:2]
        scale = _WORK_SHORT / min(h, w)
        ww, wh = int(round(w * scale)), int(round(h * scale))
        work    = cv2.resize(frame_bgr, (ww, wh), interpolation=cv2.INTER_LINEAR)
        work_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)

        res = self._mesh.process(work_rgb)
        if not res.multi_face_landmarks:
            return None, None, None

        lms = res.multi_face_landmarks[0].landmark

        # Bounding box (work 해상도)
        xs = [lm.x * ww for lm in lms]
        ys = [lm.y * wh for lm in lms]
        x1_w, x2_w = int(min(xs)), int(max(xs))
        y1_w, y2_w = int(min(ys)), int(max(ys))

        pad_x = int((x2_w - x1_w) * 0.20)
        pad_y = int((y2_w - y1_w) * 0.20)
        x1_w = max(0, x1_w - pad_x)
        y1_w = max(0, y1_w - pad_y)
        x2_w = min(ww, x2_w + pad_x)
        y2_w = min(wh, y2_w + pad_y)

        fw, fh = x2_w - x1_w, y2_w - y1_w
        if fw < 10 or fh < 10:
            return None, None, None

        face_rgb = work_rgb[y1_w:y2_w, x1_w:x2_w]

        # AU 좌표 → IMG_SIZE 스케일
        def _center(idx):
            if isinstance(idx, tuple):
                cx = np.mean([lms[i].x * ww for i in idx])
                cy = np.mean([lms[i].y * wh for i in idx])
            else:
                cx, cy = lms[idx].x * ww, lms[idx].y * wh
            cx_in = float(np.clip((cx - x1_w) * (_IMG_SIZE / fw), 0, _IMG_SIZE - 1))
            cy_in = float(np.clip((cy - y1_w) * (_IMG_SIZE / fh), 0, _IMG_SIZE - 1))
            return cx_in, cy_in

        au_coords = np.array([_center(idx) for _, idx in _AU_REGIONS], dtype=np.float32)

        # face_crop → CHW float32 [0-255]
        face_resized = cv2.resize(face_rgb, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        face_chw = face_resized.transpose(2, 0, 1).astype(np.float32)  # (3, H, W)

        # 원본 bbox
        bbox = (
            int(x1_w / scale), int(y1_w / scale),
            int(x2_w / scale), int(y2_w / scale),
        )

        return face_chw, au_coords, bbox


# ─────────────────────────────────────────────────────────────────────────────
# KMERInferencer
# ─────────────────────────────────────────────────────────────────────────────

class KMERInferencer:
    """
    K-MER 실시간 추론기.

    필수:
        kfer_ckpt : emotion_system/result/best.pth (K-FER 모델)
        kmer_ckpt : multimodal_dms/results_kmer/best_model.pth (KMERFusion 모델)

    선택:
        enable_face_expert  : HSEmotion FaceExpert 활성화 (face_stats token)
        enable_audio_experts: emotion2vec + audeering 활성화 (audio tokens)
                              Jetson 에서 메모리 제한 시 False 권장
    """

    def __init__(
        self,
        kfer_ckpt: str,
        kmer_ckpt: str,
        device: str = "cuda",
        enable_face_expert: bool = False,
        enable_audio_experts: bool = True,
        audio_src_sr: int = 48000,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.audio_src_sr = audio_src_sr

        # ── 얼굴 검출기 ──────────────────────────────────────────────────
        self._face_detector = _FaceDetector()

        # ── K-FER Expert ─────────────────────────────────────────────────
        print(f"[KMERInferencer] K-FER 로드: {kfer_ckpt}")
        from experts.kfer_expert import KFERExpert
        self._kfer = KFERExpert(checkpoint=kfer_ckpt, device=str(self.device))

        # ── FACS Auxiliary Expert ─────────────────────────────────────────
        from experts.facs_aux import FACSAuxExpert
        self._facs = FACSAuxExpert()

        # ── HSEmotion Face Expert (선택) ──────────────────────────────────
        self._face_expert = None
        if enable_face_expert:
            try:
                from experts.face_expert import FaceExpert
                self._face_expert = FaceExpert(device=str(self.device))
                print("[KMERInferencer] HSEmotion FaceExpert 로드 완료")
            except Exception as e:
                print(f"[KMERInferencer] HSEmotion 로드 실패 (skip): {e}")

        # ── Audio Experts (선택) ──────────────────────────────────────────
        self._emotion2vec = None
        self._audeering   = None
        if enable_audio_experts:
            try:
                from experts.audio_expert import Emotion2VecExpert, AudeeringExpert
                print("[KMERInferencer] emotion2vec 로드 중...")
                self._emotion2vec = Emotion2VecExpert(device=str(self.device))
                print("[KMERInferencer] audeering 로드 중...")
                self._audeering   = AudeeringExpert(device=str(self.device))
            except Exception as e:
                print(f"[KMERInferencer] Audio expert 로드 실패 (skip): {e}")

        # ── KMERFusion 모델 ───────────────────────────────────────────────
        print(f"[KMERInferencer] KMERFusion 로드: {kmer_ckpt}")
        from fusion.kmer_fusion import KMERFusion, build_valid_mask
        self._build_valid_mask = build_valid_mask

        self._fusion = self._load_kmer(kmer_ckpt, KMERFusion)

        # ── Compound Emotion Mapper ───────────────────────────────────────
        from fusion.compound_emotion import CompoundEmotionMapper
        self._compound_mapper = CompoundEmotionMapper()

        print("[KMERInferencer] 초기화 완료")

    # ── KMERFusion 체크포인트 로드 ────────────────────────────────────────
    def _load_kmer(self, ckpt_path: str, ModelClass) -> torch.nn.Module:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        # 체크포인트 구조: {"model": state_dict, "config": {...}} 또는 state_dict 직접
        cfg   = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt

        model_cfg = cfg.get("model", cfg)
        model = ModelClass(
            d_model      = model_cfg.get("d_model",      64),
            n_heads      = model_cfg.get("n_heads",       4),
            dropout      = model_cfg.get("dropout",      0.1),
            use_temporal = model_cfg.get("use_temporal", False),
            use_valence  = model_cfg.get("use_valence",  True),
            use_drowsy   = model_cfg.get("use_drowsy",   True),
        ).to(self.device)
        model.load_state_dict(state, strict=True)
        model.eval()
        epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
        print(f"[KMERInferencer] KMERFusion 로드 완료 (epoch={epoch})")
        return model

    # ── 얼굴 처리 ────────────────────────────────────────────────────────
    def _process_face(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        frame_bgr → K-FER / FACS / HSEmotion 특징 추출.
        """
        empty = {
            "kfer_probs":   np.zeros(7, dtype=np.float32),
            "kfer_meta":    np.array([0.0, np.log(7)], dtype=np.float32),  # quality, entropy
            "face_stats":   np.zeros(3, dtype=np.float32),
            "perclos_ear":  np.array([0.5, 0.25], dtype=np.float32),
            "facs_scores":  np.zeros(6, dtype=np.float32),
            "face_valid":   False,
        }

        face_chw, au_coords, bbox = self._face_detector.detect(frame_bgr)
        if face_chw is None:
            return empty

        # KFERExpert expects (T, 3, H, W)
        faces_batch = face_chw[np.newaxis]  # (1, 3, 224, 224)

        kfer_out = self._kfer.extract(faces_batch, au_coords=au_coords)

        facs_out = self._facs.extract(faces_batch)

        # HSEmotion (선택)
        face_stats = np.zeros(3, dtype=np.float32)
        if self._face_expert is not None:
            hse = self._face_expert.extract(faces_batch)
            if hse.get("valid", False):
                p = hse["probs"]
                face_stats = np.array([p.max(), p.mean(), p.std()], dtype=np.float32)
        else:
            # K-FER probs 에서 대리 계산
            p = kfer_out["probs"]
            face_stats = np.array([p.max(), p.mean(), p.std()], dtype=np.float32)

        return {
            "kfer_probs":  kfer_out["probs"],
            "kfer_meta":   np.array([kfer_out["quality"], kfer_out["entropy"]], dtype=np.float32),
            "face_stats":  face_stats,
            "perclos_ear": np.array([facs_out["perclos"], facs_out["ear_mean"]], dtype=np.float32),
            "facs_scores": facs_out["facs_scores"],
            "face_valid":  kfer_out["valid"],
            # 디버그용
            "_kfer_top1_id":   kfer_out.get("top1_id",    4),
            "_kfer_top1_label":kfer_out.get("top1_label", "neutral"),
            "_kfer_top1_conf": kfer_out.get("top1_conf",  0.0),
            "_bbox":           bbox,
        }

    # ── 오디오 처리 ──────────────────────────────────────────────────────
    def _process_audio(self, audio_1d: np.ndarray) -> Dict[str, Any]:
        """audio_1d (48kHz) → emotion2vec + audeering 특징."""
        emo2vec_probs = np.zeros(9,  dtype=np.float32)
        audeering_avd = np.zeros(3,  dtype=np.float32)
        audio_quality = np.zeros(3,  dtype=np.float32)
        audio_valid   = False

        if audio_1d is not None and len(audio_1d) > 0:
            audio_16k = _resample_audio(audio_1d, self.audio_src_sr, 16000)
            audio_quality = _compute_audio_quality(audio_16k)
            audio_valid   = (audio_quality[0] > 0.01)  # 최소 RMS

            if audio_valid:
                if self._emotion2vec is not None:
                    e2v = self._emotion2vec.extract(audio_16k, sr=16000)
                    if e2v.get("valid", False):
                        emo2vec_probs = e2v["probs"]

                if self._audeering is not None:
                    aud = self._audeering.extract(audio_16k, sr=16000)
                    if aud.get("valid", False):
                        audeering_avd = aud["avd"]

        return {
            "emo2vec_probs": emo2vec_probs,
            "audeering_avd": audeering_avd,
            "audio_quality": audio_quality,
            "audio_valid":   audio_valid,
        }

    # ── Bio 처리 ─────────────────────────────────────────────────────────
    def _process_bio(self, ppg: list, eda: list, temp: list) -> Dict[str, Any]:
        """BioQueues snapshot → bio feature tokens."""
        from experts.bio_expert import extract_bio_features_v2

        npz_data = _build_bio_npz(ppg, eda, temp)
        bio_result = extract_bio_features_v2(npz_data)
        feats = bio_result["features"]  # (15,)

        # 슬라이싱: bio_expert.BIO_FEATURE_NAMES 기준
        bvp_features    = feats[0:4]        # mean_hr, sdnn, rmssd, lf_hf
        eda_features    = feats[4:9]        # mean_scl, std_scl, n_peaks, amp, auc
        temp_feats      = feats[9:12]       # temp_mean, temp_slope, temp_range
        hr_feats        = feats[12:15]      # hr_mean, hr_std, hr_range
        hr_temp_features = np.concatenate([hr_feats, temp_feats])  # (6,)

        bio_quality = np.array([
            float(bio_result["bio_quality"]),
            float(bio_result["bvp_valid"]),
            float(bio_result["eda_valid"]),
        ], dtype=np.float32)

        return {
            "bvp_features":     bvp_features,
            "eda_features":     eda_features,
            "hr_temp_features": hr_temp_features,
            "bio_quality":      bio_quality,
            "bio_valid":        bio_result["valid"],
        }

    # ── 특징 딕셔너리 빌드 ───────────────────────────────────────────────
    def _build_feature_dict(
        self,
        face_out: dict,
        audio_out: dict,
        bio_out: dict,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Expert 출력 → KMERFusion.forward() 입력 형식.
        반환: (features_dict, valid_mask) — batch size 1.
        """

        # T13 cross-modal
        cross = _compute_cross_modal(
            kfer_probs    = face_out["kfer_probs"],
            kfer_entropy  = float(face_out["kfer_meta"][1]),
            emo2vec_probs = audio_out["emo2vec_probs"],
            audeering_avd = audio_out["audeering_avd"],
            face_valid    = face_out["face_valid"],
            audio_valid   = audio_out["audio_valid"],
        )

        # T14 validity flags
        validity_flags = np.array([
            float(face_out["face_valid"]),
            float(audio_out["audio_valid"]),
            float(bio_out["bio_valid"]),
        ], dtype=np.float32)

        # numpy → torch, batch dim 추가
        def _t(arr):
            return torch.from_numpy(arr).unsqueeze(0).to(self.device)

        features = {
            "kfer_probs":       _t(face_out["kfer_probs"]),
            "kfer_meta":        _t(face_out["kfer_meta"]),
            "face_stats":       _t(face_out["face_stats"]),
            "emo2vec_probs":    _t(audio_out["emo2vec_probs"]),
            "audeering_avd":    _t(audio_out["audeering_avd"]),
            "audio_quality":    _t(audio_out["audio_quality"]),
            "bvp_features":     _t(bio_out["bvp_features"]),
            "eda_features":     _t(bio_out["eda_features"]),
            "hr_temp_features": _t(bio_out["hr_temp_features"]),
            "bio_quality":      _t(bio_out["bio_quality"]),
            "perclos_ear":      _t(face_out["perclos_ear"]),
            "facs_scores":      _t(face_out["facs_scores"]),
            "cross_modal":      _t(cross),
            "validity_flags":   _t(validity_flags),
        }

        # valid_mask (1, 15)
        face_v  = torch.tensor([[face_out["face_valid"]]],  dtype=torch.bool,  device=self.device)
        audio_v = torch.tensor([[audio_out["audio_valid"]]], dtype=torch.bool, device=self.device)
        bio_v   = torch.tensor([[bio_out["bio_valid"]]],    dtype=torch.bool,  device=self.device)
        valid_mask = self._build_valid_mask(
            face_v.squeeze(1), audio_v.squeeze(1), bio_v.squeeze(1)
        )

        return features, valid_mask

    # ── 메인 추론 ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward(
        self,
        frame_bgr:  Optional[np.ndarray],
        audio_1d:   np.ndarray,
        ppg:        list,
        eda:        list,
        temp:       list,
    ) -> Dict[str, Any]:
        """
        실시간 멀티모달 추론.

        Args:
            frame_bgr : [H, W, 3] BGR numpy (카메라 최신 프레임)
            audio_1d  : [N] float32 mono PCM 48kHz (링 버퍼 스냅샷)
            ppg       : [(ts, d1, d2), ...] BioQueues.ppg 스냅샷
            eda       : [(ts, real), ...] BioQueues.eda 스냅샷
            temp      : [(ts, skin_c), ...] BioQueues.temp 스냅샷

        Returns:
            dict:
                arousal         : float [0, 1]          — 각성도
                valence         : float [0, 1] or None  — 감정가
                drowsy          : int {0=alert, 1=drowsy, 2=sleeping}
                compound_id     : int (0~12)
                compound_label  : str (e.g. "excited", "stressed", "drowsy")
                kfer_emotion    : str (7-class top-1)
                kfer_confidence : float
                face_detected   : bool
        """
        result: Dict[str, Any] = {
            "arousal":         0.5,
            "valence":         None,
            "drowsy":          0,
            "compound_id":     0,
            "compound_label":  "neutral",
            "kfer_emotion":    None,
            "kfer_confidence": 0.0,
            "face_detected":   False,
        }

        # ── Expert 추론 ────────────────────────────────────────────────
        face_out  = self._process_face(frame_bgr) if frame_bgr is not None else {
            "kfer_probs":   np.zeros(7,  dtype=np.float32),
            "kfer_meta":    np.array([0.0, np.log(7)], dtype=np.float32),
            "face_stats":   np.zeros(3,  dtype=np.float32),
            "perclos_ear":  np.array([0.5, 0.25], dtype=np.float32),
            "facs_scores":  np.zeros(6,  dtype=np.float32),
            "face_valid":   False,
        }

        audio_out = self._process_audio(audio_1d)
        bio_out   = self._process_bio(ppg, eda, temp)

        # ── KMERFusion ─────────────────────────────────────────────────
        features, valid_mask = self._build_feature_dict(face_out, audio_out, bio_out)
        fusion_out = self._fusion(features, valid_mask=valid_mask)

        arousal = float(fusion_out["arousal"][0, 0].cpu())
        result["arousal"] = arousal

        if "valence" in fusion_out:
            result["valence"] = float(fusion_out["valence"][0, 0].cpu())

        # Drowsy
        if "drowsy" in fusion_out:
            drowsy_logits = fusion_out["drowsy"][0].cpu()
            result["drowsy"] = int(torch.argmax(drowsy_logits).item())

        # ── K-FER 결과 ─────────────────────────────────────────────────
        if face_out.get("face_valid", False):
            result["face_detected"]   = True
            result["kfer_emotion"]    = face_out.get("_kfer_top1_label", "neutral")
            result["kfer_confidence"] = face_out.get("_kfer_top1_conf",  0.0)

        # ── Compound emotion (13-class) ────────────────────────────────
        kfer_top1_id = face_out.get("_kfer_top1_id", 4)   # 4 = neutral
        is_drowsy    = (result["drowsy"] >= 1)
        cid, clabel  = self._compound_mapper.map_single(
            kfer_top1_id   = kfer_top1_id,
            arousal_pred   = arousal,
            is_drowsy      = is_drowsy,
        )
        result["compound_id"]    = cid
        result["compound_label"] = clabel

        return result
