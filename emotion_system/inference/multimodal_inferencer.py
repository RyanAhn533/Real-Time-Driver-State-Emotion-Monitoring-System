"""
Multimodal Real-Time Inferencer
================================
Video (FER) + Audio + Bio 전체 멀티모달 추론 클래스.

FER Expert : AUFERModel (face image + AU coords → emotion 7-class)
AV  Expert : ckpt_v3 multimodal student model (audio mel + bio → arousal/valence)
             ※ ckpt_v3 모델 클래스를 외부에서 주입하거나 _load_av_model()을 override.

사용법:
    from inference.multimodal_inferencer import MultimodalInferencer

    infer = MultimodalInferencer(
        fer_ckpt="emotion_system/result/best.pth",
        av_ckpt="emotion_system/multimodal/checkpoints/ckpt_v3/fold_1/best.pth",
    )
    result = infer.forward(frame_bgr, audio_1d, ppg, eda, temp)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

# emotion_system 루트를 sys.path에 추가 (어디서 실행해도 동작)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from inference.fer_inferencer import FERInferencer
from models.av_expert import BioFeatureExtractor


# ─────────────────────────────────────────────
# 오디오 → Mel 스펙트로그램 변환 유틸
# ─────────────────────────────────────────────
def _pcm_to_mel(
    audio: np.ndarray,
    src_sr: int = 48000,
    target_sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
    mel_time_bins: int = 128,
) -> np.ndarray:
    """
    PCM float32 mono → [n_mels, mel_time_bins] mel spectrogram.
    librosa가 없으면 None 반환.
    """
    try:
        import librosa
    except ImportError:
        return None

    if audio is None or audio.size == 0:
        return np.zeros((n_mels, mel_time_bins), dtype=np.float32)

    # 리샘플링
    if src_sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=src_sr, target_sr=target_sr)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    # 시간 축 → mel_time_bins로 맞추기 (padding / truncate)
    t = mel_db.shape[1]
    if t < mel_time_bins:
        mel_db = np.pad(mel_db, ((0, 0), (0, mel_time_bins - t)), mode="constant")
    else:
        mel_db = mel_db[:, :mel_time_bins]

    return mel_db.astype(np.float32)


# ─────────────────────────────────────────────
# Bio 큐 → numpy 변환 유틸
# ─────────────────────────────────────────────
def _extract_bio_arrays(
    ppg: List[Tuple],  # [(ts, d1, d2), ...]
    eda: List[Tuple],  # [(ts, real), ...]
    temp: List[Tuple], # [(ts, skin_c), ...]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """BioQueues.snapshot() 출력 → (ppg_arr, eda_arr, temp_arr)."""
    ppg_arr  = np.array([x[1] for x in ppg],  dtype=np.float32) if ppg  else None
    eda_arr  = np.array([x[1] for x in eda],  dtype=np.float32) if eda  else None
    temp_arr = np.array([x[1] for x in temp], dtype=np.float32) if temp else None
    return ppg_arr, eda_arr, temp_arr


# ─────────────────────────────────────────────
# 멀티모달 추론기 메인 클래스
# ─────────────────────────────────────────────
class MultimodalInferencer:
    """
    전체 멀티모달 추론기.

    Modality 1 (Video)  : FERInferencer  → emotion 7-class + AU coords
    Modality 2 (Audio)  : mel + ckpt_v3  → arousal / valence
    Modality 3 (Bio)    : BioFeatureExtractor → hand-crafted 15-dim + ckpt_v3

    ckpt_v3 모델 클래스가 없는 경우 AV 예측은 None으로 반환됩니다.
    모델 클래스가 준비되면 av_model_cls 인자로 주입하세요.

        from mymodels import AudioBioStudentModel
        infer = MultimodalInferencer(
            fer_ckpt="...",
            av_ckpt="emotion_system/multimodal/checkpoints/ckpt_v3/fold_1/best.pth",
            av_model_cls=AudioBioStudentModel,
        )
    """

    def __init__(
        self,
        fer_ckpt: str,
        av_ckpt: Optional[str] = None,
        av_model_cls=None,          # ckpt_v3 모델 클래스 (외부 주입)
        device: str = "cuda",
        audio_src_sr: int = 48000,
        mel_time_bins: int = 128,
        n_mels: int = 80,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.audio_src_sr  = audio_src_sr
        self.mel_time_bins = mel_time_bins
        self.n_mels        = n_mels

        # ── Modality 1: FER ──
        print(f"[MultimodalInferencer] Loading FER model: {fer_ckpt}")
        self.fer = FERInferencer(fer_ckpt, device=str(self.device))

        # ── Modality 3: Bio feature extractor (hand-crafted, no weights) ──
        self.bio_extractor = BioFeatureExtractor()

        # ── Modality 2+3: AV model (ckpt_v3 student) ──
        self.av_model = None
        if av_ckpt and av_model_cls is not None:
            self._load_av_model(av_ckpt, av_model_cls)
        elif av_ckpt:
            print(
                "[MultimodalInferencer] av_ckpt 경로가 지정되었지만 av_model_cls가 없습니다.\n"
                "  ckpt_v3 모델 클래스를 av_model_cls 인자로 전달해야 AV 추론이 활성화됩니다.\n"
                "  예) MultimodalInferencer(..., av_model_cls=AudioBioStudentModel)"
            )

    # ─── AV 모델 로드 ───────────────────────────────────────────────────────
    def _load_av_model(self, av_ckpt: str, av_model_cls):
        """
        ckpt_v3 student model 로드.

        av_model_cls는 다음 인터페이스를 가진 클래스여야 합니다.
            class AudioBioStudentModel(nn.Module):
                def __init__(self, config: dict): ...
                def forward(self,
                            mel: torch.Tensor,   # [B, n_mels, mel_time_bins]
                            bio: torch.Tensor,   # [B, bio_dim]
                            ) -> dict:
                    # 반환: {"arousal": Tensor[B], "valence": Tensor[B], ...}
        """
        ckpt_dir = Path(av_ckpt).parent
        config_path = ckpt_dir.parent / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        print(f"[MultimodalInferencer] Loading AV model: {av_ckpt}")
        model = av_model_cls(config).to(self.device)

        ckpt = torch.load(av_ckpt, map_location=self.device, weights_only=False)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt.get("model_state", ckpt)))
        model.load_state_dict(state, strict=True)
        model.eval()
        self.av_model = model
        print(f"[MultimodalInferencer] AV model loaded (epoch={ckpt.get('epoch', '?')})")

    # ─── 멀티모달 추론 ───────────────────────────────────────────────────────
    @torch.no_grad()
    def forward(
        self,
        frame_bgr: Optional[np.ndarray],
        audio_1d: np.ndarray,
        ppg: list,
        eda: list,
        temp: list,
    ) -> Dict[str, Any]:
        """
        모든 modality 데이터를 받아 추론 결과를 반환.

        Args:
            frame_bgr : [H, W, 3] BGR numpy array (camera frame)
            audio_1d  : [N] float32 mono PCM (48kHz ring buffer snapshot)
            ppg       : [(ts, d1, d2), ...] BioQueues.ppg snapshot
            eda       : [(ts, real), ...] BioQueues.eda snapshot
            temp      : [(ts, skin_c), ...] BioQueues.temp snapshot

        Returns:
            dict with:
                emotion        : str  (7-class or None)
                confidence     : float (or None)
                probs          : dict {label: prob} (or None)
                arousal        : float (or None if AV model not loaded)
                valence        : float (or None if AV model not loaded)
                bio_feats      : np.ndarray [15] hand-crafted bio features
                au_coords      : np.ndarray [8,2] (or None if no face)
                face_detected  : bool
        """
        result: Dict[str, Any] = {
            "emotion":      None,
            "confidence":   None,
            "probs":        None,
            "arousal":      None,
            "valence":      None,
            "bio_feats":    None,
            "au_coords":    None,
            "face_detected": False,
        }

        # ── Modality 1: FER (Video) ──────────────────────────────────────
        if frame_bgr is not None:
            fer_out = self.fer.predict(frame_bgr)
            if fer_out is not None:
                result["emotion"]       = fer_out["emotion"]
                result["confidence"]    = fer_out["confidence"]
                result["probs"]         = fer_out["probs"]
                result["au_coords"]     = fer_out["au_coords"]
                result["face_detected"] = True

        # ── Modality 3: Bio feature extraction ──────────────────────────
        ppg_arr, eda_arr, temp_arr = _extract_bio_arrays(ppg, eda, temp)
        bio_feats = self.bio_extractor.extract(
            bvp=ppg_arr,
            eda=eda_arr,
            temp=temp_arr,
        )
        result["bio_feats"] = bio_feats

        # ── Modality 2+3: AV model (ckpt_v3) ────────────────────────────
        if self.av_model is not None:
            mel = _pcm_to_mel(
                audio_1d,
                src_sr=self.audio_src_sr,
                n_mels=self.n_mels,
                mel_time_bins=self.mel_time_bins,
            )
            if mel is not None:
                mel_t = torch.from_numpy(mel).unsqueeze(0).to(self.device)   # [1, n_mels, T]
                bio_t = torch.from_numpy(bio_feats).unsqueeze(0).to(self.device)  # [1, 15]
                av_out = self.av_model(mel_t, bio_t)
                result["arousal"] = float(av_out["arousal"][0].cpu())
                result["valence"] = float(av_out["valence"][0].cpu())

        return result
