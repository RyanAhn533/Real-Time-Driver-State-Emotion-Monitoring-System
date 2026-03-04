"""
A/V Expert v2: Pretrained SOTA Encoder + Lightweight Head
=============================================================
Pipeline Layer 2.2.

Strategy: K-EMocon has only ~2,259 segments across 11 participants.
Training an encoder from scratch is futile. Instead:

  Audio: emotion2vec (ACL 2024) → 768-dim frozen features
         - Pretrained on 160k hours of speech emotion data
         - Feature extraction only (no fine-tuning)
         - FunASR interface or direct HuggingFace

  Bio:   Hand-crafted signal features + lightweight MLP
         - HRV features from BVP (RMSSD, SDNN, LF/HF ratio)
         - SCR features from EDA (peak count, mean amplitude, rise time)
         - TEMP trend (slope over window)
         - Total: ~15-20 features per segment
         - These features are well-established in psychophysiology

  Fusion: Concat [audio_feat, bio_feat] → small MLP → A/V prediction
          Only the MLP is trained. Encoders are frozen/rule-based.

This approach works with 2,259 samples because:
  - Audio encoder is frozen (0 learnable params from audio side)
  - Bio features are hand-crafted (0 learnable params from bio side)
  - Only the fusion MLP needs to be trained (~10k params)

References:
  - emotion2vec: https://github.com/ddlBoJack/emotion2vec
  - Bio features: Kreibig (2010), Physiological differentiation of emotions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path


# ═══════════════════════════════════════════════════════════
#  Audio Encoder: emotion2vec wrapper
# ═══════════════════════════════════════════════════════════

class Emotion2VecEncoder:
    """
    Frozen emotion2vec feature extractor.

    Uses FunASR interface for simplicity. Falls back to stub
    when FunASR/emotion2vec is not installed.

    Output: 768-dim utterance-level embedding per audio segment.
    """

    def __init__(self,
                 model_id: str = "iic/emotion2vec_plus_base",
                 device: str = "cuda",
                 granularity: str = "utterance"):
        """
        Args:
            model_id: HuggingFace/ModelScope model ID
                - "iic/emotion2vec_base": representation only (768-dim)
                - "iic/emotion2vec_plus_seed": fine-tuned on academic data
                - "iic/emotion2vec_plus_base": ~90M params, 9-class
                - "iic/emotion2vec_plus_large": ~300M params, 9-class
            device: torch device
            granularity: "utterance" (1 vector per segment) or
                         "frame" (T vectors at 50Hz)
        """
        self.device = device
        self.granularity = granularity
        self.model = None
        self.available = False
        self.feat_dim = 768

        try:
            from funasr import AutoModel
            self.model = AutoModel(model=model_id, hub="hf")
            self.available = True
            print(f"[Emotion2Vec] Loaded: {model_id}")
        except ImportError:
            print("[Emotion2Vec] FunASR not installed. "
                  "Install with: pip install -U funasr")
        except Exception as e:
            print(f"[Emotion2Vec] Failed to load: {e}")

    @torch.no_grad()
    def extract(self, wav_path: str) -> Dict:
        """
        Extract emotion features from audio file.

        Args:
            wav_path: path to 16kHz wav file

        Returns:
            dict with:
                feats: np.ndarray [768] or [T, 768]
                scores: np.ndarray [9] emotion class scores (if plus model)
                label: int, predicted emotion class
        """
        if not self.available:
            return self._stub_output()

        result = self.model.generate(
            wav_path,
            granularity=self.granularity,
            extract_embedding=True,
        )

        out = {}
        if result and len(result) > 0:
            r = result[0]
            if "feats" in r:
                feats = r["feats"]
                if isinstance(feats, np.ndarray):
                    out["feats"] = feats
                else:
                    out["feats"] = np.array(feats, dtype=np.float32)

            if "scores" in r:
                out["scores"] = np.array(r["scores"], dtype=np.float32)
                out["label"] = int(np.argmax(out["scores"]))
        else:
            out = self._stub_output()

        return out

    @torch.no_grad()
    def extract_from_tensor(self, waveform: torch.Tensor,
                            sr: int = 16000) -> np.ndarray:
        """
        Extract features from audio tensor (for real-time pipeline).

        Args:
            waveform: [samples] or [1, samples] at 16kHz
            sr: sample rate

        Returns:
            feats: np.ndarray [768]
        """
        if not self.available:
            return np.zeros(self.feat_dim, dtype=np.float32)

        # Save temp wav and extract (FunASR requires file path)
        import tempfile, soundfile as sf
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, waveform.cpu().numpy(), sr)
            result = self.extract(f.name)
        return result.get("feats", np.zeros(self.feat_dim, dtype=np.float32))

    def _stub_output(self) -> Dict:
        return {
            "feats": np.zeros(self.feat_dim, dtype=np.float32),
            "scores": np.zeros(9, dtype=np.float32),
            "label": 4,  # neutral
        }


# ═══════════════════════════════════════════════════════════
#  Bio Feature Extractor: Hand-crafted physiological features
# ═══════════════════════════════════════════════════════════

class BioFeatureExtractor:
    """
    Hand-crafted feature extraction from BVP, EDA, TEMP, HR signals.

    No learnable parameters. Based on established psychophysiology:
      - Kreibig (2010): autonomic differentiation of emotions
      - Healey & Picard (2005): detecting stress using physiological sensors

    Features extracted (per segment window):
      BVP (4): mean_hr, sdnn, rmssd, lf_hf_ratio
      EDA (5): mean_scl, std_scl, n_peaks, mean_peak_amp, auc
      TEMP (3): mean_temp, temp_slope, temp_range
      HR  (3): mean_hr_direct, hr_std, hr_range
      Total: 15 features
    """

    def __init__(self, bvp_sr: int = 64, eda_sr: int = 4,
                 temp_sr: int = 4, hr_sr: int = 1):
        self.bvp_sr = bvp_sr
        self.eda_sr = eda_sr
        self.temp_sr = temp_sr
        self.hr_sr = hr_sr
        self.feat_dim = 15

    def extract(self,
                bvp: Optional[np.ndarray] = None,
                eda: Optional[np.ndarray] = None,
                temp: Optional[np.ndarray] = None,
                hr: Optional[np.ndarray] = None,
                ) -> np.ndarray:
        """
        Extract features from raw bio signals.

        Args:
            bvp: [N] blood volume pulse at bvp_sr Hz
            eda: [N] electrodermal activity at eda_sr Hz
            temp: [N] skin temperature at temp_sr Hz
            hr: [N] heart rate at hr_sr Hz

        Returns:
            features: [15] hand-crafted feature vector
        """
        feats = []

        # ── BVP features (4) ──
        if bvp is not None and len(bvp) > 10:
            feats.extend(self._bvp_features(bvp))
        else:
            feats.extend([0.0] * 4)

        # ── EDA features (5) ──
        if eda is not None and len(eda) > 4:
            feats.extend(self._eda_features(eda))
        else:
            feats.extend([0.0] * 5)

        # ── TEMP features (3) ──
        if temp is not None and len(temp) > 2:
            feats.extend(self._temp_features(temp))
        else:
            feats.extend([0.0] * 3)

        # ── HR features (3) ──
        if hr is not None and len(hr) > 1:
            feats.extend(self._hr_features(hr))
        else:
            feats.extend([0.0] * 3)

        return np.array(feats, dtype=np.float32)

    def _bvp_features(self, bvp: np.ndarray) -> List[float]:
        """HRV-like features from BVP."""
        # Simple peak detection for IBI (inter-beat interval)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(bvp, distance=int(self.bvp_sr * 0.5))

        if len(peaks) < 2:
            return [0.0, 0.0, 0.0, 1.0]

        ibi = np.diff(peaks) / self.bvp_sr * 1000  # ms

        mean_hr = 60000.0 / np.mean(ibi) if np.mean(ibi) > 0 else 0.0
        sdnn = np.std(ibi)
        rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2)) if len(ibi) > 1 else 0.0

        # LF/HF ratio (simplified via time-domain proxy)
        lf_hf = sdnn / (rmssd + 1e-8)

        return [
            float(np.clip(mean_hr / 200.0, 0, 1)),    # normalized HR
            float(np.clip(sdnn / 200.0, 0, 1)),        # normalized SDNN
            float(np.clip(rmssd / 200.0, 0, 1)),       # normalized RMSSD
            float(np.clip(lf_hf / 5.0, 0, 1)),         # normalized LF/HF
        ]

    def _eda_features(self, eda: np.ndarray) -> List[float]:
        """Skin conductance features."""
        mean_scl = float(np.mean(eda))
        std_scl = float(np.std(eda))

        # Simple peak detection for SCR
        from scipy.signal import find_peaks
        if std_scl > 0:
            threshold = mean_scl + 0.5 * std_scl
            peaks, properties = find_peaks(eda, height=threshold,
                                           distance=int(self.eda_sr * 1.0))
            n_peaks = len(peaks)
            if n_peaks > 0 and "peak_heights" in properties:
                mean_amp = float(np.mean(properties["peak_heights"]) - mean_scl)
            else:
                mean_amp = 0.0
        else:
            n_peaks = 0
            mean_amp = 0.0

        auc = float(np.trapz(eda)) / max(len(eda), 1)

        return [
            float(np.clip(mean_scl / 30.0, 0, 1)),     # normalized SCL
            float(np.clip(std_scl / 10.0, 0, 1)),      # normalized std
            float(np.clip(n_peaks / 10.0, 0, 1)),      # normalized peak count
            float(np.clip(mean_amp / 5.0, 0, 1)),      # normalized amplitude
            float(np.clip(auc / 50.0, 0, 1)),           # normalized AUC
        ]

    def _temp_features(self, temp: np.ndarray) -> List[float]:
        """Skin temperature features."""
        mean_temp = float(np.mean(temp))
        temp_range = float(np.ptp(temp))

        # Linear slope
        x = np.arange(len(temp), dtype=np.float64)
        if len(temp) > 1 and np.std(temp) > 0:
            slope = float(np.polyfit(x, temp, 1)[0])
        else:
            slope = 0.0

        return [
            float(np.clip((mean_temp - 30.0) / 10.0, 0, 1)),  # normalized temp
            float(np.clip(slope + 0.5, 0, 1)),                 # normalized slope
            float(np.clip(temp_range / 5.0, 0, 1)),            # normalized range
        ]

    def _hr_features(self, hr: np.ndarray) -> List[float]:
        """Direct heart rate features."""
        return [
            float(np.clip(np.mean(hr) / 200.0, 0, 1)),
            float(np.clip(np.std(hr) / 50.0, 0, 1)),
            float(np.clip(np.ptp(hr) / 100.0, 0, 1)),
        ]

    def get_quality(self,
                    bvp: Optional[np.ndarray],
                    eda: Optional[np.ndarray],
                    temp: Optional[np.ndarray],
                    hr: Optional[np.ndarray]) -> float:
        """Modality completeness as quality score [0, 1]."""
        available = sum([
            bvp is not None and len(bvp) > 10,
            eda is not None and len(eda) > 4,
            temp is not None and len(temp) > 2,
            hr is not None and len(hr) > 1,
        ])
        return available / 4.0


# ═══════════════════════════════════════════════════════════
#  A/V Expert v2: Frozen encoders + lightweight fusion head
# ═══════════════════════════════════════════════════════════

class AVExpertV2(nn.Module):
    """
    Audio + Bio Expert using pretrained encoders.

    Architecture:
      Audio: emotion2vec (frozen, 768-dim) → linear proj (128-dim)
      Bio:   hand-crafted features (15-dim) → linear proj (64-dim)
      Fusion: concat [128 + 64] → MLP → emotion logits + A/V

    Only the fusion MLP is trained (~15k params).
    This is the correct strategy for K-EMocon's 2,259 samples.
    """

    def __init__(self,
                 num_classes: int = 7,
                 audio_feat_dim: int = 768,
                 bio_feat_dim: int = 15,
                 d_proj: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.audio_feat_dim = audio_feat_dim
        self.bio_feat_dim = bio_feat_dim

        # Audio projection (from frozen emotion2vec 768-dim)
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feat_dim, d_proj),
            nn.LayerNorm(d_proj),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Bio projection (from 15-dim hand-crafted features)
        bio_proj_dim = d_proj // 2
        self.bio_proj = nn.Sequential(
            nn.Linear(bio_feat_dim, bio_proj_dim),
            nn.LayerNorm(bio_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        fused_dim = d_proj + bio_proj_dim

        # Modality attention gate
        self.mod_gate = nn.Sequential(
            nn.Linear(fused_dim, 2),
            nn.Softmax(dim=-1),
        )

        # Emotion head
        self.emotion_head = nn.Sequential(
            nn.Linear(fused_dim, d_proj),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj, num_classes),
        )

        # A/V regression head
        self.av_head = nn.Sequential(
            nn.Linear(fused_dim, d_proj // 2),
            nn.GELU(),
            nn.Linear(d_proj // 2, 2),
            nn.Tanh(),
        )

    def forward(self,
                audio_feats: Optional[torch.Tensor] = None,
                bio_feats: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_feats: [B, 768] from emotion2vec (pre-extracted)
            bio_feats: [B, 15] from BioFeatureExtractor

        Returns dict:
            emotion_logits: [B, num_classes]
            arousal: [B]
            valence: [B]
            quality: [B]
            uncertainty: [B]
        """
        B = audio_feats.size(0) if audio_feats is not None else bio_feats.size(0)
        device = audio_feats.device if audio_feats is not None else bio_feats.device

        # Project modalities
        if audio_feats is not None:
            z_audio = self.audio_proj(audio_feats)
        else:
            z_audio = torch.zeros(B, 128, device=device)

        if bio_feats is not None:
            z_bio = self.bio_proj(bio_feats)
        else:
            z_bio = torch.zeros(B, 64, device=device)

        # Concat + gate
        z_cat = torch.cat([z_audio, z_bio], dim=-1)  # [B, 192]

        # Quality
        has_audio = 1.0 if audio_feats is not None else 0.0
        has_bio = 1.0 if bio_feats is not None else 0.0
        quality = torch.full((B,), (has_audio + has_bio) / 2.0, device=device)

        # Heads
        emotion_logits = self.emotion_head(z_cat)
        av = self.av_head(z_cat)

        # Uncertainty
        probs = F.softmax(emotion_logits, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
        uncertainty = entropy / np.log(self.num_classes)

        return {
            "emotion_logits": emotion_logits,
            "arousal": av[:, 0],
            "valence": av[:, 1],
            "quality": quality,
            "uncertainty": uncertainty,
        }


# ═══════════════════════════════════════════════════════════
#  Offline Feature Extraction for K-EMocon
# ═══════════════════════════════════════════════════════════

def extract_kemocon_features(base_dir: str,
                             index_csv: str,
                             out_dir: str,
                             emotion2vec_model: str = "iic/emotion2vec_plus_base"):
    """
    Pre-extract all audio and bio features from K-EMocon dataset.
    Run this ONCE, then train AVExpertV2 on the extracted features.

    Creates:
      out_dir/audio_feats.npy   [N, 768]
      out_dir/bio_feats.npy     [N, 15]
      out_dir/labels.npy        [N, 2] (arousal, valence)
      out_dir/metadata.npy      [N] (segment indices)
    """
    import pandas as pd
    from tqdm import tqdm

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(index_csv)
    for col in ["has_bio_img", "has_audio_mel", "has_video_face"]:
        if col in df.columns:
            df = df[df[col] == 1]
    df = df.dropna(subset=["label_ext_A_norm", "label_ext_V_norm"]).reset_index(drop=True)

    print(f"Processing {len(df)} segments from K-EMocon...")

    # Init extractors
    audio_enc = Emotion2VecEncoder(model_id=emotion2vec_model)
    bio_ext = BioFeatureExtractor()

    audio_feats_all = []
    bio_feats_all = []
    labels_all = []

    base = Path(base_dir)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        pid = int(row["pid"])
        seg = int(row.get("seg_idx", idx))

        # Audio
        audio_path = base / f"audio_wav" / f"pid_{pid:02d}" / f"seg_{seg:05d}.wav"
        if audio_path.exists():
            result = audio_enc.extract(str(audio_path))
            audio_feats_all.append(result["feats"])
        else:
            audio_feats_all.append(np.zeros(768, dtype=np.float32))

        # Bio signals
        bio_path = base / f"bio_raw" / f"pid_{pid:02d}" / f"seg_{seg:05d}.npz"
        if bio_path.exists():
            bio_data = np.load(str(bio_path))
            bio_feat = bio_ext.extract(
                bvp=bio_data.get("bvp", None),
                eda=bio_data.get("eda", None),
                temp=bio_data.get("temp", None),
                hr=bio_data.get("hr", None),
            )
            bio_feats_all.append(bio_feat)
        else:
            bio_feats_all.append(np.zeros(15, dtype=np.float32))

        # Labels
        labels_all.append([
            float(row["label_ext_A_norm"]),
            float(row["label_ext_V_norm"]),
        ])

    # Save
    np.save(str(out_path / "audio_feats.npy"),
            np.stack(audio_feats_all))
    np.save(str(out_path / "bio_feats.npy"),
            np.stack(bio_feats_all))
    np.save(str(out_path / "labels.npy"),
            np.array(labels_all, dtype=np.float32))

    print(f"Saved to {out_path}")
    print(f"  audio_feats: {np.stack(audio_feats_all).shape}")
    print(f"  bio_feats:   {np.stack(bio_feats_all).shape}")
    print(f"  labels:      {np.array(labels_all).shape}")
