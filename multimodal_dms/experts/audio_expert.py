"""
Audio Expert — emotion2vec + audeering wav2vec2

emotion2vec_plus_large: 1024-dim embedding, 9-class emotion scores
audeering:              arousal/valence/dominance 직접 출력
"""
import numpy as np
import torch
from pathlib import Path

# emotion2vec+ labels (9 classes)
EMOTION2VEC_LABELS = [
    "angry", "disgusted", "fearful", "happy", "neutral",
    "other", "sad", "surprised", "unknown"
]
EMBED_DIM = 1024  # emotion2vec_plus_large uses 1024d
NUM_CLASSES = 9


class Emotion2VecExpert:
    """emotion2vec via FunASR — 1024-dim frozen embeddings + 9-class scores"""

    def __init__(self, model_name="iic/emotion2vec_plus_large", device="cuda"):
        self.device = device
        try:
            from funasr import AutoModel
            import os
            # Try local cache first (avoids slow modelscope re-download)
            local_dir = os.path.expanduser(
                "~/.cache/modelscope/hub/models/iic/emotion2vec_plus_large"
            )
            model_pt = os.path.join(local_dir, "model.pt")
            if os.path.exists(model_pt) and os.path.getsize(model_pt) > 1e9:
                self.model = AutoModel(model=local_dir, device=device,
                                       disable_update=True)
                print(f"[emotion2vec] Loaded from local cache: {local_dir}")
            else:
                self.model = AutoModel(model=model_name, device=device)
                print(f"[emotion2vec] Loaded from remote: {model_name}")
            self.available = True
        except Exception as e:
            print(f"[emotion2vec] Failed to load: {e}")
            self.available = False

    def extract(self, waveform: np.ndarray, sr: int = 16000) -> dict:
        """
        waveform: (N,) float32, 16kHz
        Returns: {embed: (1024,), probs: (9,), valid: bool}
        """
        if not self.available or waveform is None or len(waveform) < 1600:
            return {"embed": np.zeros(EMBED_DIM, dtype=np.float32),
                    "probs": np.zeros(NUM_CLASSES, dtype=np.float32),
                    "valid": False}

        try:
            res = self.model.generate(
                input=waveform,
                output_dir=None,
                granularity="utterance",
            )
            # Extract embedding and scores
            if isinstance(res, list) and len(res) > 0:
                r = res[0]
                embed = np.array(r.get("feats", np.zeros(EMBED_DIM)), dtype=np.float32)
                scores = np.array(r.get("scores", np.zeros(NUM_CLASSES)), dtype=np.float32)

                # Ensure correct shape
                if embed.ndim > 1:
                    embed = embed.mean(axis=0)
                if embed.shape[0] != EMBED_DIM:
                    # Pad or truncate
                    if embed.shape[0] < EMBED_DIM:
                        embed = np.pad(embed, (0, EMBED_DIM - embed.shape[0]))
                    else:
                        embed = embed[:EMBED_DIM]

                return {"embed": embed, "probs": scores, "valid": True}
        except Exception as e:
            pass

        return {"embed": np.zeros(EMBED_DIM, dtype=np.float32),
                "probs": np.zeros(NUM_CLASSES, dtype=np.float32),
                "valid": False}


class _AudeeringRegressionHead(torch.nn.Module):
    """Custom regression head matching audeering checkpoint."""
    def __init__(self, hidden_size=1024, num_labels=3):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.out_proj = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class AudeeringExpert:
    """audeering wav2vec2 — direct arousal/valence/dominance prediction"""

    MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

    def __init__(self, device="cuda"):
        self.device = device
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            import huggingface_hub

            self.processor = Wav2Vec2Processor.from_pretrained(self.MODEL_NAME)
            self.backbone = Wav2Vec2Model.from_pretrained(self.MODEL_NAME).to(device).eval()
            self.head = _AudeeringRegressionHead(1024, 3).to(device)

            # Load classifier weights from full checkpoint
            ckpt_path = huggingface_hub.hf_hub_download(self.MODEL_NAME, "pytorch_model.bin")
            state = torch.load(ckpt_path, weights_only=False, map_location=device)
            head_state = {
                k.replace("classifier.", ""): v
                for k, v in state.items() if k.startswith("classifier.")
            }
            self.head.load_state_dict(head_state)
            self.head.eval()

            self.available = True
            print(f"[audeering] Loaded with custom head: {self.MODEL_NAME}")
        except Exception as e:
            print(f"[audeering] Failed to load: {e}")
            self.available = False

    @torch.no_grad()
    def extract(self, waveform: np.ndarray, sr: int = 16000) -> dict:
        """
        waveform: (N,) float32, 16kHz
        Returns: {avd: (3,)} — arousal, valence, dominance
        """
        if not self.available or waveform is None or len(waveform) < 1600:
            return {"avd": np.zeros(3, dtype=np.float32), "valid": False}

        try:
            inputs = self.processor(
                waveform, sampling_rate=sr, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(self.device)
            outputs = self.backbone(input_values)
            hidden = outputs.last_hidden_state.mean(dim=1)  # (1, 1024)
            logits = self.head(hidden).cpu().numpy()[0]      # (3,)
            # audeering order: arousal, dominance, valence → reorder to A, V, D
            avd = np.array([logits[0], logits[2], logits[1]], dtype=np.float32)
            return {"avd": avd, "valid": True}
        except Exception as e:
            return {"avd": np.zeros(3, dtype=np.float32), "valid": False}
