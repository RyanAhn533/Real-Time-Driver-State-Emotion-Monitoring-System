"""
Pipeline Orchestrator: Full Multimodal DMS Pipeline
======================================================
Connects all 5 layers of the architecture:

  Layer 1: Input & Preprocessing (FaceMesh, AU extraction, EAR)
  Layer 2: Expert Layer (FER Expert + A/V Expert)
  Layer 3: PerClos + Drowsy Head
  Layer 4: Event Encoder (temporal summary)
  Layer 5: Agent / Gating (final decision)

Output spec per event window:
  - final_emotion: 12-class emotion/state
  - p_final: full probability distribution
  - arousal, valence: continuous A/V
  - p_drowsy, drowsiness_level: none / mild / drowsy
  - diagnostics: per-expert confidence, disagreement, PerClos

Usage:
  pipeline = DrivingMonitorPipeline.from_config("configs/pipeline.yaml")
  result = pipeline.process_window(frames, audio_segment, bio_segment)
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque

from models.drowsiness.perclos import compute_ear, compute_perclos
from integration.emotion_refiner import refine_emotion, REFINED_LABELS


@dataclass
class FrameResult:
    """Per-frame outputs from Layer 1-3."""
    # FER Expert
    fer_logits: np.ndarray = None      # [C]
    fer_probs: np.ndarray = None       # [C]
    fer_quality: float = 0.0
    fer_uncertainty: float = 1.0
    fer_pred: int = 0
    fer_label: str = "neutral"
    # A/V Expert
    av_logits: np.ndarray = None       # [C]
    av_arousal: float = 0.0
    av_valence: float = 0.0
    av_quality: float = 0.0
    av_uncertainty: float = 1.0
    # PerClos
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_avg: float = 0.0
    eyes_closed: bool = False
    # Metadata
    timestamp: float = 0.0
    frame_idx: int = 0


@dataclass
class WindowResult:
    """Per-window outputs from Layer 4-5 (final pipeline output)."""
    # Final emotion
    final_emotion: str = "neutral"
    final_emotion_id: int = 0
    emotion_probs: np.ndarray = None    # [num_classes]
    # A/V
    arousal: float = 0.0
    valence: float = 0.0
    # Drowsiness
    perclos: float = 0.0
    q_perclos: float = 1.0
    p_drowsy: float = 0.0
    drowsiness_level: str = "none"      # none / mild / drowsy
    # Gating
    gate_weight: float = 0.5           # FER trust weight
    # Diagnostics
    fer_emotion: str = "neutral"
    av_emotion: str = "neutral"
    fer_confidence: float = 0.0
    av_confidence: float = 0.0
    expert_disagreement: float = 0.0
    # Timing
    latency_ms: float = 0.0
    window_idx: int = 0


class DrivingMonitorPipeline:
    """
    Full 5-layer pipeline orchestrator.

    Manages temporal buffers and coordinates all modules.
    """

    def __init__(self,
                 fer_expert=None,
                 av_expert=None,
                 event_encoder=None,
                 agent=None,
                 window_size: int = 30,
                 fps: int = 30,
                 ear_threshold: float = 0.21,
                 perclos_drowsy: float = 0.4,
                 perclos_sleeping: float = 0.8,
                 device: str = "cuda",
                 fer_id2label: Dict[int, str] = None,
                 num_refined_classes: int = 13):
        """
        Args:
            fer_expert: FERExpert instance (Layer 2.1)
            av_expert: AVExpert instance (Layer 2.2, optional)
            event_encoder: EventEncoder instance (Layer 4, optional)
            agent: AgentGating instance (Layer 5, optional)
            window_size: frames per event window
            fps: camera frame rate
            ear_threshold: EAR threshold for eyes closed
            device: torch device
            fer_id2label: FER class id to label mapping
            num_refined_classes: final output class count (13 with drowsy)
        """
        self.fer_expert = fer_expert
        self.av_expert = av_expert
        self.event_encoder = event_encoder
        self.agent = agent

        self.window_size = window_size
        self.fps = fps
        self.ear_threshold = ear_threshold
        self.perclos_drowsy = perclos_drowsy
        self.perclos_sleeping = perclos_sleeping
        self.device = torch.device(device)
        self.fer_id2label = fer_id2label or {}
        self.num_refined_classes = num_refined_classes

        # Temporal buffers
        self.frame_buffer: deque = deque(maxlen=window_size * 3)
        self.ear_buffer: deque = deque(maxlen=window_size * 3)
        self.prev_emotion_logits: Optional[torch.Tensor] = None

        # Counters
        self.frame_count = 0
        self.window_count = 0

    def process_frame(self,
                      image: np.ndarray,
                      au_coords: np.ndarray,
                      ear_left: float = None,
                      ear_right: float = None,
                      face_detected: bool = True,
                      audio_segment: Optional[np.ndarray] = None,
                      bio_segment: Optional[np.ndarray] = None,
                      timestamp: float = None,
                      ) -> FrameResult:
        """
        Process a single frame through Layer 1-3.

        Args:
            image: [H, W, 3] BGR face crop
            au_coords: [K, 2] AU center pixel coords
            ear_left, ear_right: pre-computed EAR (or None to skip)
            face_detected: whether face detection succeeded
            audio_segment: [1, n_mels, T] mel (for A/V Expert)
            bio_segment: [n_ch, seq_len] bio signals (for A/V Expert)
            timestamp: frame timestamp
        """
        result = FrameResult(
            timestamp=timestamp or time.time(),
            frame_idx=self.frame_count,
        )

        # ── Layer 2.1: FER Expert ──
        if self.fer_expert is not None and face_detected:
            img_tensor = self._preprocess_image(image)
            au_tensor = torch.tensor(au_coords, dtype=torch.float32).unsqueeze(0)

            fer_out = self.fer_expert.predict(
                img_tensor, au_tensor,
                face_detected=torch.tensor([float(face_detected)]),
            )
            result.fer_logits = fer_out["emotion_logits"][0].cpu().numpy()
            result.fer_probs = fer_out["emotion_probs"][0].cpu().numpy()
            result.fer_quality = fer_out["quality"][0].item()
            result.fer_uncertainty = fer_out["uncertainty"][0].item()
            result.fer_pred = result.fer_probs.argmax()
            result.fer_label = self.fer_id2label.get(result.fer_pred, "unknown")

        # ── Layer 2.2: A/V Expert ──
        if self.av_expert is not None:
            audio_t = None
            bio_t = None
            if audio_segment is not None:
                audio_t = torch.tensor(audio_segment, dtype=torch.float32).unsqueeze(0).to(self.device)
            if bio_segment is not None:
                bio_t = torch.tensor(bio_segment, dtype=torch.float32).unsqueeze(0).to(self.device)

            if audio_t is not None or bio_t is not None:
                av_out = self.av_expert(audio=audio_t, bio=bio_t)
                result.av_logits = av_out["emotion_logits"][0].cpu().detach().numpy()
                result.av_arousal = av_out["arousal"][0].item()
                result.av_valence = av_out["valence"][0].item()
                result.av_quality = av_out["quality"][0].item()
                result.av_uncertainty = av_out["uncertainty"][0].item()

        # ── Layer 3.1: PerClos ──
        if ear_left is not None and ear_right is not None:
            result.ear_left = ear_left
            result.ear_right = ear_right
            result.ear_avg = (ear_left + ear_right) / 2.0
            result.eyes_closed = result.ear_avg < self.ear_threshold
            self.ear_buffer.append(result.ear_avg)

        # Buffer the frame result
        self.frame_buffer.append(result)
        self.frame_count += 1

        return result

    def process_window(self) -> Optional[WindowResult]:
        """
        Process accumulated frames through Layer 3.2, 4, 5.
        Call this after every `window_size` frames.

        Returns WindowResult or None if insufficient frames.
        """
        if len(self.frame_buffer) < self.window_size // 2:
            return None

        t0 = time.time()
        window_frames = list(self.frame_buffer)[-self.window_size:]
        result = WindowResult(window_idx=self.window_count)

        # ── Layer 3: PerClos computation ──
        ear_window = list(self.ear_buffer)[-self.window_size:]
        if len(ear_window) > 0:
            result.perclos = compute_perclos(ear_window, self.ear_threshold)
            # q_perclos: fraction of frames with valid EAR
            valid_ears = sum(1 for e in ear_window if e > 0)
            result.q_perclos = valid_ears / len(ear_window) if ear_window else 0.0

        # ── Layer 3.2: Drowsy Head (rule-based) ──
        avg_arousal = np.mean([f.av_arousal for f in window_frames
                               if f.av_arousal != 0.0]) if any(
            f.av_arousal != 0.0 for f in window_frames) else None

        result.p_drowsy, result.drowsiness_level = self._compute_drowsiness(
            result.perclos, avg_arousal
        )

        # ── Prepare tensors for Layer 4-5 ──
        has_agent = self.agent is not None
        has_event_enc = self.event_encoder is not None

        # Get latest frame's expert outputs
        last = window_frames[-1]

        if has_agent or has_event_enc:
            # Build z(t) sequence for event encoder
            z_seq = self._build_z_sequence(window_frames)

            # ── Layer 4: Event Encoder ──
            h_event = None
            if has_event_enc and z_seq is not None:
                z_tensor = torch.tensor(z_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                h_event = self.event_encoder(z_tensor)

            # ── Layer 5: Agent / Gating ──
            if has_agent:
                agent_out = self._run_agent(last, h_event, result)
                result.final_emotion_id = agent_out["emotion_logits"][0].argmax().item()
                result.final_emotion = REFINED_LABELS.get(
                    result.final_emotion_id, "unknown")
                result.emotion_probs = F.softmax(
                    agent_out["emotion_logits"][0], dim=-1).cpu().detach().numpy()
                av = agent_out["arousal_valence"][0].cpu().detach().numpy()
                result.arousal = float(av[0])
                result.valence = float(av[1])

                if "gate_weight" in agent_out:
                    result.gate_weight = agent_out["gate_weight"][0].item()

                # Store for temporal smoothing
                self.prev_emotion_logits = agent_out["emotion_logits"].detach()

        # Fallback: no Agent, use FER + refiner
        if not has_agent:
            if last.fer_probs is not None:
                base_emotion = last.fer_label
                arousal = last.av_arousal if last.av_arousal != 0.0 else None
                valence = last.av_valence if last.av_valence != 0.0 else None
                refined_id, refined_name = refine_emotion(base_emotion, arousal, valence)
                result.final_emotion_id = refined_id
                result.final_emotion = refined_name
                result.emotion_probs = last.fer_probs
                result.arousal = last.av_arousal
                result.valence = last.av_valence

        # Override to drowsy if drowsiness is high
        if result.p_drowsy > 0.7:
            result.final_emotion = "drowsy"
            result.final_emotion_id = 12

        # Diagnostics
        result.fer_emotion = last.fer_label
        result.fer_confidence = 1.0 - last.fer_uncertainty
        if last.av_logits is not None:
            av_pred = last.av_logits.argmax()
            result.av_emotion = self.fer_id2label.get(av_pred, "unknown")
            result.av_confidence = 1.0 - last.av_uncertainty
        result.expert_disagreement = self._compute_disagreement(last)

        result.latency_ms = (time.time() - t0) * 1000
        self.window_count += 1

        return result

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """BGR uint8 [H,W,3] -> normalized [1,3,224,224] tensor."""
        import cv2
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(self.device)

    def _compute_drowsiness(self, perclos: float,
                            arousal: Optional[float]) -> tuple:
        """Rule-based drowsiness from PerClos + arousal."""
        th_drowsy = self.perclos_drowsy
        th_sleeping = self.perclos_sleeping

        if arousal is not None and arousal < 0.3:
            th_drowsy *= 0.7
            th_sleeping *= 0.8

        if perclos >= th_sleeping:
            return 1.0, "drowsy"
        elif perclos >= th_drowsy:
            p = (perclos - th_drowsy) / (th_sleeping - th_drowsy)
            return p, "mild"
        return 0.0, "none"

    def _build_z_sequence(self, frames: List[FrameResult]) -> Optional[np.ndarray]:
        """Build z(t) = concat(z_FER, z_AV, perclos, ...) for each frame."""
        z_list = []
        for f in frames:
            parts = []
            # FER features
            if f.fer_probs is not None:
                parts.append(f.fer_probs)
                parts.append([f.fer_quality, f.fer_uncertainty])
            else:
                parts.append(np.zeros(7))
                parts.append([0.0, 1.0])
            # A/V features
            if f.av_logits is not None:
                parts.append(F.softmax(
                    torch.tensor(f.av_logits), dim=-1).numpy())
                parts.append([f.av_arousal, f.av_valence,
                              f.av_quality, f.av_uncertainty])
            else:
                parts.append(np.zeros(7))
                parts.append([0.0, 0.0, 0.0, 1.0])
            # PerClos
            parts.append([f.ear_avg, float(f.eyes_closed)])

            z_list.append(np.concatenate([np.atleast_1d(p) for p in parts]))

        if len(z_list) == 0:
            return None
        return np.stack(z_list)  # [T, z_dim]

    def _run_agent(self, last_frame: FrameResult,
                   h_event: Optional[torch.Tensor],
                   window_result: WindowResult) -> Dict[str, torch.Tensor]:
        """Run Agent/Gating on current window data."""
        B = 1

        fer_logits = torch.tensor(
            last_frame.fer_logits if last_frame.fer_logits is not None
            else np.zeros(7), dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        fer_quality = torch.tensor(
            [last_frame.fer_quality], dtype=torch.float32
        ).to(self.device)

        fer_uncertainty = torch.tensor(
            [last_frame.fer_uncertainty], dtype=torch.float32
        ).to(self.device)

        perclos = torch.tensor(
            [window_result.perclos], dtype=torch.float32
        ).to(self.device)

        q_perclos = torch.tensor(
            [window_result.q_perclos], dtype=torch.float32
        ).to(self.device)

        p_drowsy = torch.tensor(
            [window_result.p_drowsy], dtype=torch.float32
        ).to(self.device)

        av_logits = None
        av_arousal = None
        av_valence = None
        av_quality = None
        av_uncertainty = None

        if last_frame.av_logits is not None:
            av_logits = torch.tensor(
                last_frame.av_logits, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            av_arousal = torch.tensor(
                [last_frame.av_arousal], dtype=torch.float32
            ).to(self.device)
            av_valence = torch.tensor(
                [last_frame.av_valence], dtype=torch.float32
            ).to(self.device)
            av_quality = torch.tensor(
                [last_frame.av_quality], dtype=torch.float32
            ).to(self.device)
            av_uncertainty = torch.tensor(
                [last_frame.av_uncertainty], dtype=torch.float32
            ).to(self.device)

        return self.agent(
            fer_logits=fer_logits,
            fer_quality=fer_quality,
            fer_uncertainty=fer_uncertainty,
            perclos=perclos,
            q_perclos=q_perclos,
            p_drowsy=p_drowsy,
            h_event=h_event,
            av_logits=av_logits,
            av_arousal=av_arousal,
            av_valence=av_valence,
            av_quality=av_quality,
            av_uncertainty=av_uncertainty,
        )

    def _compute_disagreement(self, frame: FrameResult) -> float:
        """KL divergence between FER and A/V Expert predictions."""
        if frame.fer_probs is None or frame.av_logits is None:
            return 0.0
        p = frame.fer_probs + 1e-8
        q = F.softmax(torch.tensor(frame.av_logits), dim=-1).numpy() + 1e-8
        # Symmetric KL
        kl_pq = np.sum(p * np.log(p / q))
        kl_qp = np.sum(q * np.log(q / p))
        return float((kl_pq + kl_qp) / 2.0)

    def get_output_json(self, window_result: WindowResult) -> Dict[str, Any]:
        """Convert WindowResult to JSON-serializable dict for ECU/HMI."""
        return {
            "emotion": window_result.final_emotion,
            "emotion_id": window_result.final_emotion_id,
            "emotion_probs": window_result.emotion_probs.tolist()
                if window_result.emotion_probs is not None else None,
            "arousal": round(window_result.arousal, 4),
            "valence": round(window_result.valence, 4),
            "drowsiness": {
                "level": window_result.drowsiness_level,
                "perclos": round(window_result.perclos, 4),
                "p_drowsy": round(window_result.p_drowsy, 4),
            },
            "diagnostics": {
                "fer_emotion": window_result.fer_emotion,
                "fer_confidence": round(window_result.fer_confidence, 4),
                "av_confidence": round(window_result.av_confidence, 4),
                "gate_weight": round(window_result.gate_weight, 4),
                "expert_disagreement": round(window_result.expert_disagreement, 4),
            },
            "meta": {
                "window_idx": window_result.window_idx,
                "latency_ms": round(window_result.latency_ms, 2),
            },
        }
