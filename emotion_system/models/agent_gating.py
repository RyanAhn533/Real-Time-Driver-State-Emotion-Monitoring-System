"""
Agent / Gating Layer: Quality-Aware Expert Fusion
====================================================
Pipeline Layer 5.

Takes outputs from FER Expert, A/V Expert, PerClos/Drowsy module,
and the Event Encoder, then produces final emotion/state predictions
via learned soft gating.

Architecture:
  state s = [h_event, z_FER(T), z_AV(T), perclos_stats, p_drowsy]
      |
  GatingMLP -> w in [0,1] (FER vs A/V weight)
      |
  p_final = softmax(w * log_p_FER + (1-w) * log_p_AV)
      |
  EmotionHead -> final_emotion (C classes)
  AVHead -> arousal, valence
  DrowsyHead -> drowsiness_level

Loss:
  L = L_cls + L_AV + L_smooth + L_consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class QualityAwareGating(nn.Module):
    """
    Soft gating MLP that decides how much to trust FER vs A/V Expert.

    Input: state vector (concatenation of all available signals)
    Output: w in [0, 1] where w=1 means fully trust FER, w=0 means fully trust A/V
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim]
        Returns:
            w: [B, 1] gating weight for FER expert
        """
        return self.gate_mlp(state)


class AgentGating(nn.Module):
    """
    Full Agent module: state assembly + gating + final predictions.

    Combines:
      - h_event from Event Encoder
      - Latest FER Expert outputs (logits, quality, uncertainty)
      - Latest A/V Expert outputs (logits, quality, uncertainty)
      - PerClos statistics
      - Drowsiness probability
    """

    def __init__(self,
                 num_classes: int = 12,
                 d_event: int = 128,
                 d_fer_logits: int = 7,
                 d_av_logits: int = 7,
                 d_fer_extra: int = 2,
                 d_av_extra: int = 4,
                 d_perclos: int = 2,
                 d_drowsy: int = 1,
                 hidden_dim: int = 128,
                 use_av_expert: bool = True,
                 use_event_encoder: bool = True):
        """
        Args:
            num_classes: final output emotion classes
            d_event: event encoder output dim
            d_fer_logits: FER Expert class count
            d_av_logits: A/V Expert class count (for emotion)
            d_fer_extra: FER quality + uncertainty (q_face, u_FER)
            d_av_extra: A/V quality + uncertainty (arousal, valence, q_bioaudio, u_AV)
            d_perclos: perclos + q_perclos
            d_drowsy: p_drowsy
            hidden_dim: internal hidden dim
            use_av_expert: whether A/V Expert is available
            use_event_encoder: whether Event Encoder is available
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_av_expert = use_av_expert
        self.use_event_encoder = use_event_encoder

        # State dimension calculation
        state_dim = d_fer_logits + d_fer_extra  # FER outputs always present
        state_dim += d_perclos + d_drowsy  # PerClos always present

        if use_event_encoder:
            state_dim += d_event
        if use_av_expert:
            state_dim += d_av_logits + d_av_extra

        self.state_dim = state_dim

        # Gating: only needed when A/V Expert is available
        if use_av_expert:
            self.gating = QualityAwareGating(state_dim, hidden_dim)

        # State encoder: project state to hidden
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Emotion classification head
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

        # Arousal / Valence regression head
        self.av_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh(),  # A/V in [-1, 1]
        )

        # Drowsiness head (3 levels: none/mild/drowsy)
        self.drowsy_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

    def assemble_state(self,
                       fer_logits: torch.Tensor,
                       fer_quality: torch.Tensor,
                       fer_uncertainty: torch.Tensor,
                       perclos: torch.Tensor,
                       q_perclos: torch.Tensor,
                       p_drowsy: torch.Tensor,
                       h_event: Optional[torch.Tensor] = None,
                       av_logits: Optional[torch.Tensor] = None,
                       av_arousal: Optional[torch.Tensor] = None,
                       av_valence: Optional[torch.Tensor] = None,
                       av_quality: Optional[torch.Tensor] = None,
                       av_uncertainty: Optional[torch.Tensor] = None,
                       ) -> torch.Tensor:
        """
        Assemble all signals into a single state vector.
        All inputs: [B, *] tensors.
        """
        parts = [
            F.softmax(fer_logits, dim=-1),  # [B, C_fer]
            fer_quality.unsqueeze(-1) if fer_quality.dim() == 1 else fer_quality,
            fer_uncertainty.unsqueeze(-1) if fer_uncertainty.dim() == 1 else fer_uncertainty,
            perclos.unsqueeze(-1) if perclos.dim() == 1 else perclos,
            q_perclos.unsqueeze(-1) if q_perclos.dim() == 1 else q_perclos,
            p_drowsy.unsqueeze(-1) if p_drowsy.dim() == 1 else p_drowsy,
        ]

        if self.use_event_encoder and h_event is not None:
            parts.append(h_event)

        if self.use_av_expert and av_logits is not None:
            parts.append(F.softmax(av_logits, dim=-1))
            if av_arousal is not None:
                parts.append(av_arousal.unsqueeze(-1) if av_arousal.dim() == 1 else av_arousal)
            if av_valence is not None:
                parts.append(av_valence.unsqueeze(-1) if av_valence.dim() == 1 else av_valence)
            if av_quality is not None:
                parts.append(av_quality.unsqueeze(-1) if av_quality.dim() == 1 else av_quality)
            if av_uncertainty is not None:
                parts.append(av_uncertainty.unsqueeze(-1) if av_uncertainty.dim() == 1 else av_uncertainty)

        state = torch.cat(parts, dim=-1)  # [B, state_dim]
        return state

    def forward(self,
                fer_logits: torch.Tensor,
                fer_quality: torch.Tensor,
                fer_uncertainty: torch.Tensor,
                perclos: torch.Tensor,
                q_perclos: torch.Tensor,
                p_drowsy: torch.Tensor,
                h_event: Optional[torch.Tensor] = None,
                av_logits: Optional[torch.Tensor] = None,
                av_arousal: Optional[torch.Tensor] = None,
                av_valence: Optional[torch.Tensor] = None,
                av_quality: Optional[torch.Tensor] = None,
                av_uncertainty: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns dict:
            emotion_logits: [B, num_classes]
            arousal_valence: [B, 2]
            drowsy_logits: [B, 3]
            gate_weight: [B, 1] (if A/V Expert available)
            fused_emotion_probs: [B, C] (gated mixture, if applicable)
        """
        state = self.assemble_state(
            fer_logits, fer_quality, fer_uncertainty,
            perclos, q_perclos, p_drowsy,
            h_event, av_logits, av_arousal, av_valence,
            av_quality, av_uncertainty,
        )

        # Gate weight
        gate_w = None
        fused_probs = None
        if self.use_av_expert and av_logits is not None:
            gate_w = self.gating(state)  # [B, 1]

            # Log-probability mixture
            log_p_fer = F.log_softmax(fer_logits, dim=-1)  # [B, C_fer]
            log_p_av = F.log_softmax(av_logits, dim=-1)    # [B, C_av]

            # Align dimensions if different class counts
            C_fer = log_p_fer.size(-1)
            C_av = log_p_av.size(-1)
            C_min = min(C_fer, C_av)
            log_p_fer_aligned = log_p_fer[:, :C_min]
            log_p_av_aligned = log_p_av[:, :C_min]

            fused_log = gate_w * log_p_fer_aligned + (1 - gate_w) * log_p_av_aligned
            fused_probs = F.softmax(fused_log, dim=-1)

        # State encoding -> heads
        h = self.state_encoder(state)  # [B, hidden_dim]

        emotion_logits = self.emotion_head(h)     # [B, num_classes]
        av_pred = self.av_head(h)                  # [B, 2]
        drowsy_logits = self.drowsy_head(h)        # [B, 3]

        result = {
            "emotion_logits": emotion_logits,
            "arousal_valence": av_pred,
            "drowsy_logits": drowsy_logits,
        }

        if gate_w is not None:
            result["gate_weight"] = gate_w
        if fused_probs is not None:
            result["fused_emotion_probs"] = fused_probs

        return result


class AgentLoss(nn.Module):
    """
    Multi-task loss for Agent training.

    L = w_cls * L_cls + w_av * L_AV + w_drowsy * L_drowsy
      + w_smooth * L_smooth + w_consist * L_consistency
    """

    def __init__(self,
                 w_cls: float = 1.0,
                 w_av: float = 0.5,
                 w_drowsy: float = 0.3,
                 w_smooth: float = 0.1,
                 w_consist: float = 0.1,
                 num_classes: int = 12):
        super().__init__()
        self.w_cls = w_cls
        self.w_av = w_av
        self.w_drowsy = w_drowsy
        self.w_smooth = w_smooth
        self.w_consist = w_consist

        self.cls_loss = nn.CrossEntropyLoss()
        self.av_loss = nn.MSELoss()
        self.drowsy_loss = nn.CrossEntropyLoss()

    def forward(self,
                pred: Dict[str, torch.Tensor],
                gt_emotion: torch.Tensor,
                gt_av: Optional[torch.Tensor] = None,
                gt_drowsy: Optional[torch.Tensor] = None,
                prev_emotion_logits: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: output dict from AgentGating.forward()
            gt_emotion: [B] ground truth emotion class
            gt_av: [B, 2] ground truth arousal/valence (optional)
            gt_drowsy: [B] ground truth drowsiness level (optional)
            prev_emotion_logits: [B, C] previous window's emotion logits (for smoothing)
        """
        losses = {}
        total = torch.tensor(0.0, device=gt_emotion.device)

        # Classification loss
        l_cls = self.cls_loss(pred["emotion_logits"], gt_emotion)
        losses["L_cls"] = l_cls.item()
        total = total + self.w_cls * l_cls

        # A/V regression loss
        if gt_av is not None and self.w_av > 0:
            l_av = self.av_loss(pred["arousal_valence"], gt_av)
            losses["L_av"] = l_av.item()
            total = total + self.w_av * l_av

        # Drowsiness loss
        if gt_drowsy is not None and self.w_drowsy > 0:
            l_drowsy = self.drowsy_loss(pred["drowsy_logits"], gt_drowsy)
            losses["L_drowsy"] = l_drowsy.item()
            total = total + self.w_drowsy * l_drowsy

        # Temporal smoothing loss (KL between consecutive windows)
        if prev_emotion_logits is not None and self.w_smooth > 0:
            p_curr = F.log_softmax(pred["emotion_logits"], dim=-1)
            p_prev = F.softmax(prev_emotion_logits.detach(), dim=-1)
            l_smooth = F.kl_div(p_curr, p_prev, reduction="batchmean")
            losses["L_smooth"] = l_smooth.item()
            total = total + self.w_smooth * l_smooth

        # Consistency loss (agent output vs expert outputs)
        if "fused_emotion_probs" in pred and self.w_consist > 0:
            p_agent = F.log_softmax(pred["emotion_logits"][:, :pred["fused_emotion_probs"].size(-1)], dim=-1)
            p_fused = pred["fused_emotion_probs"].detach()
            l_consist = F.kl_div(p_agent, p_fused, reduction="batchmean")
            losses["L_consist"] = l_consist.item()
            total = total + self.w_consist * l_consist

        losses["total"] = total.item()
        return total, losses
