"""
K-MER Fusion: Pool-FFN + MHSA + Bi-GRU
=========================================
15 tokens × 64d → Arousal/Valence/Drowsy heads. ~144K params.

Pipeline:
  Stage 1: Token Formation (15 projectors)
  Stage 2: Intra-Modal Pool-FFN (EfficientFormer-style)
  Stage 3: Global Cross-Modal MHSA (1 layer, 4 heads)
  Stage 4: CLS Pooling → fused_repr (64d)
  Stage 5: Temporal Context (Bi-GRU, optional)
  Stage 6: Output Heads (arousal, valence, drowsy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


# ══════════════════════════════════════════════════════════════
# Stage 2: EfficientFormer-style Pool-FFN
# ══════════════════════════════════════════════════════════════

class PoolFFNBlock(nn.Module):
    """
    EfficientFormer-style local token mixer.
    AvgPool1d → FFN → residual. <0.1ms for ≤4 tokens.
    """
    def __init__(self, d_model: int = 64, expansion: int = 2, kernel: int = 3):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel, stride=1, padding=kernel // 2)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T_group, d) → (B, T_group, d)"""
        residual = x
        # Pool across token dimension
        pooled = self.pool(x.transpose(1, 2)).transpose(1, 2)
        return residual + self.ffn(self.norm(pooled))


# ══════════════════════════════════════════════════════════════
# Full Fusion Model
# ══════════════════════════════════════════════════════════════

# Token input dimensions (must match V2 NPZ structure)
TOKEN_INPUT_DIMS = {
    "kfer_probs":       7,    # T1
    "kfer_meta":        2,    # T2: quality + entropy
    "face_stats":       3,    # T3: max_conf, mean_conf, std_conf
    "emo2vec_probs":    9,    # T4
    "audeering_avd":    3,    # T5
    "audio_quality":    3,    # T6
    "bvp_features":     4,    # T7
    "eda_features":     5,    # T8
    "hr_temp_features": 6,    # T9
    "bio_quality":      3,    # T10
    "perclos_ear":      2,    # T11
    "facs_scores":      6,    # T12
    "cross_modal":      3,    # T13
    "validity_flags":   3,    # T14
}

NUM_TOKENS = 15  # 14 feature tokens + 1 CLS


class KMERFusion(nn.Module):
    """
    K-MER Multimodal Fusion: Pool-FFN(local) + MHSA(global) + Bi-GRU(temporal).
    Total ~144K params.
    """

    def __init__(self,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_modality_types: int = 5,  # face, audio, bio, aux, meta
                 pool_ffn_expansion: int = 2,
                 pool_ffn_kernel: int = 3,
                 ffn_expansion: int = 2,
                 dropout: float = 0.1,
                 use_temporal: bool = False,
                 temporal_hidden: int = 64,
                 use_valence: bool = True,
                 use_drowsy: bool = True):
        super().__init__()

        self.d_model = d_model
        self.use_temporal = use_temporal
        self.use_valence = use_valence
        self.use_drowsy = use_drowsy

        # ── Stage 1: Token Projectors ──
        self.projectors = nn.ModuleDict()
        for name, d_in in TOKEN_INPUT_DIMS.items():
            self.projectors[name] = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LayerNorm(d_model),
            )

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Modality type embedding
        # 0=face(T1-T3), 1=audio(T4-T6), 2=bio(T7-T10), 3=aux(T11-T13), 4=meta(T14,CLS)
        self.modality_embed = nn.Embedding(n_modality_types, d_model)
        # Token index → modality type
        self.register_buffer("token_modality_ids", torch.tensor(
            [0, 0, 0,  # T1-T3: face
             1, 1, 1,  # T4-T6: audio
             2, 2, 2, 2,  # T7-T10: bio
             3, 3, 3,  # T11-T13: aux
             4, 4],    # T14, CLS: meta
            dtype=torch.long
        ))

        # ── Stage 2: Intra-Modal Pool-FFN ──
        self.face_pool_ffn = PoolFFNBlock(d_model, pool_ffn_expansion, min(pool_ffn_kernel, 3))
        self.audio_pool_ffn = PoolFFNBlock(d_model, pool_ffn_expansion, min(pool_ffn_kernel, 3))
        self.bio_pool_ffn = PoolFFNBlock(d_model, pool_ffn_expansion, min(pool_ffn_kernel, 4))

        # ── Stage 3: Global MHSA (1 layer) ──
        self.mhsa_norm = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion, d_model),
            nn.Dropout(dropout),
        )

        # ── Stage 5: Temporal (optional Bi-GRU) ──
        if use_temporal:
            self.temporal_gru = nn.GRU(
                input_size=d_model,
                hidden_size=temporal_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.temporal_proj = nn.Linear(temporal_hidden * 2, d_model)
            head_input_dim = d_model
        else:
            head_input_dim = d_model

        # ── Stage 6: Output Heads ──
        self.arousal_head = nn.Sequential(
            nn.Linear(head_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        if use_valence:
            self.valence_head = nn.Sequential(
                nn.Linear(head_input_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        if use_drowsy:
            # Drowsy head uses detached fused_repr + perclos + ear
            self.drowsy_head = nn.Sequential(
                nn.Linear(head_input_dim + 2, 32),  # +2 for perclos, ear
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 3),  # 3 levels: alert, drowsy, sleeping
            )

    def _build_tokens(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Project raw features → (B, 14, d_model) feature tokens.
        Token order: T1-T14 as defined in TOKEN_INPUT_DIMS.
        """
        tokens = []
        for name in TOKEN_INPUT_DIMS:
            if name in features:
                tok = self.projectors[name](features[name])  # (B, d)
            else:
                B = next(iter(features.values())).shape[0]
                tok = torch.zeros(B, self.d_model, device=next(iter(features.values())).device)
            tokens.append(tok.unsqueeze(1))  # (B, 1, d)

        return torch.cat(tokens, dim=1)  # (B, 14, d)

    def forward(self,
                features: Dict[str, torch.Tensor],
                valid_mask: Optional[torch.Tensor] = None,
                temporal_seq: Optional[torch.Tensor] = None,
                return_attn: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: dict of {name: (B, d_in)} tensors for each token
            valid_mask: (B, 15) bool — True=valid, False=masked
            temporal_seq: (B, T_temporal, d_model) — adjacent segment reprs for GRU
            return_attn: if True, return attention weights

        Returns:
            {
                "arousal":      (B, 1) — arousal prediction [0, 1]
                "valence":      (B, 1) — valence prediction [0, 1] (if use_valence)
                "drowsy":       (B, 3) — drowsy logits (if use_drowsy)
                "fused_repr":   (B, d_model) — CLS representation (for KD)
                "attn_weights": (B, n_heads, 15, 15) — attention weights (if return_attn)
            }
        """
        B = next(iter(features.values())).shape[0]
        device = next(iter(features.values())).device

        # ── Stage 1: Token Formation ──
        tokens = self._build_tokens(features)  # (B, 14, d)

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d)
        tokens = torch.cat([tokens, cls], dim=1)  # (B, 15, d)

        # Add modality type embedding
        mod_ids = self.token_modality_ids.unsqueeze(0).expand(B, -1)
        tokens = tokens + self.modality_embed(mod_ids)

        # ── Stage 2: Intra-Modal Pool-FFN ──
        face_tokens = self.face_pool_ffn(tokens[:, 0:3, :])
        audio_tokens = self.audio_pool_ffn(tokens[:, 3:6, :])
        bio_tokens = self.bio_pool_ffn(tokens[:, 6:10, :])
        other_tokens = tokens[:, 10:, :]  # aux + meta (passthrough)

        tokens = torch.cat([face_tokens, audio_tokens, bio_tokens, other_tokens], dim=1)

        # ── Stage 3: Global MHSA ──
        # Build key_padding_mask (True = IGNORE in nn.MultiheadAttention)
        if valid_mask is not None:
            key_padding_mask = ~valid_mask  # invert: True means masked
        else:
            key_padding_mask = None

        # Pre-norm MHSA
        normed = self.mhsa_norm(tokens)
        attn_out, attn_weights = self.mhsa(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
        )
        tokens = tokens + attn_out

        # FFN
        tokens = tokens + self.ffn(self.ffn_norm(tokens))

        # ── Stage 4: CLS Pooling ──
        fused_repr = tokens[:, -1, :]  # (B, d) — CLS token is last

        # ── Stage 5: Temporal (optional) ──
        if self.use_temporal and temporal_seq is not None:
            # temporal_seq: (B, T_segments, d_model)
            gru_out, _ = self.temporal_gru(temporal_seq)
            # Use center segment
            center = temporal_seq.shape[1] // 2
            fused_repr = self.temporal_proj(gru_out[:, center, :])

        # ── Stage 6: Output Heads ──
        outputs = {
            "fused_repr": fused_repr,
            "arousal": self.arousal_head(fused_repr),
        }

        if self.use_valence:
            outputs["valence"] = self.valence_head(fused_repr)

        if self.use_drowsy and "perclos_ear" in features:
            # Detach fused_repr to prevent arousal gradient interference
            drowsy_input = torch.cat([
                fused_repr.detach(),
                features["perclos_ear"],  # (B, 2)
            ], dim=1)
            outputs["drowsy"] = self.drowsy_head(drowsy_input)

        if return_attn:
            outputs["attn_weights"] = attn_weights

        return outputs

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_param_count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_valid_mask(face_valid: torch.Tensor,
                     audio_valid: torch.Tensor,
                     bio_valid: torch.Tensor) -> torch.Tensor:
    """
    Build per-token validity mask (B, 15) from modality validity flags.

    Token layout:
      T1-T3  (0-2):  face → face_valid
      T4-T6  (3-5):  audio → audio_valid
      T7-T10 (6-9):  bio → bio_valid
      T11    (10):   perclos/ear → face_valid
      T12    (11):   facs → face_valid
      T13    (12):   cross-modal → face_valid & audio_valid
      T14    (13):   validity flags → always valid
      T15/CLS (14):  always valid
    """
    B = face_valid.shape[0]
    mask = torch.ones(B, 15, dtype=torch.bool, device=face_valid.device)

    # Face tokens
    mask[:, 0:3] = face_valid.unsqueeze(1).expand(-1, 3)
    # Audio tokens
    mask[:, 3:6] = audio_valid.unsqueeze(1).expand(-1, 3)
    # Bio tokens
    mask[:, 6:10] = bio_valid.unsqueeze(1).expand(-1, 4)
    # PERCLOS/EAR → face dependent
    mask[:, 10] = face_valid
    # FACS → face dependent
    mask[:, 11] = face_valid
    # Cross-modal → needs at least face
    mask[:, 12] = face_valid
    # T14, T15(CLS) always valid (already True)

    return mask
