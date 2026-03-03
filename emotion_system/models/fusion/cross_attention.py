"""
Lightweight Cross-Attention Fusion
====================================

기존 구조 문제점:
  1. Self-Attention → Cross-Attention 순서로 역할 구분 모호
  2. Cross-Attention Q가 2개뿐 (CLS, Global) → attention 평탄화

개선:
  1. Cross-Attention FIRST: CLS가 AU tokens에서 정보 선택적 추출
  2. 그 다음 Self-Attention: fused tokens 간 정보 교환
  3. Pre-Norm 구조로 안정적 학습
  4. Gate를 per-head로 세분화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionFusion(nn.Module):
    """
    [CLS] + [Global] + [AU_1, ..., AU_K] 토큰 시퀀스를 융합.

    순서: Cross-Attention → Gated Residual → Self-Attention → FFN
    CLS token이 최종 분류에 사용됨.
    """

    def __init__(self, d_model: int = 384, n_heads: int = 8,
                 n_layers: int = 1, dropout: float = 0.1,
                 gate_init: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            FusionLayer(d_model, n_heads, dropout, gate_init)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, cls_token: torch.Tensor, global_feat: torch.Tensor,
                au_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_token:   [B, d] learnable CLS token (expanded)
            global_feat: [B, d] global image feature
            au_tokens:   [B, K, d] AU region features

        Returns:
            tokens: [B, K+2, d] fused token sequence (CLS at index 0)
        """
        B, K, d = au_tokens.shape

        # Assemble token sequence: [CLS, Global, AU_1, ..., AU_K]
        tokens = torch.cat([
            cls_token.unsqueeze(1),    # [B, 1, d]
            global_feat.unsqueeze(1),  # [B, 1, d]
            au_tokens,                 # [B, K, d]
        ], dim=1)  # [B, K+2, d]

        for layer in self.layers:
            tokens = layer(tokens, K)

        return self.final_norm(tokens)


class FusionLayer(nn.Module):
    """Single fusion layer: Cross-Attn → Self-Attn → FFN"""

    def __init__(self, d_model, n_heads, dropout, gate_init):
        super().__init__()

        # Cross-Attention: Q=[CLS,Global], KV=[AU tokens]
        self.cross_norm_q = nn.LayerNorm(d_model)
        self.cross_norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # Learnable gate (per-dim for fine control)
        self.gate = nn.Parameter(torch.full((d_model,), gate_init))

        # Self-Attention: all tokens
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # FFN
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, K: int) -> torch.Tensor:
        """
        tokens: [B, K+2, d] where first 2 are CLS and Global
        K: number of AU tokens
        """
        # ── Cross-Attention: CLS+Global query AU tokens ──
        q_tokens = tokens[:, :2, :]   # [B, 2, d]
        kv_tokens = tokens[:, 2:, :]  # [B, K, d]

        q_normed = self.cross_norm_q(q_tokens)
        kv_normed = self.cross_norm_kv(kv_tokens)

        cross_out, _ = self.cross_attn(q_normed, kv_normed, kv_normed)

        # Gated residual: sigmoid gate controls information flow
        gate = torch.sigmoid(self.gate)  # [d]
        q_tokens = q_tokens + gate * cross_out

        # Reassemble
        tokens = torch.cat([q_tokens, kv_tokens], dim=1)

        # ── Self-Attention: all tokens interact ──
        normed = self.self_norm(tokens)
        self_out, _ = self.self_attn(normed, normed, normed)
        tokens = tokens + self_out

        # ── FFN ──
        tokens = tokens + self.ffn(self.ffn_norm(tokens))

        return tokens
