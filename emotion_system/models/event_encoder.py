"""
Event Encoder: Frame/Segment-level signals -> Event-level summary
==================================================================
Pipeline Layer 4.

Temporal context module that summarizes noisy per-frame outputs
from FER Expert, A/V Expert, and PerClos into a single event-level
latent vector h_event.

Supports:
  - GRU (lightweight, default for Jetson)
  - TCN (1D temporal convolution)

Input per timestep t:
  z(t) = concat(z_FER(t), z_AV(t), perclos(t), q_perclos(t), ...)

Output:
  h_event: [B, d_event] summary of the temporal window
"""

import torch
import torch.nn as nn


class GRUEventEncoder(nn.Module):
    """Bi-GRU based event encoder. Default for Jetson deployment."""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_layers: int = 1, dropout: float = 0.1,
                 bidirectional: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        dir_mult = 2 if bidirectional else 1
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * dir_mult, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.out_dim = hidden_dim

    def forward(self, z_seq: torch.Tensor,
                lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z_seq: [B, T, input_dim] concatenated expert outputs over time
            lengths: [B] actual sequence lengths (for packing, optional)

        Returns:
            h_event: [B, hidden_dim] event-level summary
        """
        x = self.input_proj(z_seq)  # [B, T, hidden_dim]

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        out, h_n = self.gru(x)  # h_n: [n_layers*dirs, B, hidden_dim]

        if lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        if self.bidirectional:
            # Concat last hidden from both directions
            h_fwd = h_n[-2]  # [B, hidden_dim]
            h_bwd = h_n[-1]  # [B, hidden_dim]
            h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # [B, hidden_dim*2]
        else:
            h_cat = h_n[-1]  # [B, hidden_dim]

        h_event = self.output_proj(h_cat)  # [B, hidden_dim]
        return h_event


class TCNBlock(nn.Module):
    """Single TCN block with residual connection."""

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size,
                      padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size,
                      padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T]"""
        out = self.conv(x)
        # Causal trim: remove future padding
        if out.size(-1) > x.size(-1):
            out = out[:, :, :x.size(-1)]
        return x + out


class TCNEventEncoder(nn.Module):
    """1D Temporal Convolutional Network event encoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_layers: int = 3, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            TCNBlock(hidden_dim, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(n_layers)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = hidden_dim

    def forward(self, z_seq: torch.Tensor,
                lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z_seq: [B, T, input_dim]
        Returns:
            h_event: [B, hidden_dim]
        """
        x = self.input_proj(z_seq)  # [B, T, hidden_dim]
        x = x.permute(0, 2, 1)  # [B, hidden_dim, T]

        for block in self.blocks:
            x = block(x)

        if lengths is not None:
            # Mask padding before pooling
            B, C, T = x.shape
            mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            x = x * mask.unsqueeze(1).float()
            h_event = x.sum(dim=-1) / lengths.unsqueeze(1).float().clamp(min=1)
        else:
            h_event = self.pool(x).squeeze(-1)  # [B, hidden_dim]

        return h_event


def build_event_encoder(encoder_type: str = "gru", **kwargs) -> nn.Module:
    """Factory function for event encoders."""
    if encoder_type == "gru":
        return GRUEventEncoder(**kwargs)
    elif encoder_type == "tcn":
        return TCNEventEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown event encoder type: {encoder_type}")
