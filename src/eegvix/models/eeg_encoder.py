"""Spatiotemporal Transformer EEG Encoder.

Architecture:
    1. Temporal convolution stem: 1D convolutions across time per channel with residual
       connections, reducing 100 timepoints to ~25 temporal tokens.
    2. Spatial attention: Multi-head attention across 17 channels at each temporal position,
       using learned positional embeddings from 2D electrode coordinates.
    3. Temporal transformer: Stack of transformer encoder layers over the temporal dimension,
       with a prepended learnable CLS token.
    4. Optional frequency branch: Parallel FFT-based pathway that extracts spectral features
       and concatenates them before the final projection.
    5. Subject conditioning: Per-subject embedding added to CLS token.
    6. Final linear projection to output_dim (768 for CLIP ViT-L/14 compatibility).
"""

import math

import torch
import torch.nn as nn
from einops import rearrange

from eegvix.data.channel_info import get_channel_positions_tensor, N_CHANNELS
from eegvix.models.subject_embedding import SubjectEmbedding


class TemporalConvStem(nn.Module):
    """Multi-layer 1D temporal convolution with residual connections.

    Processes each channel's time series, reducing temporal resolution
    while increasing feature dimensionality.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 512,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]

        layers = []
        current_channels = in_channels
        dims = self._compute_layer_dims(in_channels, embed_dim, len(kernel_sizes))

        for i, (k, out_dim) in enumerate(zip(kernel_sizes, dims)):
            layers.append(TemporalConvBlock(current_channels, out_dim, k, dropout))
            current_channels = out_dim

        self.layers = nn.ModuleList(layers)
        self.out_dim = dims[-1]

    @staticmethod
    def _compute_layer_dims(in_dim: int, out_dim: int, n_layers: int) -> list[int]:
        """Linearly interpolate channel dimensions across layers."""
        if n_layers == 1:
            return [out_dim]
        dims = []
        for i in range(n_layers):
            d = in_dim + (out_dim - in_dim) * (i + 1) / n_layers
            dims.append(int(d))
        dims[-1] = out_dim  # Ensure exact output dim
        return dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch * n_channels, 1, n_timepoints)

        Returns:
            (batch * n_channels, embed_dim, reduced_timepoints)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class TemporalConvBlock(nn.Module):
    """Single temporal conv block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=padding)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions change
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1, stride=2)
            if in_channels != out_channels
            else nn.AvgPool1d(kernel_size=2, stride=2)
            if in_channels == out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.conv(x)
        # LayerNorm expects (batch, time, channels), so transpose
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.dropout(x)
        # Ensure matching temporal dimensions for residual
        min_t = min(x.size(2), residual.size(2))
        return x[:, :, :min_t] + residual[:, :, :min_t]


class SpatialChannelAttention(nn.Module):
    """Multi-head attention across EEG channels, informed by electrode topology.

    Uses 2D scalp positions of electrodes as positional embeddings, learned through
    a small MLP that maps (x, y) coordinates to the embedding dimension.
    """

    def __init__(self, embed_dim: int = 512, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Learn spatial positional embeddings from 2D electrode coordinates
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # Register electrode positions as buffer (not a parameter)
        self.register_buffer("channel_positions", get_channel_positions_tensor())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, embed_dim)

        Returns:
            (batch, n_channels, embed_dim)
        """
        # Add spatial positional encoding
        pos_embed = self.pos_encoder(self.channel_positions)  # (17, embed_dim)
        x_pos = x + pos_embed.unsqueeze(0)

        # Self-attention across channels
        attended, _ = self.attention(x_pos, x_pos, x_pos)
        return self.norm(x + attended)


class FrequencyBranch(nn.Module):
    """Parallel branch that extracts frequency-domain features from raw EEG.

    Computes FFT and learns features from the power spectrum of each channel.
    """

    def __init__(self, n_channels: int = 17, n_timepoints: int = 100, output_dim: int = 128):
        super().__init__()
        n_freq_bins = n_timepoints // 2 + 1  # 51 for 100 timepoints

        self.freq_encoder = nn.Sequential(
            nn.Linear(n_freq_bins, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dim),
            nn.GELU(),
        )
        self.channel_pool = nn.Sequential(
            nn.Linear(n_channels * output_dim, output_dim * 2),
            nn.GELU(),
            nn.LayerNorm(output_dim * 2),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg: (batch, n_channels, n_timepoints) raw EEG signal

        Returns:
            (batch, output_dim) frequency features
        """
        fft = torch.fft.rfft(eeg, dim=-1)
        power = fft.real ** 2 + fft.imag ** 2  # (batch, 17, 51)

        # Per-channel frequency encoding
        freq_features = self.freq_encoder(power)  # (batch, 17, output_dim)

        # Pool across channels
        pooled = freq_features.reshape(freq_features.size(0), -1)  # (batch, 17 * output_dim)
        return self.channel_pool(pooled)  # (batch, output_dim)


class EEGEncoder(nn.Module):
    """Full spatiotemporal transformer encoder for EEG signals.

    Combines temporal convolutions, spatial attention, temporal transformer,
    optional frequency features, and per-subject embeddings to produce
    CLIP-compatible embeddings.
    """

    def __init__(
        self,
        n_channels: int = 17,
        n_timepoints: int = 100,
        embed_dim: int = 512,
        num_temporal_conv_layers: int = 3,
        temporal_kernel_sizes: list[int] | None = None,
        n_spatial_heads: int = 4,
        n_temporal_transformer_layers: int = 4,
        n_temporal_heads: int = 8,
        dropout: float = 0.1,
        use_frequency_branch: bool = True,
        output_dim: int = 768,
        n_subjects: int = 10,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.embed_dim = embed_dim
        self.use_frequency_branch = use_frequency_branch

        # 1. Temporal convolution stem (per channel)
        if temporal_kernel_sizes is None:
            temporal_kernel_sizes = [7, 5, 3][:num_temporal_conv_layers]
        self.temporal_stem = TemporalConvStem(
            in_channels=1, embed_dim=embed_dim, kernel_sizes=temporal_kernel_sizes, dropout=dropout
        )

        # 2. Spatial attention across channels
        self.spatial_attention = SpatialChannelAttention(
            embed_dim=embed_dim, n_heads=n_spatial_heads, dropout=dropout
        )

        # 3. Temporal transformer with CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.temporal_pos_embed = None  # Dynamically created based on sequence length

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_temporal_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_temporal_transformer_layers
        )

        # 4. Frequency branch (optional)
        freq_dim = 128 if use_frequency_branch else 0
        if use_frequency_branch:
            self.frequency_branch = FrequencyBranch(
                n_channels=n_channels, n_timepoints=n_timepoints, output_dim=freq_dim
            )

        # 5. Subject embedding
        self.subject_embedding = SubjectEmbedding(n_subjects=n_subjects, embed_dim=embed_dim)

        # 6. Final projection
        self.norm = nn.LayerNorm(embed_dim + freq_dim)
        self.projection = nn.Linear(embed_dim + freq_dim, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _get_temporal_pos_embed(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Sinusoidal positional embeddings for the temporal transformer."""
        position = torch.arange(seq_len, device=device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=device).float()
            * (-math.log(10000.0) / self.embed_dim)
        )
        pe = torch.zeros(1, seq_len, self.embed_dim, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self,
        eeg: torch.Tensor,
        subject_ids: torch.Tensor,
        enable_subject_embed: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            eeg: (batch, 17, 100) raw EEG signal
            subject_ids: (batch,) integer subject indices
            enable_subject_embed: Set False during warmup to disable subject conditioning

        Returns:
            (batch, output_dim) EEG embedding in CLIP-compatible space
        """
        batch_size = eeg.size(0)

        # --- Temporal convolution stem (per channel) ---
        # Reshape to process each channel independently
        x = rearrange(eeg, "b c t -> (b c) 1 t")  # (batch*17, 1, 100)
        x = self.temporal_stem(x)  # (batch*17, embed_dim, ~12)
        reduced_time = x.size(2)

        # Reshape back: (batch, 17, embed_dim, reduced_time)
        x = rearrange(x, "(b c) d t -> b c t d", b=batch_size, c=self.n_channels)
        # x is now (batch, 17, reduced_time, embed_dim)

        # --- Spatial attention at each temporal position ---
        # Process each time step: attention across 17 channels
        spatial_out = []
        for t in range(reduced_time):
            channel_features = x[:, :, t, :]  # (batch, 17, embed_dim)
            attended = self.spatial_attention(channel_features)  # (batch, 17, embed_dim)
            # Pool across channels -> (batch, embed_dim)
            pooled = attended.mean(dim=1)
            spatial_out.append(pooled)

        # Stack temporal tokens: (batch, reduced_time, embed_dim)
        temporal_tokens = torch.stack(spatial_out, dim=1)

        # --- Prepend CLS token with subject embedding ---
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
        if enable_subject_embed:
            subj_embed = self.subject_embedding(subject_ids)  # (batch, embed_dim)
            cls = cls + subj_embed.unsqueeze(1)

        tokens = torch.cat([cls, temporal_tokens], dim=1)  # (batch, 1+reduced_time, embed_dim)

        # Add sinusoidal positional embeddings
        pos_embed = self._get_temporal_pos_embed(tokens.size(1), tokens.device)
        tokens = tokens + pos_embed

        # --- Temporal transformer ---
        transformer_out = self.temporal_transformer(tokens)  # (batch, 1+reduced_time, embed_dim)
        cls_out = transformer_out[:, 0, :]  # (batch, embed_dim) — CLS token output

        # --- Frequency branch (optional) ---
        if self.use_frequency_branch:
            freq_features = self.frequency_branch(eeg)  # (batch, 128)
            cls_out = torch.cat([cls_out, freq_features], dim=-1)  # (batch, embed_dim + 128)

        # --- Final projection ---
        cls_out = self.norm(cls_out)
        output = self.projection(cls_out)  # (batch, output_dim)

        return output
