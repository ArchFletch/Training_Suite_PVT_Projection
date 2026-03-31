"""Compact branch/trunk model for XFMR spectral prediction.

The model follows a simple neural-operator-style pattern:
- the branch network reads per-sample input features
- the trunk network reads frequency coordinates
- the final prediction is their interaction across a latent dimension

This layout is a good fit for problems where one sample produces an entire curve
over frequency rather than a single scalar output.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierEncoder(nn.Module):
    """Expand normalized frequency into sinusoidal features."""

    def __init__(self, bands: int) -> None:
        super().__init__()
        # Use exponentially spaced frequencies so the trunk can represent both
        # slow and fast variation across the normalized frequency axis.
        band_values = torch.tensor([2.0**i for i in range(bands)], dtype=torch.float32) * math.pi
        self.register_buffer("band_values", band_values, persistent=False)

    @property
    def out_dim(self) -> int:
        # Each band contributes one sine and one cosine term, plus the original
        # coordinate itself so the network can still use the raw frequency value.
        return 1 + 2 * int(self.band_values.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either shape `(freq,)` or `(freq, 1)` to keep caller code simple.
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        phase = x * self.band_values
        # Concatenate the raw coordinate with its sinusoidal features.
        return torch.cat((x, torch.sin(phase), torch.cos(phase)), dim=-1)


class MLP(nn.Module):
    """Shared helper for the branch and trunk nets."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        # Build `depth - 1` hidden blocks, then end with a final linear projection.
        for _ in range(depth - 1):
            layers += [nn.Linear(dim, hidden), nn.SiLU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            dim = hidden
        layers.append(nn.Linear(dim, out_dim))
        self.network = nn.Sequential(*layers)

        # Xavier initialization is a sensible default for fully connected layers and
        # keeps the model behavior more predictable across random seeds.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FlatMLPNet(nn.Module):
    """Standalone MLP that directly maps input features to the full spectral output.

    Unlike SpectralNet, this model does not use frequency coordinates or a
    branch/trunk decomposition.  It simply predicts the entire
    ``(channels, num_frequencies)`` output from input features in one shot,
    making it a natural flat-MLP baseline for comparison.

    The ``num_frequencies`` value must be provided at construction time because
    the output layer size depends on it.
    """

    def __init__(
        self,
        input_feature_dim: int,
        ground_truth_channels: int,
        num_frequencies: int,
        width: int = 256,
        depth: int = 4,
        dropout: float = 0.05,
        # Accept but ignore SpectralNet-only args so the same config dict works.
        latent_dim: int = 128,
        fourier_bands: int = 16,
    ) -> None:
        super().__init__()
        self.ground_truth_channels = ground_truth_channels
        self.num_frequencies = num_frequencies
        self.network = MLP(
            input_feature_dim,
            width,
            ground_truth_channels * num_frequencies,
            depth,
            dropout,
        )

    def forward(self, input_features: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        # ``frequency`` is accepted for interface compatibility but not used.
        out = self.network(input_features)
        return out.view(input_features.shape[0], self.ground_truth_channels, self.num_frequencies)


class SpectralNet(nn.Module):
    """Input-feature branch + frequency trunk network."""

    def __init__(
        self,
        input_feature_dim: int,
        ground_truth_channels: int,
        latent_dim: int = 128,
        width: int = 256,
        depth: int = 4,
        fourier_bands: int = 16,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        # The branch network outputs one latent vector per output channel.
        self.ground_truth_channels = ground_truth_channels
        self.latent_dim = latent_dim
        self.frequency_encoder = FourierEncoder(fourier_bands)
        self.branch = MLP(input_feature_dim, width, ground_truth_channels * latent_dim, depth, dropout)
        self.trunk = MLP(self.frequency_encoder.out_dim, width, latent_dim, depth, dropout)
        # A learned per-channel bias helps the network model simple offsets directly.
        self.bias = nn.Parameter(torch.zeros(ground_truth_channels, 1))

    def forward(self, input_features: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        # Branch output: one latent representation per sample and output channel.
        branch = self.branch(input_features).view(input_features.shape[0], self.ground_truth_channels, self.latent_dim)
        # Trunk output: one latent representation per frequency coordinate.
        trunk = self.trunk(self.frequency_encoder(frequency))
        # Contract the latent dimension to produce `(batch, channels, frequency)`.
        return torch.einsum("bcl,fl->bcf", branch, trunk) + self.bias


def _symmetric_hidden_sizes(width: int, depth: int) -> list[int]:
    """Build a symmetric expanding/contracting hidden-size list.

    For depth=5, width=256 this gives [256, 512, 1024, 512, 256], matching the
    CTLE notebook architecture.  For fewer layers the expansion is smaller.

    >>> _symmetric_hidden_sizes(256, 5)
    [256, 512, 1024, 512, 256]
    >>> _symmetric_hidden_sizes(256, 4)
    [256, 512, 512, 256]
    >>> _symmetric_hidden_sizes(256, 3)
    [256, 512, 256]
    """
    if depth <= 0:
        return []
    if depth == 1:
        return [width]
    if depth == 2:
        return [width, width]
    # Build an expanding sequence up to the midpoint, then mirror.
    half = depth // 2
    expanding = [width * (2 ** i) for i in range(half)]
    if depth % 2 == 1:
        # Odd depth: add a peak layer, then mirror the expanding part.
        peak = width * (2 ** half)
        return expanding + [peak] + expanding[::-1]
    # Even depth: mirror the expanding part (no single peak).
    return expanding + expanding[::-1]


class CTLEMultiTaskMLP(nn.Module):
    """Shared-encoder MLP with per-channel output heads.

    This architecture mirrors the CTLE training notebooks: a stack of
    ``Linear → LayerNorm → GELU`` blocks forms a shared encoder, and each
    ground-truth channel gets its own linear output head.  The model does
    **not** use frequency coordinates; it predicts all frequency points in
    one shot, similar to :class:`FlatMLPNet`.

    The ``forward`` method accepts the same ``(input_features, frequency)``
    signature as the other models for interface compatibility, but ``frequency``
    is unused.
    """

    def __init__(
        self,
        input_feature_dim: int,
        ground_truth_channels: int,
        num_frequencies: int,
        width: int = 256,
        depth: int = 5,
        dropout: float = 0.0,
        # Accept but ignore SpectralNet-only args.
        latent_dim: int = 128,
        fourier_bands: int = 16,
    ) -> None:
        super().__init__()
        self.ground_truth_channels = ground_truth_channels
        self.num_frequencies = num_frequencies

        # Build the shared encoder: Linear → LayerNorm → GELU blocks.
        # The default architecture mirrors the CTLE notebook:
        # 6 → 256 → 512 → 1024 → 512 → 256 (expanding then contracting).
        # When `depth` layers are requested with base `width`, the encoder
        # expands to 4*width at the middle then contracts symmetrically.
        encoder_layers: list[nn.Module] = []
        dim = input_feature_dim
        hidden_sizes = _symmetric_hidden_sizes(width, depth)
        for hidden in hidden_sizes:
            encoder_layers.append(nn.Linear(dim, hidden))
            encoder_layers.append(nn.LayerNorm(hidden))
            encoder_layers.append(nn.GELU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            dim = hidden
        self.encoder = nn.Sequential(*encoder_layers)

        # One output head per channel.
        self.heads = nn.ModuleList([
            nn.Linear(dim, num_frequencies) for _ in range(ground_truth_channels)
        ])

    def forward(self, input_features: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(input_features)
        # Stack per-channel predictions into (batch, channels, frequency).
        return torch.stack([head(latent) for head in self.heads], dim=1)
