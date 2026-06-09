"""Compact feed-forward surrogate models for XFMR spectral prediction.

Each model maps per-sample input features to an entire ``(channels, frequency)``
output curve in one shot:
- :class:`SpectraNet` is a single dense network from inputs to the flattened output.
- :class:`SpectraHydra` shares an encoder and uses one linear head per channel.
"""

from __future__ import annotations

import torch
from torch import nn


class DenseStack(nn.Module):
    """Shared dense feed-forward stack used by the models below."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        # Build `depth - 1` hidden blocks, then end with a final linear projection.
        for _ in range(depth - 1):
            layers += [nn.Linear(dim, hidden), nn.SiLU()]
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


class SpectraNet(nn.Module):
    """Standalone dense network that maps input features to the full spectral output.

    It does not use frequency coordinates; it predicts the entire
    ``(channels, num_frequencies)`` output from input features in one shot.

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
    ) -> None:
        super().__init__()
        self.ground_truth_channels = ground_truth_channels
        self.num_frequencies = num_frequencies
        self.network = DenseStack(
            input_feature_dim,
            width,
            ground_truth_channels * num_frequencies,
            depth,
        )

    def forward(self, input_features: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        # ``frequency`` is accepted for interface compatibility but not used.
        out = self.network(input_features)
        return out.view(input_features.shape[0], self.ground_truth_channels, self.num_frequencies)


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


class SpectraHydra(nn.Module):
    """Shared-encoder network with per-channel output heads.

    A stack of ``Linear → LayerNorm → GELU`` blocks forms a shared encoder
    (a "trunk"), and each ground-truth channel gets its own linear output
    head -- many heads on one body, hence the name.  The architecture mirrors
    the original CTLE training notebooks.  The model does **not** use frequency
    coordinates; it predicts all frequency points in one shot, similar to
    :class:`SpectraNet`.

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
