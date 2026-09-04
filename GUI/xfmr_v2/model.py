"""Compact feed-forward surrogate models for XFMR spectral prediction.

Each model maps per-sample input features to an entire ``(channels, frequency)``
output curve in one shot:
- :class:`SpectraNet` is a single dense network from inputs to the flattened output.
- :class:`SpectraHydra` shares an encoder and uses one linear head per channel.
- :class:`SpectraHydraProj` adds a learned projection of PVT corner columns on
  top of :class:`SpectraHydra`.
- :class:`SpectraTrunk` runs one weight-shared residual trunk per frequency
  point (context MLP + Fourier frequency embedding) instead of a single
  spectrum-sized output layer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

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

    It predicts the entire ``(channels, num_frequencies)`` output from input
    features in one shot.

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

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
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
    the original CTLE training notebooks.  It predicts all frequency points in
    one shot, similar to :class:`SpectraNet`.
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

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(input_features)
        # Stack per-channel predictions into (batch, channels, frequency).
        return torch.stack([head(latent) for head in self.heads], dim=1)


class SpectraTrunk(nn.Module):
    """Frequency-trunk surrogate: one shared network evaluated at every frequency.

    Architecture from the M:N transformer study's "knobs -> trunk" model. The
    other models in this module emit the whole spectrum from one output layer;
    here a context MLP compresses the input features into a frequency-flat
    context vector, that context is broadcast into one row per frequency point,
    each row appends a Fourier embedding of its frequency coordinate, and a
    single weight-shared trunk of pre-LayerNorm residual blocks maps every row
    to that frequency's channel values::

        ctx    = GELU(Linear(GELU(Linear(features -> 256)) -> 256))
        row(f) = [ ctx | fn, sin(pi k fn), cos(pi k fn) for k=1..16 ]
        h      = Linear(row -> width)
        h      = h + Linear(Dropout(GELU(Linear(LayerNorm(h)))))   # x depth
        out(f) = Linear(LayerNorm(h) -> channels)

    The trunk cannot memorize one output slot per frequency point — its outputs
    are forced to be a function of the Fourier embedding — which is the study's
    measured advantage over spectrum-sized heads on knob-style tabular inputs.

    ``fn`` is the frequency INDEX normalized to [0, 1] (a persistent buffer),
    not the physical GHz value: every instance built with the same
    ``num_frequencies`` is byte-identical in architecture AND buffers, which the
    self-transfer band warm start relies on (band submodels strictly load each
    other's state dicts and never see the physical grid). On a uniform grid the
    two normalizations are affinely equivalent; on a log-spaced grid the index
    form keeps the embedding uniformly sampled.

    Unlike the sibling models, weight shapes do not depend on
    ``num_frequencies`` — only the Fourier buffer does.
    """

    # Fixed sub-architecture (not exposed through TrainConfig): checkpoints and
    # ONNX export rebuild the model from (model_type, width, depth) alone, so
    # anything else that changes shapes must be a constant.
    CONTEXT_DIM = 256
    NUM_HARMONICS = 16
    DROPOUT = 0.05

    def __init__(
        self,
        input_feature_dim: int,
        ground_truth_channels: int,
        num_frequencies: int,
        width: int = 512,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be a positive integer.")
        if depth < 1:
            raise ValueError("depth must be at least 1 (number of residual blocks).")
        self.ground_truth_channels = ground_truth_channels
        self.num_frequencies = num_frequencies

        self.context = nn.Sequential(
            nn.Linear(input_feature_dim, self.CONTEXT_DIM),
            nn.GELU(),
            nn.Linear(self.CONTEXT_DIM, self.CONTEXT_DIM),
            nn.GELU(),
        )
        fourier_dim = 1 + 2 * self.NUM_HARMONICS
        self.entry = nn.Linear(self.CONTEXT_DIM + fourier_dim, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Dropout(self.DROPOUT),
                nn.Linear(width, width),
            )
            for _ in range(depth)
        )
        self.head_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, ground_truth_channels)

        # One frequency point degenerates to fn = 0 rather than a 0/0 division.
        if num_frequencies == 1:
            coords = torch.zeros(1)
        else:
            coords = torch.linspace(0.0, 1.0, num_frequencies)
        harmonics = torch.arange(1, self.NUM_HARMONICS + 1, dtype=coords.dtype)
        angles = math.pi * coords[:, None] * harmonics[None, :]
        fourier = torch.cat([coords[:, None], torch.sin(angles), torch.cos(angles)], dim=1)
        self.register_buffer("fourier_features", fourier)  # (num_frequencies, 33)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        context = self.context(input_features)  # (batch, CONTEXT_DIM)
        batch = context.shape[0]
        rows = torch.cat(
            [
                context.unsqueeze(1).expand(-1, self.num_frequencies, -1),
                self.fourier_features.unsqueeze(0).expand(batch, -1, -1),
            ],
            dim=-1,
        )  # (batch, frequency, CONTEXT_DIM + 33)
        hidden = self.entry(rows)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        out = self.head(self.head_norm(hidden))  # (batch, frequency, channels)
        return out.transpose(1, 2)  # (batch, channels, frequency)


class SpectraHydraProj(SpectraHydra):
    """:class:`SpectraHydra` with a learned linear projection of PVT corner columns.

    For PVT (process / voltage / temperature) datasets, the corner-condition
    columns (e.g. ``Temp_C``, ``VDD``, process one-hots) are passed through a
    trainable ``Linear(len(corner_indices) -> projection_dim)`` and the projected
    embedding is concatenated onto the full input-feature vector before the
    shared encoder.  The projection is a regular submodule, so it trains
    end-to-end with the rest of the network — during baseline training and
    inside every band submodel during self-transfer.

    ``corner_indices`` index into the model's input features (the *active*
    feature columns, after constant columns are dropped), in the same order the
    features are fed to :meth:`forward`.  ``input_feature_dim`` is still the raw
    input width: callers feed the same feature vector as for the other models,
    and the embedding is derived internally.
    """

    def __init__(
        self,
        input_feature_dim: int,
        ground_truth_channels: int,
        num_frequencies: int,
        width: int = 256,
        depth: int = 5,
        corner_indices: Sequence[int] = (),
        projection_dim: int = 16,
    ) -> None:
        corner_indices = [int(i) for i in corner_indices]
        if not corner_indices:
            raise ValueError("SpectraHydraProj requires at least one corner column index.")
        if any(i < 0 or i >= input_feature_dim for i in corner_indices):
            raise ValueError(
                f"corner_indices {corner_indices} out of range for input_feature_dim={input_feature_dim}."
            )
        if projection_dim <= 0:
            raise ValueError("projection_dim must be a positive integer.")
        # The encoder sees the raw features plus the projected corner embedding.
        super().__init__(
            input_feature_dim + projection_dim,
            ground_truth_channels,
            num_frequencies,
            width,
            depth,
        )
        self.cemb = nn.Linear(len(corner_indices), projection_dim)
        # A buffer (not a parameter) so the indices travel inside every state dict:
        # band-to-band warm starts, checkpoint reloads, and ONNX export all see them.
        self.register_buffer("corner_indices", torch.tensor(corner_indices, dtype=torch.long))

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        embedding = self.cemb(input_features[:, self.corner_indices])
        return super().forward(torch.cat([input_features, embedding], dim=-1))
