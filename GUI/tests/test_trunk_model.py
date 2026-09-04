"""Tests for the SpectraTrunk frequency-trunk model.

SpectraTrunk (from the M:N transformer study's "knobs -> trunk" architecture)
broadcasts a context MLP's output into one row per frequency point, appends a
Fourier embedding of the frequency coordinate to each row, and maps every row
through one weight-shared residual trunk. These tests pin the registration, the
(batch, channels, frequency) output contract, the frequency-independence of the
weight shapes (what the self-transfer band warm start relies on), and the
end-to-end baseline and self-transfer flows.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xfmr_v2 import runner
from xfmr_v2.model import SpectraTrunk


def test_trunk_registered_and_forward_shape() -> None:
    assert "SpectraTrunk" in runner.MODEL_TYPES
    assert runner.canonical_model_type("SpectraTrunk") == "SpectraTrunk"

    model = runner.build_model(
        "SpectraTrunk",
        num_frequencies=5,
        input_feature_dim=6,
        ground_truth_channels=2,
        width=8,
        depth=2,
    )
    assert isinstance(model, SpectraTrunk)

    out = model(torch.randn(3, 6))
    assert out.shape == (3, 2, 5)

    # The Fourier frequency coordinates travel inside the state dict as a
    # persistent buffer, so checkpoint reloads restore them byte-identically.
    state = model.state_dict()
    assert "fourier_features" in state
    assert state["fourier_features"].shape == (5, 1 + 2 * SpectraTrunk.NUM_HARMONICS)

    clone = runner.build_model(
        "SpectraTrunk",
        num_frequencies=5,
        input_feature_dim=6,
        ground_truth_channels=2,
        width=8,
        depth=2,
    )
    clone.load_state_dict(state)  # strict — shapes must match exactly
    model.eval()  # dropout makes train-mode forwards non-deterministic
    clone.eval()
    x = torch.randn(2, 6)
    assert torch.allclose(clone(x), model(x))


def test_trunk_weight_shapes_do_not_depend_on_frequency_count() -> None:
    """The per-frequency head is what lets band submodels of any width share
    weights: only the Fourier coordinate buffer changes with num_frequencies."""
    kwargs = dict(input_feature_dim=4, ground_truth_channels=3, width=8, depth=2)
    small = SpectraTrunk(num_frequencies=4, **kwargs)
    large = SpectraTrunk(num_frequencies=9, **kwargs)
    small_shapes = {name: tuple(p.shape) for name, p in small.named_parameters()}
    large_shapes = {name: tuple(p.shape) for name, p in large.named_parameters()}
    assert small_shapes == large_shapes
    assert small.fourier_features.shape[0] == 4
    assert large.fourier_features.shape[0] == 9


def test_trunk_matches_reference_parameter_count() -> None:
    """The study's Model 1 (18 knobs, 20 outputs, width 512, 4 blocks) is 2.34M
    parameters — pin our implementation to that within rounding, so a silent
    architecture drift (extra layer, wrong context width) fails a test."""
    model = SpectraTrunk(
        input_feature_dim=18, ground_truth_channels=20, num_frequencies=100,
        width=512, depth=4,
    )
    num_params = sum(p.numel() for p in model.parameters())
    assert 2_300_000 < num_params < 2_400_000


def test_trunk_fourier_coordinates_are_index_normalized() -> None:
    """fn spans [0, 1] over the frequency INDEX (uniform even on log grids, and
    identical across same-width band submodels); a single point sits at 0."""
    model = SpectraTrunk(input_feature_dim=2, ground_truth_channels=1, num_frequencies=5)
    coords = model.fourier_features[:, 0]
    assert torch.allclose(coords, torch.linspace(0.0, 1.0, 5))

    single = SpectraTrunk(input_feature_dim=2, ground_truth_channels=1, num_frequencies=1)
    assert single(torch.randn(2, 2)).shape == (2, 1, 1)
    assert float(single.fourier_features[0, 0]) == 0.0


def test_trunk_constructor_validates_inputs() -> None:
    with pytest.raises(ValueError):
        SpectraTrunk(input_feature_dim=2, ground_truth_channels=1, num_frequencies=0)
    with pytest.raises(ValueError):
        SpectraTrunk(input_feature_dim=2, ground_truth_channels=1, num_frequencies=3, depth=0)


def test_trunk_dropout_is_train_mode_only() -> None:
    """The runner relies on train()/eval() toggles; eval must be deterministic."""
    model = SpectraTrunk(input_feature_dim=3, ground_truth_channels=2, num_frequencies=4)
    x = torch.randn(8, 3)
    model.eval()
    assert torch.equal(model(x), model(x))
    model.train()
    # With dropout 0.05 over thousands of activations, two forwards collide with
    # negligible probability; equality here would mean dropout is not applied.
    assert not torch.equal(model(x), model(x))


def test_trunk_baseline_train_and_checkpoint(synthetic_dataset: dict[str, Path], tmp_path: Path) -> None:
    """End-to-end baseline training: the checkpoint stores everything needed to
    rebuild the model from (model_type, width, depth) alone."""
    config = runner.TrainConfig(
        data_root=str(synthetic_dataset["root"]),
        input_feature_path=None,
        ground_truth_data_dir=None,
        cache_path=str(synthetic_dataset["cache_path"]),
        output_dir=str(tmp_path / "runs/trunk_baseline"),
        model_type="SpectraTrunk",
        width=8,
        depth=2,
        epochs=2,
        batch_size=4,
        use_amp=False,
        device="cpu",
    )
    summary = runner.train_baseline(config, show_progress=False)
    assert summary.get("status") != "stopped"

    checkpoint = torch.load(
        Path(summary["run_dir"]) / "best_model.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["config"]["model_type"] == "SpectraTrunk"
    rebuilt = runner.build_model(
        "SpectraTrunk",
        num_frequencies=checkpoint["target_mean"].shape[1],
        input_feature_dim=len(checkpoint["active_input_feature_names"]),
        ground_truth_channels=len(checkpoint["target_channel_names"]),
        width=checkpoint["config"]["width"],
        depth=checkpoint["config"]["depth"],
    )
    rebuilt.load_state_dict(checkpoint["model_state"])  # strict


def _write_six_frequency_cache(path: Path) -> None:
    """A tiny prebuilt engine cache with 6 frequency points, so a 3-band transfer
    gets width-2 bands (the synthetic_dataset fixture's 3 points would degenerate
    every band to a single frequency, which tests nothing trunk-specific)."""
    import numpy as np

    rng = np.random.default_rng(7)
    np.savez_compressed(
        path,
        features=rng.standard_normal((40, 3)).astype(np.float32),
        targets=rng.standard_normal((40, 2, 6)).astype(np.float32),
        frequency_hz=np.linspace(1e9, 6e9, 6).astype(np.float32),
        input_feature_names=np.asarray(["a", "b", "c"]),
        target_names=np.asarray(["gain", "phase"]),
        channel_names=np.asarray(["gain", "phase"]),
        channel_units=np.asarray(["dB", "deg"]),
        channel_transforms=np.asarray(["", ""]),
        sweep_label=np.asarray("Frequency (GHz)"),
    )


def test_trunk_self_transfer_band_warm_start(tmp_path: Path) -> None:
    """Self-transfer: band submodels warm-start from each other via strict state
    dict loads, which the trunk supports because its weight shapes are
    frequency-independent and same-width bands share the coordinate buffer.

    Bands must be wider than one frequency here, or the buffer-sharing property
    is vacuous (every width-1 buffer is the single row for fn=0)."""
    cache_path = tmp_path / "six_freq_cache.npz"
    _write_six_frequency_cache(cache_path)
    config = runner.TransferConfig(
        cache_path=str(cache_path),
        output_dir=str(tmp_path / "runs/trunk_transfer"),
        model_type="SpectraTrunk",
        width=8,
        depth=2,
        num_bands=3,
        iterations=1,
        transfer_epochs=1,
        batch_size=8,
        use_amp=False,
        device="cpu",
    )
    summary = runner.run_self_transfer(config, show_progress=False)
    assert summary["status"] == "ok"

    bundle = torch.load(
        Path(summary["run_dir"]) / "final_submodels.pt", map_location="cpu", weights_only=False
    )
    assert bundle["model_type"] == "SpectraTrunk"
    band_width = len(bundle["bands"][0])
    assert band_width == 2  # non-degenerate: the per-frequency mechanism is exercised
    sub = runner.build_model(
        "SpectraTrunk", num_frequencies=band_width, **bundle["model_kwargs"]
    )
    # Every band must have trained with the INDEX-normalized coordinates a fresh
    # build produces — not its own physical GHz slice. A physical-frequency
    # refactor would give each band a different buffer, and the strict
    # band-to-band warm starts would silently overwrite them with a neighbour's.
    expected_coords = sub.fourier_features.clone()
    assert torch.allclose(expected_coords[:, 0], torch.tensor([0.0, 1.0]))
    for state in bundle["states"]:
        assert torch.equal(state["fourier_features"], expected_coords)
        sub.load_state_dict(state)  # strict — the export path does exactly this
