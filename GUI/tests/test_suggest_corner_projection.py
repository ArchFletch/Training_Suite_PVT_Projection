"""PVT corner detection and the corner-projection recommendation.

``detect_corner_columns`` decides whether a dataset re-simulates the same design
across PVT corners, and ``suggest_initial_settings`` turns that into a
SpectraHydraProj recommendation with the corner columns already filled in. These
tests pin both halves plus the join between them: a recommended config must be
one the model builder actually accepts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from xfmr_v2 import runner
from xfmr_v2.data import _save_cache, detect_corner_columns
from xfmr_v2.suggest import SuggestConfig, suggest_initial_settings

# A design-of-experiments crossed with a PVT corner grid: every design is
# simulated at each of 3 temperatures x 3 supplies.
TEMPERATURES = [-40.0, 27.0, 125.0]
SUPPLIES = [0.9, 1.0, 1.1]
NUM_DESIGNS = 24


def _pvt_features() -> tuple[np.ndarray, list[str]]:
    """Rows of (width, spacing, fingers, Temp_C, VDD) over designs x corners.

    ``fingers`` is a deliberately low-cardinality DESIGN knob: it has only four
    levels, so a cardinality-only rule would call it a corner column, but it is
    constant within a design and must survive as part of the design key.
    """
    rng = np.random.default_rng(11)
    widths = rng.uniform(1.0, 9.0, NUM_DESIGNS)
    spacings = rng.uniform(0.1, 0.9, NUM_DESIGNS)
    fingers = np.array([1.0 + (d % 4) for d in range(NUM_DESIGNS)])

    rows = []
    for d in range(NUM_DESIGNS):
        for temp in TEMPERATURES:
            for vdd in SUPPLIES:
                rows.append([widths[d], spacings[d], fingers[d], temp, vdd])
    return np.asarray(rows, dtype=np.float32), ["width", "spacing", "fingers", "Temp_C", "VDD"]


def _smooth_targets(features: np.ndarray, channels: int = 2, freqs: int = 8) -> np.ndarray:
    """Low-rank smooth spectra so the capacity tier lands above 'conservative'."""
    grid = np.linspace(0.0, 1.0, freqs)
    scale = features[:, 0] * 0.1 + features[:, 3] * 0.001 + features[:, 4]
    offset = features[:, 1] * 0.5
    out = np.empty((len(features), channels, freqs), dtype=np.float32)
    for c in range(channels):
        out[:, c, :] = (
            scale[:, None] * np.sin((c + 1) * np.pi * grid)[None, :] + offset[:, None]
        )
    return out


def _write_cache(path: Path, features: np.ndarray, names: list[str]) -> Path:
    targets = _smooth_targets(features)
    _save_cache(
        path,
        dataset_root=path.parent,
        dataset_name="PvtCornerDataset",
        features=features.astype(np.float32),
        targets=targets,
        frequency_hz=np.linspace(1e9, 8e9, targets.shape[-1]).astype(np.float32),
        feature_names=names,
        channel_names=["gain", "phase"],
        target_names=["gain", "phase"],
        channel_units=["dB", "deg"],
        channel_transforms=["", ""],
        sweep_label="Frequency (GHz)",
    )
    return path


# ---------------------------------------------------------------------------
# detect_corner_columns
# ---------------------------------------------------------------------------
def test_detects_the_corner_axes_of_a_cross_product() -> None:
    features, names = _pvt_features()
    found = detect_corner_columns(features, names)

    assert found["corner_columns"] == ["Temp_C", "VDD"]
    assert found["design_count"] == NUM_DESIGNS
    assert found["corner_count"] == len(TEMPERATURES) * len(SUPPLIES)
    assert found["rows_per_design"] == pytest.approx(9.0)
    assert found["designs_at_multiple_corners"] == pytest.approx(1.0)


def test_low_cardinality_design_knob_is_not_called_a_corner() -> None:
    """``fingers`` has four levels but never varies within a design.

    Cardinality alone would sweep it into the corner set, which would merge
    designs that differ and hand the projection a geometry knob to embed.
    """
    features, names = _pvt_features()
    assert "fingers" not in detect_corner_columns(features, names)["corner_columns"]


def test_corner_columns_are_found_without_helpful_names() -> None:
    """Detection is structural, so obfuscated column names still work."""
    features, _ = _pvt_features()
    opaque = ["c0", "c1", "c2", "c3", "c4"]
    found = detect_corner_columns(features, opaque)

    assert found["corner_columns"] == ["c3", "c4"]
    # Nothing in those names says "PVT", which the caller needs in order to warn.
    assert found["name_matched_columns"] == []


def test_randomized_sweep_reports_no_corner_structure() -> None:
    """Every row its own design: there is nothing to condition on."""
    rng = np.random.default_rng(3)
    features = rng.uniform(0.0, 1.0, (120, 4)).astype(np.float32)
    found = detect_corner_columns(features, ["a", "b", "c", "d"])

    assert found["corner_columns"] == []
    assert found["design_count"] == 0


def test_no_design_key_left_refuses_to_guess() -> None:
    """When every column is corner-like there is no way to separate the two."""
    rng = np.random.default_rng(5)
    features = rng.integers(0, 3, (60, 3)).astype(np.float32)
    assert detect_corner_columns(features, ["p", "q", "r"])["corner_columns"] == []


def test_name_matching_ignores_substrings_of_unrelated_words() -> None:
    features, _ = _pvt_features()
    from xfmr_v2.data import corner_column_name_matches

    assert corner_column_name_matches(["Temp_C", "VDD", "proc_ss", "pvt_corner"]) == [
        "Temp_C",
        "VDD",
        "proc_ss",
        "pvt_corner",
    ]
    # "template" contains "temp" and "processing_gain" contains "proc"; neither is
    # a corner condition, and a naive substring match would flag both.
    assert corner_column_name_matches(["template", "processing_gain", "width"]) == []
    assert features.shape[1] == 5


# ---------------------------------------------------------------------------
# suggest_initial_settings
# ---------------------------------------------------------------------------
def test_pvt_dataset_recommends_the_corner_projection(tmp_path: Path) -> None:
    features, names = _pvt_features()
    cache_path = _write_cache(tmp_path / "pvt_cache.npz", features, names)

    result = suggest_initial_settings(SuggestConfig(data_root=None, cache_path=str(cache_path)))
    config = result["suggested_baseline_config"]

    assert config["model_type"] == "SpectraHydraProj"
    assert config["projection_columns"] == ["Temp_C", "VDD"]
    assert config["projection_dim"] == 16
    # The shared-encoder recipe carries over unchanged: the projection is one
    # small extra Linear, not a different architecture.
    assert config["loss_function"] == "mse"
    assert config["scheduler"] == "cosine"
    assert config["learning_rate"] == pytest.approx(1e-3)

    diagnostics = result["diagnostics"]
    assert diagnostics["detected_corner_columns"] == ["Temp_C", "VDD"]
    assert diagnostics["detected_corner_count"] == 9
    assert diagnostics["corner_named_columns"] == ["Temp_C", "VDD"]
    assert "SpectraHydraProj" in result["baseline_rationale"]["model_type"]


def test_recommended_projection_columns_are_ones_the_model_accepts(tmp_path: Path) -> None:
    """The join that matters: a recommended config must build.

    Corner names are resolved against the ACTIVE (constant-dropped) feature list
    at model-build time, so a recommendation drawn from a different list would
    raise here instead of training.
    """
    features, names = _pvt_features()
    cache_path = _write_cache(tmp_path / "pvt_cache.npz", features, names)
    result = suggest_initial_settings(SuggestConfig(data_root=None, cache_path=str(cache_path)))
    config = result["suggested_baseline_config"]

    kwargs = runner.resolve_projection_kwargs(
        config["model_type"],
        config["projection_columns"],
        config["projection_dim"],
        result["active_input_feature_names"],
    )
    model = runner.build_model(
        config["model_type"],
        num_frequencies=8,
        input_feature_dim=len(result["active_input_feature_names"]),
        ground_truth_channels=2,
        width=config["width"],
        depth=config["depth"],
        **kwargs,
    )
    features_in = torch.randn(4, len(result["active_input_feature_names"]))
    assert model(features_in).shape == (4, 2, 8)


def test_pvt_dataset_warns_that_the_row_level_split_leaks(tmp_path: Path) -> None:
    """Corner rows of one design are near-duplicates across train and test."""
    features, names = _pvt_features()
    cache_path = _write_cache(tmp_path / "pvt_cache.npz", features, names)

    warnings_text = "\n".join(
        suggest_initial_settings(SuggestConfig(data_root=None, cache_path=str(cache_path)))["warnings"]
    )
    assert "PVT corner structure was detected" in warnings_text
    assert "--split-corner-columns" in warnings_text


def test_structural_only_corner_columns_are_flagged_for_review(tmp_path: Path) -> None:
    features, _ = _pvt_features()
    cache_path = _write_cache(tmp_path / "opaque.npz", features, ["c0", "c1", "c2", "c3", "c4"])

    result = suggest_initial_settings(SuggestConfig(data_root=None, cache_path=str(cache_path)))
    assert result["suggested_baseline_config"]["projection_columns"] == ["c3", "c4"]
    assert any("not from their names" in w for w in result["warnings"])


def test_non_pvt_dataset_keeps_the_plain_model(tmp_path: Path) -> None:
    """No corner structure must leave the recommendation exactly as it was."""
    rng = np.random.default_rng(17)
    features = rng.uniform(0.5, 2.0, (216, 5)).astype(np.float32)
    cache_path = _write_cache(
        tmp_path / "plain.npz", features, ["width", "spacing", "fingers", "turns", "radius"]
    )

    result = suggest_initial_settings(SuggestConfig(data_root=None, cache_path=str(cache_path)))
    config = result["suggested_baseline_config"]

    assert config["model_type"] == "SpectraHydra"
    assert config["projection_columns"] is None
    assert result["diagnostics"]["detected_corner_columns"] == []
    assert not any("PVT corner structure was detected" in w for w in result["warnings"])

