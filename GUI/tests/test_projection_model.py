"""Tests for the SpectraHydraProj PVT corner-projection model.

SpectraHydraProj is a SpectraHydra that passes the configured corner/condition
columns (temperature, supply, process one-hots) through a trainable linear
projection and concatenates the embedding onto the input features. These tests
pin down the model registration, the name -> index resolution against the
active (constant-dropped) feature list, and the end-to-end baseline and
self-transfer flows including checkpoint contents.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xfmr_v2 import runner
from xfmr_v2.model import SpectraHydra, SpectraHydraProj


def test_projection_model_registered_and_forward_shape() -> None:
    assert "SpectraHydraProj" in runner.MODEL_TYPES
    assert runner.canonical_model_type("SpectraHydraProj") == "SpectraHydraProj"

    model = runner.build_model(
        "SpectraHydraProj",
        num_frequencies=5,
        input_feature_dim=6,
        ground_truth_channels=2,
        width=8,
        depth=2,
        corner_indices=[3, 4, 5],
        projection_dim=4,
    )
    assert isinstance(model, SpectraHydraProj)
    assert isinstance(model, SpectraHydra)  # drop-in for every SpectraHydra call site

    out = model(torch.randn(3, 6))
    assert out.shape == (3, 2, 5)

    # The projection layer and its corner indices travel inside the state dict, so
    # band-to-band warm starts and checkpoint reloads carry them automatically.
    state = model.state_dict()
    assert any(key.startswith("cemb.") for key in state)
    assert "corner_indices" in state

    clone = runner.build_model(
        "SpectraHydraProj",
        num_frequencies=5,
        input_feature_dim=6,
        ground_truth_channels=2,
        width=8,
        depth=2,
        corner_indices=[3, 4, 5],
        projection_dim=4,
    )
    clone.load_state_dict(state)  # strict — shapes must match exactly
    x = torch.randn(2, 6)
    assert torch.allclose(clone(x), model(x))


def test_projection_constructor_validates_inputs() -> None:
    with pytest.raises(ValueError):
        SpectraHydraProj(6, 2, 5, 8, 2, corner_indices=[])
    with pytest.raises(ValueError):
        SpectraHydraProj(6, 2, 5, 8, 2, corner_indices=[6])  # out of range
    with pytest.raises(ValueError):
        SpectraHydraProj(6, 2, 5, 8, 2, corner_indices=[0], projection_dim=0)


def test_resolve_projection_kwargs_by_name() -> None:
    active = ["geom1", "geom2", "Temp_C", "VDD", "proc_tt"]
    dropped = ["proc_ss"]

    # Other model types resolve to an empty dict so callers can splat blindly.
    assert runner.resolve_projection_kwargs("SpectraNet", ["Temp_C"], 16, active, dropped) == {}
    assert runner.resolve_projection_kwargs("SpectraHydra", None, 16, active, dropped) == {}

    kwargs = runner.resolve_projection_kwargs(
        "SpectraHydraProj", ["Temp_C", "VDD", "proc_tt"], 8, active, dropped
    )
    assert kwargs == {"corner_indices": [2, 3, 4], "projection_dim": 8}

    # A corner column dropped as constant is skipped (it carries no information);
    # the skip is loud (see test_resolve_projection_kwargs_warns_on_skipped_columns).
    with pytest.warns(UserWarning):
        kwargs = runner.resolve_projection_kwargs(
            "SpectraHydraProj", ["Temp_C", "proc_ss"], 8, active, dropped
        )
    assert kwargs["corner_indices"] == [2]

    with pytest.raises(ValueError, match="at least one"):
        runner.resolve_projection_kwargs("SpectraHydraProj", None, 8, active, dropped)
    with pytest.raises(ValueError, match="not input features"):
        runner.resolve_projection_kwargs("SpectraHydraProj", ["nope"], 8, active, dropped)
    with pytest.raises(ValueError, match="constant"):
        runner.resolve_projection_kwargs("SpectraHydraProj", ["proc_ss"], 8, active, dropped)

    # Duplicate names (possible via CLI flags) embed each column only once.
    kwargs = runner.resolve_projection_kwargs(
        "SpectraHydraProj", ["Temp_C", "Temp_C", "VDD"], 8, active, dropped
    )
    assert kwargs["corner_indices"] == [2, 3]


def test_resolve_projection_kwargs_warns_on_skipped_columns() -> None:
    """Skipping a split-constant corner column must be loud: constancy depends on the
    seed/split, so a silently smaller embedding could diverge between runs."""
    active = ["geom1", "Temp_C"]
    dropped = ["proc_ss"]
    with pytest.warns(UserWarning, match="proc_ss"):
        kwargs = runner.resolve_projection_kwargs(
            "SpectraHydraProj", ["Temp_C", "proc_ss"], 8, active, dropped
        )
    assert kwargs["corner_indices"] == [1]


def test_search_config_requires_projection_columns_up_front() -> None:
    """Quick search fails fast on a projection model without corner columns, before
    the expensive suggestion pass runs."""
    from xfmr_v2.search import SearchConfig, quick_hyperparameter_search

    with pytest.raises(ValueError, match="projection_columns"):
        quick_hyperparameter_search(
            SearchConfig(model_type="SpectraHydraProj", projection_columns=None)
        )


def test_projection_baseline_train_and_checkpoint(synthetic_dataset: dict[str, Path], tmp_path: Path) -> None:
    """Baseline training with SpectraHydraProj: the projection trains end-to-end and
    the checkpoint stores everything needed to rebuild the model (config names +
    active feature list + state dict with cemb)."""
    config = runner.TrainConfig(
        data_root=str(synthetic_dataset["root"]),
        input_feature_path=None,
        ground_truth_data_dir=None,
        cache_path=str(synthetic_dataset["cache_path"]),
        output_dir=str(tmp_path / "runs/proj_baseline"),
        model_type="SpectraHydraProj",
        # 'const' is constant in the fixture dataset and must be skipped; 'y' varies.
        projection_columns=["y", "const"],
        projection_dim=4,
        width=8,
        depth=2,
        epochs=2,
        batch_size=4,
        use_amp=False,
        device="cpu",
    )
    # 'const' is constant, so training warns that it was skipped from the projection.
    with pytest.warns(UserWarning, match="const"):
        summary = runner.train_baseline(config, show_progress=False)
    assert summary.get("status") != "stopped"

    checkpoint = torch.load(
        Path(summary["run_dir"]) / "best_model.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["config"]["model_type"] == "SpectraHydraProj"
    assert checkpoint["config"]["projection_columns"] == ["y", "const"]
    assert checkpoint["config"]["projection_dim"] == 4
    assert any(key.startswith("cemb.") for key in checkpoint["model_state"])
    # 'const' was dropped, so the projection embeds exactly one column ('y').
    assert checkpoint["model_state"]["cemb.weight"].shape == (4, 1)
    assert checkpoint["active_input_feature_names"] == ["x", "y"]
    assert checkpoint["dropped_input_feature_names"] == ["const"]


def test_projection_self_transfer_bands_carry_projection(
    synthetic_dataset: dict[str, Path], tmp_path: Path
) -> None:
    """Self-transfer with SpectraHydraProj: every band submodel trains the projection
    live, and final_submodels.pt persists the resolved corner kwargs for export."""
    config = runner.TransferConfig(
        cache_path=str(synthetic_dataset["cache_path"]),
        output_dir=str(tmp_path / "runs/proj_transfer"),
        model_type="SpectraHydraProj",
        projection_columns=["y"],
        projection_dim=4,
        width=8,
        depth=2,
        num_bands=3,
        iterations=1,
        transfer_epochs=1,
        batch_size=4,
        use_amp=False,
        device="cpu",
    )
    summary = runner.run_self_transfer(config, show_progress=False)
    assert summary["status"] == "ok"

    bundle = torch.load(
        Path(summary["run_dir"]) / "final_submodels.pt", map_location="cpu", weights_only=False
    )
    assert bundle["model_type"] == "SpectraHydraProj"
    assert bundle["model_kwargs"]["corner_indices"] == [1]  # 'y' after 'const' is dropped
    assert bundle["model_kwargs"]["projection_dim"] == 4
    assert all(any(key.startswith("cemb.") for key in state) for state in bundle["states"])


def test_projection_baseline_fails_loudly_without_corner_columns(
    synthetic_dataset: dict[str, Path], tmp_path: Path
) -> None:
    config = runner.TrainConfig(
        data_root=str(synthetic_dataset["root"]),
        cache_path=str(synthetic_dataset["cache_path"]),
        output_dir=str(tmp_path / "runs/proj_missing"),
        model_type="SpectraHydraProj",
        projection_columns=None,
        epochs=1,
        use_amp=False,
        device="cpu",
    )
    with pytest.raises(ValueError, match="at least one PVT corner column"):
        runner.train_baseline(config, show_progress=False)
