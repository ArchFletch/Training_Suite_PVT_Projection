"""Integration-style tests for schema parsing, caching, suggestions, and search."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from xfmr_v2 import data, dataset_schema, search, suggest
from xfmr_v2.runner import TrainConfig
from xfmr_v2.search import SearchConfig
from xfmr_v2.suggest import SuggestConfig


def test_parse_dataset_readme_uses_parent_name_and_default_parts(tmp_path: Path) -> None:
    """A README without an explicit dataset name should fall back to the folder name."""

    dataset_root = tmp_path / "readme_defaults"
    dataset_root.mkdir()
    readme = dataset_root / "README.md"
    readme.write_text(
        """# Dataset

```json
{
  "input_feature": {
    "columns": ["x", "sample_id"],
    "feature_columns": ["x"],
    "sample_id_column": "sample_id"
  },
  "ground_truth": {
    "format": "TouchStone",
    "file_extension": ".S2P",
    "ground_truth_parameters": ["S11"]
  }
}
```
""",
        encoding="utf-8",
    )

    schema = dataset_schema.parse_dataset_readme(readme)

    assert schema.dataset_name == "readme_defaults"
    assert schema.input_feature.columns == ("x", "sample_id")
    assert schema.input_feature.feature_columns == ("x",)
    assert schema.ground_truth.format == "touchstone"
    assert schema.ground_truth.file_extension == ".s2p"
    assert schema.ground_truth.ground_truth_parts == ("re", "im")
    assert schema.ground_truth.channel_names == ["S11_re", "S11_im"]


def test_parse_dataset_readme_rejects_duplicate_columns(tmp_path: Path) -> None:
    """Schema validation should fail early when input columns are ambiguous."""

    dataset_root = tmp_path / "bad_schema"
    dataset_root.mkdir()
    readme = dataset_root / "README.md"
    readme.write_text(
        """```json
{
  "input_feature": {
    "columns": ["x", "x", "sample_id"],
    "feature_columns": ["x"],
    "sample_id_column": "sample_id"
  },
  "ground_truth": {
    "format": "touchstone",
    "file_extension": ".s2p",
    "ground_truth_parameters": ["S11"]
  }
}
```
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Input-feature columns must be unique"):
        dataset_schema.parse_dataset_readme(readme)


def test_build_cache_and_load_split_bundle_with_separate_paths(synthetic_dataset: dict[str, Path]) -> None:
    """The cache builder should work when input and output paths are passed explicitly."""

    cache_path = synthetic_dataset["cache_path"]
    summary = data.build_cache(
        input_feature_path=synthetic_dataset["input_dir"],
        ground_truth_data_dir=synthetic_dataset["output_dir"],
        cache_path=cache_path,
    )

    assert summary["status"] == "created"
    assert summary["dataset_name"] == "SyntheticTouchstoneDataset"
    assert summary["num_samples"] == 10
    assert summary["num_features"] == 3
    assert summary["num_channels"] == 4
    assert summary["num_frequencies"] == 3
    assert Path(summary["dataset_root"]) == synthetic_dataset["root"]
    assert Path(summary["input_feature_path"]) == synthetic_dataset["input_file"]
    assert Path(summary["ground_truth_data_dir"]) == synthetic_dataset["output_dir"]

    second = data.build_cache(
        input_feature_path=synthetic_dataset["input_dir"],
        ground_truth_data_dir=synthetic_dataset["output_dir"],
        cache_path=cache_path,
    )
    assert second["status"] == "existing"

    with np.load(cache_path, allow_pickle=False) as arrays:
        assert arrays["features"].shape == (10, 3)
        assert arrays["targets"].shape == (10, 4, 3)
        assert arrays["input_feature_names"].astype(str).tolist() == ["x", "y", "const"]
        assert arrays["channel_names"].astype(str).tolist() == ["S11_re", "S11_im", "S12_re", "S12_im"]
        assert arrays["targets"][0, 0, 0] == pytest.approx(0.07, abs=1e-6)
        assert arrays["targets"][0, 1, 0] == pytest.approx(-0.035, abs=1e-6)
        assert arrays["targets"][0, 2, 0] == pytest.approx(0.07 / 3.0, abs=1e-6)
        assert arrays["targets"][0, 3, 0] == pytest.approx(0.07 / 4.0, abs=1e-6)

    bundle = data.load_split_bundle(cache_path=cache_path, batch_size=4, seed=3, train_frac=0.6, val_frac=0.2)

    assert bundle.input_feature_names == ["x", "y", "const"]
    assert bundle.active_names == ["x", "y"]
    assert bundle.dropped_names == ["const"]
    assert bundle.frequency_ghz.tolist() == pytest.approx([2.0, 3.0, 4.0])
    train_x, train_y = next(iter(bundle.train_loader))
    assert train_x.shape[1] == 2
    assert tuple(train_y.shape[1:]) == (4, 3)


def test_suggest_initial_settings_reports_schema_driven_diagnostics(
    synthetic_dataset: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The suggestion pass should report diagnostics derived from the cached schema-aware data."""

    monkeypatch.setattr(
        suggest,
        "_detect_gpu",
        lambda: {"available": False, "name": None, "memory_gb": None},
    )

    result = suggest.suggest_initial_settings(
        SuggestConfig(
            input_feature_path=str(synthetic_dataset["input_dir"]),
            ground_truth_data_dir=str(synthetic_dataset["output_dir"]),
            cache_path=str(synthetic_dataset["cache_path"]),
            seed=5,
            train_frac=0.6,
            val_frac=0.2,
            variance_threshold=0.9,
        )
    )

    assert result["status"] == "ok"
    assert result["active_input_feature_names"] == ["x", "y"]
    assert result["dropped_input_feature_names"] == ["const"]
    assert result["diagnostics"]["num_samples_total"] == 10
    assert result["diagnostics"]["frequency_point_count"] == 3
    assert result["diagnostics"]["gpu_available"] is False
    assert result["suggested_baseline_config"]["data_root"] is None
    assert result["suggested_baseline_config"]["input_feature_path"] == str(synthetic_dataset["input_dir"])
    assert result["suggested_baseline_config"]["ground_truth_data_dir"] == str(synthetic_dataset["output_dir"])
    assert result["suggested_baseline_config"]["use_amp"] is False
    assert result["quick_search_hints"]["latent_dim"]


def test_normalize_objective_accepts_aliases() -> None:
    """Friendly objective spellings should normalize to canonical search names."""

    assert search._normalize_objective("best accuracy") == "best_accuracy"
    assert search._normalize_objective("fastest-acceptable") == "fastest_acceptable"


def test_quick_hyperparameter_search_ranks_trials_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A monkeypatched search run should still rank candidates and save artifacts."""

    base_config = TrainConfig(
        data_root=None,
        input_feature_path="synthetic_inputs",
        ground_truth_data_dir="synthetic_outputs",
        cache_path="synthetic_cache.npz",
        output_dir="artifacts/runs/baseline_v2",
        seed=11,
        batch_size=16,
        epochs=200,
        patience=20,
        learning_rate=1e-4,
        weight_decay=1e-4,
        gradient_clip=1.0,
        train_frac=0.8,
        val_frac=0.1,
        latent_dim=64,
        width=192,
        depth=4,
        fourier_bands=12,
        dropout=0.10,
        use_amp=False,
        max_samples=24,
    )
    suggestion = {
        "status": "ok",
        "confidence": "medium",
        "confidence_reason": "synthetic search suggestion",
        "suggested_baseline_config": asdict(base_config),
        "suggested_baseline_ranges": {
            "latent_dim": {"selected": 64, "candidates": [48, 64, 96]},
            "width": {"selected": 192, "candidates": [128, 192, 256]},
            "depth": {"selected": 4, "candidates": [3, 4, 5]},
            "fourier_bands": {"selected": 12, "candidates": [8, 12, 16]},
            "batch_size": {"selected": 16, "candidates": [8, 16, 32]},
            "learning_rate": {"selected": 1e-4, "candidates": [7e-5, 1e-4, 2e-4]},
            "dropout": {"selected": 0.10, "candidates": [0.05, 0.10, 0.15]},
            "weight_decay": {"selected": 1e-4, "candidates": [5e-5, 1e-4, 3e-4]},
        },
    }

    def fake_suggest_initial_settings(*args, **kwargs) -> dict[str, object]:
        return suggestion

    def fake_run_baseline_trial(config: TrainConfig, **kwargs) -> dict[str, object]:
        # Manufacture a simple accuracy/runtime tradeoff so ranking logic can be tested.
        mae = 0.020 + abs(config.width - 128) / 10000.0 + abs(config.depth - 3) / 5000.0
        mae += abs(config.learning_rate - 1e-4) * 100.0
        runtime = 10.0 + config.width / 32.0 + config.depth
        return {
            "status": "ok",
            "best_val_loss": mae + 0.004,
            "average_evaluation_mae": mae,
            "evaluation_loss": mae + 0.007,
            "frequency_mae": [mae, mae + 0.001, mae + 0.002],
            "runtime_seconds": runtime,
            "epochs_completed": config.epochs,
            "best_epoch": min(config.epochs, 4),
            "history": [{"epoch": 1, "train_loss": 0.2, "val_loss": 0.1}],
            "device": "cpu",
        }

    monkeypatch.setattr(search, "suggest_initial_settings", fake_suggest_initial_settings)
    monkeypatch.setattr(search, "run_baseline_trial", fake_run_baseline_trial)

    summary = search.quick_hyperparameter_search(
        SearchConfig(
            data_root=None,
            output_dir=str(tmp_path / "search_runs"),
            trial_count=4,
            epochs_per_trial=5,
            objective="balanced",
        )
    )

    assert summary["status"] == "ok"
    assert summary["objective"] == "balanced"
    assert summary["trial_count_completed"] == 4
    assert summary["recommended_trial"]["label"] == "smaller_model"
    assert Path(summary["run_dir"]).is_dir()
    assert Path(summary["tradeoff_plot_path"]).is_file()
    assert Path(summary["trial_results_path"]).is_file()
    assert (Path(summary["run_dir"]) / "summary.json").is_file()
    assert (Path(summary["run_dir"]) / "suggestion.json").is_file()

    saved_summary = json.loads((Path(summary["run_dir"]) / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["recommended_trial"]["label"] == "smaller_model"
    assert saved_summary["objective_label"] == "Balanced accuracy and time"
