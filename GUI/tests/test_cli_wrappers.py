"""Tests for the thin command-line wrapper scripts.

These tests do not exercise real training or data loading. Instead they verify
that each script:
- parses the intended CLI arguments
- builds the expected config object
- prints the expected JSON summary shape
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from xfmr_v2.search import SearchConfig
from xfmr_v2.suggest import SuggestConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
# `runpy` needs the real repository root so each CLI test can execute the script
# file exactly the way a user would from the command line.


def test_suggest_initial_settings_cli_passes_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The suggest CLI should forward parsed arguments into `SuggestConfig`."""

    import xfmr_v2.suggest as suggest_module

    seen: dict[str, SuggestConfig] = {}

    def fake_suggest_initial_settings(config: SuggestConfig) -> dict[str, object]:
        seen["config"] = config
        return {"status": "ok", "source": "stub"}

    monkeypatch.setattr(suggest_module, "suggest_initial_settings", fake_suggest_initial_settings)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "suggest_initial_settings.py",
            "--input-feature-path",
            str(tmp_path / "input"),
            "--ground-truth-data-dir",
            str(tmp_path / "output"),
            "--cache-path",
            "artifacts/cache/cli_suggest_test.npz",
            "--seed",
            "9",
            "--train-frac",
            "0.7",
            "--val-frac",
            "0.2",
            "--max-samples",
            "12",
            "--variance-threshold",
            "0.9",
        ],
    )

    # Running the script through `runpy` exercises the real argparse wiring.
    runpy.run_path(str(REPO_ROOT / "suggest_initial_settings.py"), run_name="__main__")

    printed = json.loads(capsys.readouterr().out)
    assert printed == {"status": "ok", "source": "stub"}
    assert seen["config"].data_root is None
    assert seen["config"].input_feature_path == str(tmp_path / "input")
    assert seen["config"].ground_truth_data_dir == str(tmp_path / "output")
    assert seen["config"].cache_path == "artifacts/cache/cli_suggest_test.npz"
    assert seen["config"].seed == 9
    assert seen["config"].train_frac == pytest.approx(0.7)
    assert seen["config"].val_frac == pytest.approx(0.2)
    assert seen["config"].max_samples == 12
    assert seen["config"].variance_threshold == pytest.approx(0.9)


def test_quick_search_cli_passes_arguments_and_prints_compact_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The quick-search CLI should build `SearchConfig` and print the compact summary."""

    import xfmr_v2.search as search_module

    seen: dict[str, object] = {}

    def fake_quick_hyperparameter_search(config: SearchConfig, show_progress: bool = False) -> dict[str, object]:
        seen["config"] = config
        seen["show_progress"] = show_progress
        return {
            "status": "ok",
            "run_dir": "C:/tmp/search_run",
            "objective_label": "Best accuracy",
            "epochs_per_trial": 8,
            "trial_count_completed": 3,
            "recommended_trial": {"label": "base"},
            "alternative_trials": [{"label": "smaller_model"}],
            "recommended_full_config": {"width": 192},
            "tradeoff_plot_path": "C:/tmp/tradeoff_plot.png",
            "trial_results_path": "C:/tmp/trial_results.json",
        }

    monkeypatch.setattr(search_module, "quick_hyperparameter_search", fake_quick_hyperparameter_search)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quick_search.py",
            "--input-feature-path",
            str(tmp_path / "input"),
            "--ground-truth-data-dir",
            str(tmp_path / "output"),
            "--cache-path",
            "artifacts/cache/cli_search_test.npz",
            "--output-dir",
            "artifacts/runs/cli_search_test",
            "--seed",
            "5",
            "--train-frac",
            "0.65",
            "--val-frac",
            "0.2",
            "--max-samples",
            "14",
            "--search-max-samples",
            "10",
            "--trial-count",
            "3",
            "--epochs-per-trial",
            "8",
            "--objective",
            "best_accuracy",
            "--variance-threshold",
            "0.92",
            "--show-trial-progress",
        ],
    )

    runpy.run_path(str(REPO_ROOT / "quick_search.py"), run_name="__main__")

    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "status": "ok",
        "run_dir": "C:/tmp/search_run",
        "objective": "Best accuracy",
        "epochs_per_trial": 8,
        "trial_count_completed": 3,
        "recommended_trial": {"label": "base"},
        "alternative_trials": [{"label": "smaller_model"}],
        "recommended_full_config": {"width": 192},
        "tradeoff_plot_path": "C:/tmp/tradeoff_plot.png",
        "trial_results_path": "C:/tmp/trial_results.json",
    }
    assert seen["show_progress"] is True
    assert isinstance(seen["config"], SearchConfig)
    assert seen["config"].data_root is None
    assert seen["config"].input_feature_path == str(tmp_path / "input")
    assert seen["config"].ground_truth_data_dir == str(tmp_path / "output")
    assert seen["config"].cache_path == "artifacts/cache/cli_search_test.npz"
    assert seen["config"].output_dir == "artifacts/runs/cli_search_test"
    assert seen["config"].seed == 5
    assert seen["config"].train_frac == pytest.approx(0.65)
    assert seen["config"].val_frac == pytest.approx(0.2)
    assert seen["config"].max_samples == 14
    assert seen["config"].search_max_samples == 10
    assert seen["config"].trial_count == 3
    assert seen["config"].epochs_per_trial == 8
    assert seen["config"].objective == "best_accuracy"
    assert seen["config"].variance_threshold == pytest.approx(0.92)
    assert seen["config"].show_trial_progress is True
