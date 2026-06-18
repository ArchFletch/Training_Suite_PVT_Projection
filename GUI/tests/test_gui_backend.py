"""Tests for the GUI-facing backend helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from xfmr_v2 import data, gui_backend


def test_scan_dataset_returns_preview_rows(synthetic_dataset: dict[str, Path]) -> None:
    result = gui_backend.scan_dataset(
        input_feature_path=str(synthetic_dataset["input_file"]),
        ground_truth_data_dir=str(synthetic_dataset["output_dir"]),
        cache_path=str(synthetic_dataset["cache_path"]),
        train_frac=0.6,
        val_frac=0.2,
        seed=7,
    )

    assert result["status"] == "ok"
    assert result["dataset_name"] == "SyntheticTouchstoneDataset"
    assert result["schema_status"] == "Valid"
    assert result["active_input_feature_names"] == ["x", "y"]
    assert result["dropped_input_feature_names"] == ["const"]
    assert any(row[0] == "Ground-Truth Channels" for row in result["preview_rows"])


def test_scan_dataset_reports_progress_events(synthetic_dataset: dict[str, Path]) -> None:
    progress_events: list[dict[str, object]] = []

    result = gui_backend.scan_dataset(
        input_feature_path=str(synthetic_dataset["input_file"]),
        ground_truth_data_dir=str(synthetic_dataset["output_dir"]),
        cache_path=str(synthetic_dataset["cache_path"]),
        train_frac=0.6,
        val_frac=0.2,
        seed=7,
        progress_callback=progress_events.append,
    )

    assert result["status"] == "ok"
    assert progress_events
    assert progress_events[0]["event"] == "started"
    assert progress_events[-1]["event"] == "completed"
    assert all(event["phase"] == "scan" for event in progress_events)
    assert any(event["event"] == "split_loading_started" for event in progress_events)


def test_scan_dataset_loads_existing_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache_path = tmp_path / "test_cache.npz"

    monkeypatch.setattr(
        gui_backend,
        "load_existing_cache",
        lambda *a, **kw: {
            "status": "existing",
            "cache_path": str(cache_path),
            "dataset_name": "TestDataset",
            "dataset_root": str(tmp_path),
            "input_feature_path": "",
            "ground_truth_data_dir": "",
            "readme_path": "",
            "dataset_schema": {},
            "num_samples": 3,
            "num_features": 2,
            "num_channels": 2,
            "num_frequencies": 2,
        },
    )
    monkeypatch.setattr(
        gui_backend,
        "load_split_bundle",
        lambda **kwargs: SimpleNamespace(
            split_indices={"train": [0], "val": [1], "test": [2]},
            frequency_ghz=np.asarray([2.0, 3.0], dtype=float),
            active_names=["x", "y"],
            dropped_names=[],
            channel_names=["S11_re", "S11_im"],
        ),
    )

    result = gui_backend.scan_dataset(
        cache_path=str(cache_path),
        train_frac=0.6,
        val_frac=0.2,
        seed=7,
    )

    assert result["status"] == "ok"
    assert result["dataset_name"] == "TestDataset"


def test_run_search_filters_noisy_trial_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    forwarded: list[dict[str, object]] = []

    def fake_quick_hyperparameter_search(config, *, show_progress=False, progress_callback=None, should_stop=None):
        for payload in [
            {"phase": "search", "event": "trial_started", "trial_index": 2, "trial_count": 6, "trial_label": "larger_model"},
            {"phase": "baseline", "event": "started", "trial_index": 2, "trial_count": 6, "trial_label": "larger_model"},
            {"phase": "baseline", "event": "cache_check_started", "trial_index": 2, "trial_count": 6, "trial_label": "larger_model"},
            {"phase": "baseline", "event": "checkpoint_updated", "trial_index": 2, "trial_count": 6, "trial_label": "larger_model"},
            {"phase": "baseline", "event": "epoch_end", "trial_index": 2, "trial_count": 6, "trial_label": "larger_model", "epoch": 1, "total_epochs": 10},
            {"phase": "baseline", "event": "epoch_end", "trial_index": 2, "trial_count": 6, "trial_label": "larger_model", "epoch": 2, "total_epochs": 10},
            {"phase": "baseline", "event": "epoch_end", "trial_index": 2, "trial_count": 6, "trial_label": "larger_model", "epoch": 5, "total_epochs": 10},
            {"phase": "baseline", "event": "epoch_end", "trial_index": 2, "trial_count": 6, "trial_label": "larger_model", "epoch": 10, "total_epochs": 10},
            # Real runner events nest the per-run fields under "data" (emit_progress);
            # the filter must throttle that shape too.
            {"phase": "baseline", "event": "checkpoint_updated", "data": {"trial_index": 2, "epoch": 3}},
            {"phase": "baseline", "event": "epoch_end", "data": {"trial_index": 2, "epoch": 3, "total_epochs": 10}},
            {"phase": "search", "event": "trial_completed", "trial_index": 2, "trial_count": 6, "trial_label": "larger_model"},
        ]:
            if progress_callback is not None:
                progress_callback(payload)
        return {"status": "ok"}

    monkeypatch.setattr(gui_backend, "quick_hyperparameter_search", fake_quick_hyperparameter_search)

    result = gui_backend.run_search(gui_backend.SearchConfig(), progress_callback=forwarded.append)

    assert result["status"] == "ok"
    forwarded_events = [payload["event"] for payload in forwarded]
    assert "checkpoint_updated" not in forwarded_events
    assert forwarded_events.count("epoch_end") == 3
    assert any(payload["event"] == "epoch_end" and payload["epoch"] == 1 for payload in forwarded)
    assert any(payload["event"] == "epoch_end" and payload["epoch"] == 5 for payload in forwarded)
    assert any(payload["event"] == "epoch_end" and payload["epoch"] == 10 for payload in forwarded)
    assert not any(payload["event"] == "epoch_end" and payload["epoch"] == 2 for payload in forwarded)
