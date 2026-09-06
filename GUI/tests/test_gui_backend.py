"""Tests for the GUI-facing backend helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from xfmr_v2 import atomic_json, data, gui_backend


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
            design_counts=None,
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


def _write_pvt_backend_cache(path, num_designs: int = 12) -> None:
    corners = [(temp, vdd) for temp in (-40.0, 27.0, 125.0) for vdd in (0.9, 1.0)]
    features = np.asarray(
        [
            [0.5 * design, 1.25 * design + 3.0, temp, vdd]
            for temp, vdd in corners
            for design in range(num_designs)
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(3)
    np.savez_compressed(
        path,
        features=features,
        targets=rng.standard_normal((len(features), 2, 4)).astype(np.float32),
        frequency_hz=np.linspace(1e9, 4e9, 4).astype(np.float32),
        input_feature_names=np.asarray(["geom_a", "geom_b", "Temp_C", "VDD"]),
        target_names=np.asarray(["gain", "phase"]),
        channel_names=np.asarray(["gain", "phase"]),
        channel_units=np.asarray(["dB", "deg"]),
        channel_transforms=np.asarray(["", ""]),
        sweep_label=np.asarray("Frequency (GHz)"),
    )


def test_scan_dataset_previews_design_level_split(tmp_path: Path) -> None:
    """The preview must show the split the run would use, not always row-level."""
    cache_path = tmp_path / "pvt_cache.npz"
    _write_pvt_backend_cache(cache_path)

    result = gui_backend.scan_dataset(
        cache_path=str(cache_path),
        train_frac=0.8,
        val_frac=0.1,
        seed=42,
        split_corner_columns=["Temp_C", "VDD"],
    )

    rows = dict(result["preview_rows"])
    # 12 designs x 6 corners -> 9/1/2 designs = 54/6/12 rows (row-level: 57/7/8).
    assert rows["Training Samples"] == "54"
    assert rows["Validation Samples"] == "6"
    assert rows["Test Samples"] == "12"
    assert rows["Split Mode"].startswith("Design-level: 9 / 1 / 2")


def test_scan_dataset_falls_back_to_row_level_on_bad_corner_columns(tmp_path: Path) -> None:
    """A stale corner selection must not break scanning (scanning is what
    populates the picker) — the preview drops to row-level and says why."""
    cache_path = tmp_path / "pvt_cache.npz"
    _write_pvt_backend_cache(cache_path)

    result = gui_backend.scan_dataset(
        cache_path=str(cache_path),
        train_frac=0.8,
        val_frac=0.1,
        seed=42,
        split_corner_columns=["Temp_K"],  # not a column of this dataset
    )

    rows = dict(result["preview_rows"])
    assert rows["Training Samples"] == "57"
    assert rows["Split Mode"].startswith("Row-level (requested design-level split unavailable")
    assert "Temp_K" in rows["Split Mode"]


def test_save_gui_config_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    target = tmp_path / "gui" / "last_session.json"

    gui_backend.save_gui_config(target, {"device": "cpu"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"device": "cpu"}
    assert [p.name for p in target.parent.iterdir()] == ["last_session.json"]


def test_a_failed_save_keeps_the_previous_session_file_intact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The save used to truncate the target before writing, so a crash or a
    full disk part-way through left a JSON fragment -- and the GUI would not
    start again until someone deleted it. The old file must survive a failed
    write untouched, with no fragment left beside it."""
    target = tmp_path / "gui" / "last_session.json"
    gui_backend.save_gui_config(target, {"device": "cpu", "run": 1})
    before = target.read_text(encoding="utf-8")

    def out_of_space(_fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(atomic_json.os, "fsync", out_of_space)
    with pytest.raises(OSError):
        gui_backend.save_gui_config(target, {"device": "cuda:0", "run": 2})

    assert target.read_text(encoding="utf-8") == before
    assert json.loads(before)["run"] == 1
    assert [p.name for p in target.parent.iterdir()] == ["last_session.json"]


def test_a_failure_at_the_final_rename_still_cleans_up_and_keeps_the_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The write completed but the swap did not (a locked target on Windows, a
    permission change in between): no temp may be left beside the old file."""
    target = tmp_path / "gui" / "last_session.json"
    gui_backend.save_gui_config(target, {"run": 1})

    def cannot_swap(_src: str, _dst: str) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(atomic_json.os, "replace", cannot_swap)
    with pytest.raises(PermissionError):
        gui_backend.save_gui_config(target, {"run": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"run": 1}
    assert [p.name for p in target.parent.iterdir()] == ["last_session.json"]


def test_an_unserializable_payload_touches_nothing_on_disk(tmp_path: Path) -> None:
    target = tmp_path / "gui" / "last_session.json"

    with pytest.raises(TypeError):
        gui_backend.save_gui_config(target, {"bad": object()})

    assert not target.parent.exists()

