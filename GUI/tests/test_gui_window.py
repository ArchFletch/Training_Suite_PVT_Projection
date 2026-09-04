"""GUI interaction tests for the Surrogate Model Training Suite."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox

import xfmr_v2.gui_window as gui_window_module
from xfmr_v2.progress import emit_progress
from xfmr_v2.gui_workers import ImmediateTaskExecutor


@pytest.fixture
def gui_window(qtbot, monkeypatch: pytest.MonkeyPatch):
    def fake_list_available_devices():
        return [
            {"id": "cuda:0", "label": "cuda:0 — Synthetic GPU (16.0 GB)"},
            {"id": "cuda:1", "label": "cuda:1 — Synthetic GPU (16.0 GB)"},
            {"id": "cpu", "label": "cpu — CPU"},
        ]

    monkeypatch.setattr(gui_window_module, "list_available_devices", fake_list_available_devices)
    monkeypatch.setattr(gui_window_module, "load_last_session", lambda: None)
    monkeypatch.setattr(gui_window_module, "save_last_session", lambda payload: None)
    monkeypatch.setattr(gui_window_module.QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    window = gui_window_module.MlpTrainingStudio(executor=ImmediateTaskExecutor())
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(50)
    _grant_test_seat(window)
    return window


def _grant_test_seat(window) -> None:
    """Put the window in a licensed state so run-gated tests can reach their subject.

    Licensing fails closed, so without a seat every start_* call stops at the
    licence warning and tests of validation, training and export would silently
    assert against the wrong message. This drives the real gate rather than
    stubbing it, so a regression in _license_allows_new_runs still shows up here;
    the gate's own behaviour is covered in tests/gui_license.
    """
    from dataclasses import replace

    server_url = "http://license-test:27850"
    window.license_server_url_edit.setText(server_url)
    window.license_lease_state = replace(
        window.license_lease_state,
        phase="checked_out",
        badge_text="Checked Out",
        server_url=server_url,
        lease_id="lease_test",
    )


def test_window_auto_detects_devices_and_selects_first(gui_window) -> None:
    combo = gui_window.training_device_combo_box
    assert [combo.itemData(i) for i in range(combo.count())] == ["cuda:0", "cuda:1", "cpu"]
    # First CUDA device is selected by default, matching prior auto-select behavior.
    assert gui_window._current_device_id() == "cuda:0"


def test_window_selected_device_flows_into_configs(gui_window, synthetic_dataset, tmp_path) -> None:
    _configure_dataset_paths(gui_window, synthetic_dataset, tmp_path)
    gui_window._set_device_selection("cuda:1")
    assert gui_window._current_device_id() == "cuda:1"
    assert gui_window._build_baseline_train_config().device == "cuda:1"


def _configure_dataset_paths(window, synthetic_dataset: dict[str, Path], tmp_path: Path) -> None:
    # The dataset is one .npz file. Selecting it rewrites the run name, so set the
    # run name afterwards.
    window.dataset_file_edit.setText(str(synthetic_dataset["cache_path"]))
    window.model_output_folder_path_edit.setText(str(tmp_path / "gui_runs"))
    window.run_name_edit.setText("synthetic_gui")


def test_window_scan_and_suggest_populate_preview_and_forms(
    gui_window,
    synthetic_dataset: dict[str, Path],
    tmp_path: Path,
) -> None:
    _configure_dataset_paths(gui_window, synthetic_dataset, tmp_path)

    gui_window.scan_dataset()
    assert gui_window.dataset_schema_status_badge.text() == "Valid"
    assert gui_window.data_preview_table.rowCount() >= 10
    assert gui_window.scan_dataset_button.text() == "Scan Dataset"
    assert "[Scan] Dataset scan completed" in gui_window.run_log_text_edit.toPlainText()
    assert gui_window.metric_cards["current_phase"].value_label.text() == "Scan"

    gui_window.run_suggest_initial_settings()
    assert gui_window.initial_suggestion_confidence_badge.text() in {"High", "Medium", "Low"}
    # Baseline rows only. The engine still returns transfer suggestions, but the
    # GUI cannot run that workflow, so listing them would offer settings with no
    # control to apply them to.
    table = gui_window.initial_settings_table
    assert table.rowCount() >= 7
    shown = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert not [name for name in shown if name.startswith("Transfer:")], shown

    gui_window.apply_suggested_settings()
    baseline_config = gui_window.last_suggest_result["suggested_baseline_config"]
    assert gui_window.baseline_width_spin_box.value() == baseline_config["width"]


def test_data_source_browse_buttons_remain_visible(gui_window) -> None:
    assert gui_window.left_pane_container.minimumWidth() >= 560
    assert gui_window.browse_dataset_file_button.width() >= 90
    assert gui_window.browse_model_output_folder_button.width() >= 90


def test_data_sources_card_offers_only_the_npz_dataset_file(gui_window) -> None:
    """The folder picker and the Advanced overrides are gone, not just hidden.

    Leaving a disabled widget behind would keep the old two-ways-in ambiguity in
    saved sessions and in the config payload.
    """
    for removed in (
        "dataset_folder_edit",
        "input_feature_path_edit",
        "ground_truth_data_folder_path_edit",
        "cache_path_edit",
        "advanced_paths_section",
    ):
        assert not hasattr(gui_window, removed), f"{removed} should be gone"

    assert gui_window.dataset_file_edit.text() == ""
    assert "npz" in gui_window.dataset_file_edit.placeholderText()


def test_window_exposes_only_the_baseline_run_action(gui_window) -> None:
    """Frequency-domain self-transfer is gone from the GUI.

    Removed rather than hidden: a leftover button or monitor tab would still
    round-trip through the saved session and still offer a workflow the GUI no
    longer runs.
    """
    assert gui_window.start_baseline_button.text() == "Start Baseline Training"
    assert not hasattr(gui_window, "start_transfer_button")
    assert not hasattr(gui_window, "transfer_average_mae_plot")
    assert not hasattr(gui_window, "enable_transfer_learning_checkbox")
    assert not hasattr(gui_window, "transfer_num_bands_spin_box")

    assert [gui_window.monitor_tabs.tabText(i) for i in range(gui_window.monitor_tabs.count())] == [
        "Baseline Monitor",
        "Test Samples",
    ]
    assert [
        gui_window.training_tabs.tabText(i) for i in range(gui_window.training_tabs.count())
    ] == ["Baseline Training"]


def test_auto_managed_cache_path_tracks_run_name_changes(gui_window) -> None:
    output_dir = str(Path("artifacts/output_for_test").resolve())
    initial_cache_path = str((Path(output_dir) / "cache" / "GUI_test_1.npz").resolve())
    gui_window.apply_config_payload(
        {
            "data_sources": {
                "output_dir": output_dir,
                "run_name": "GUI_test_1",
                "cache_path": initial_cache_path,
            }
        }
    )

    gui_window.run_name_edit.setText("GUI_test_2")

    assert gui_window._cache_path_value.endswith("GUI_test_2.npz")


def test_auto_managed_cache_path_tracks_output_folder_changes(gui_window, tmp_path: Path) -> None:
    gui_window.apply_config_payload(
        {
            "data_sources": {
                "output_dir": str(tmp_path / "initial_output"),
                "run_name": "GUI_test_1",
            }
        }
    )

    gui_window.model_output_folder_path_edit.setText(str(tmp_path / "new_output"))

    assert gui_window._cache_path_value == str((tmp_path / "new_output" / "cache" / "GUI_test_1.npz").resolve())


def test_a_saved_manual_cache_path_is_reported_not_restored(gui_window, tmp_path: Path) -> None:
    """Choosing a cache file was removed; the cache always follows the run name.

    An old session that pinned one must not silently keep training against it,
    so the restore says what happened and points at the CLI flag.
    """
    manual_cache_path = str(tmp_path / "manual_cache.npz")
    gui_window.apply_config_payload(
        {
            "data_sources": {
                "output_dir": str(tmp_path / "out"),
                "run_name": "GUI_test_1",
                "cache_path": manual_cache_path,
                "cache_path_manually_selected": True,
            }
        }
    )

    log = gui_window.run_log_text_edit.toPlainText()
    assert manual_cache_path in log
    assert "--cache-path" in log
    assert gui_window._cache_path_value == str((tmp_path / "out" / "cache" / "GUI_test_1.npz").resolve())

    gui_window.run_name_edit.setText("GUI_test_2")
    assert gui_window._cache_path_value.endswith("GUI_test_2.npz")


def test_window_baseline_training_progress_updates_live_metrics_and_plots(
    gui_window,
    synthetic_dataset: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_dataset_paths(gui_window, synthetic_dataset, tmp_path)
    gui_window.scan_dataset()

    def fake_run_training_workflow(
        *,
        baseline_config,
        transfer_config,
        progress_callback=None,
        should_stop=None,
    ):
        assert baseline_config is not None
        assert transfer_config is None
        emit_progress(progress_callback, event="started", phase="baseline", message="Baseline started.")
        emit_progress(
            progress_callback,
            event="data_ready",
            phase="baseline",
            train_samples=6,
            val_samples=2,
            test_samples=2,
            message="Baseline data ready.",
        )
        emit_progress(
            progress_callback,
            event="epoch_end",
            phase="baseline",
            epoch=1,
            total_epochs=2,
            train_loss=0.30,
            val_loss=0.20,
            best_val_loss=0.20,
            best_epoch=1,
            elapsed_seconds=1.0,
            eta_seconds=1.0,
            message="Epoch 1 complete.",
        )
        emit_progress(
            progress_callback,
            event="epoch_end",
            phase="baseline",
            epoch=2,
            total_epochs=2,
            train_loss=0.15,
            val_loss=0.12,
            best_val_loss=0.12,
            best_epoch=2,
            elapsed_seconds=2.0,
            eta_seconds=0.0,
            message="Epoch 2 complete.",
        )
        emit_progress(
            progress_callback,
            event="evaluation_completed",
            phase="baseline",
            average_evaluation_mae=0.05,
            # Per-channel values are still emitted, but the card must ignore them and
            # show only the single average over all ground-truth channels.
            per_channel_mae=[0.06, 0.04],
            channel_mae_with_units=["gain: 0.0600 dB", "phase: 0.0400 deg"],
            frequency_ghz=[2.0, 3.0, 4.0],
            frequency_mae=[0.05, 0.04, 0.03],
            message="Baseline evaluation complete.",
        )
        emit_progress(progress_callback, event="completed", phase="baseline", message="Baseline complete.")
        return {
            "status": "ok",
            "baseline": {
                "run_dir": str(tmp_path / "baseline_run"),
                "history": [
                    {"epoch": 1, "train_loss": 0.30, "val_loss": 0.20},
                    {"epoch": 2, "train_loss": 0.15, "val_loss": 0.12},
                ],
                "best_val_loss": 0.12,
                "average_evaluation_mae": 0.05,
                "channel_mae_with_units": ["gain: 0.0600 dB", "phase: 0.0400 deg"],
                "runtime_seconds": 2.0,
            },
            "transfer": None,
        }

    monkeypatch.setattr(gui_window_module, "run_training_workflow", fake_run_training_workflow)
    gui_window.start_baseline_training()

    assert gui_window.last_workflow_summary["status"] == "ok"
    assert gui_window.run_state_badge.text() == "Completed"
    assert gui_window._baseline_epochs == [1, 2]
    # The card shows the single MAE averaged over all channels, not the per-channel
    # breakdown that was also emitted (in both the live event and the summary).
    assert gui_window.metric_cards["average_mae"].value_label.text() == "0.050000"
    assert gui_window.last_baseline_summary["run_dir"].endswith("baseline_run")
    # Elapsed shows the real runtime (runtime_seconds=2.0) once baseline training completes.
    assert gui_window.metric_cards["elapsed"].value_label.text() == "2s"


def _write_minimal_baseline_checkpoint(run_dir: Path) -> None:
    """Write a tiny but valid best_model.pt the ONNX exporter can consume."""
    import numpy as np
    import torch

    from xfmr_v2.runner import build_model

    run_dir.mkdir(parents=True, exist_ok=True)
    n_features, channels, freqs, width, depth = 4, 2, 8, 16, 3
    model = build_model(
        "SpectraHydra",
        num_frequencies=freqs,
        input_feature_dim=n_features,
        ground_truth_channels=channels,
        width=width,
        depth=depth,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {"model_type": "SpectraHydra", "width": width, "depth": depth, "cache_path": ""},
            "active_input_feature_names": [f"f{i}" for i in range(n_features)],
            "target_channel_names": ["gain", "phase"],
            "input_feature_mean": np.zeros(n_features, dtype=np.float32),
            "input_feature_std": np.ones(n_features, dtype=np.float32),
            "target_mean": np.zeros((channels, freqs), dtype=np.float32),
            "target_std": np.ones((channels, freqs), dtype=np.float32),
        },
        run_dir / "best_model.pt",
    )


def test_window_export_baseline_to_onnx(gui_window, tmp_path, monkeypatch) -> None:
    pytest.importorskip("onnx")
    run_dir = tmp_path / "baseline_run"
    _write_minimal_baseline_checkpoint(run_dir)
    gui_window.last_baseline_summary = {"run_dir": str(run_dir)}

    out_path = tmp_path / "exported.onnx"
    monkeypatch.setattr(
        gui_window_module.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(out_path), "ONNX Files (*.onnx)"),
    )

    gui_window.export_baseline_to_onnx()

    assert out_path.exists()
    assert out_path.with_suffix(".meta.json").exists()
    assert gui_window.last_onnx_export_path == str(out_path)
    assert "Exported ONNX model to" in gui_window.run_log_text_edit.toPlainText()
    # The transient "Exporting" badge must settle back to Idle when the task finishes.
    assert gui_window.run_state_badge.text() == "Idle"


def test_window_export_to_onnx_without_run_warns(gui_window, monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        gui_window_module.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args[-1]) or QMessageBox.StandardButton.Ok,
    )
    dialog_calls: list[bool] = []
    monkeypatch.setattr(
        gui_window_module.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: dialog_calls.append(True) or ("", ""),
    )
    gui_window.last_baseline_summary = None
    gui_window.last_workflow_summary = None

    gui_window.export_baseline_to_onnx()

    assert warnings, "expected a warning when no baseline run exists"
    assert not dialog_calls, "save dialog should not open without a run"


def test_window_export_to_onnx_handles_backend_error(gui_window, tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "baseline_run"
    _write_minimal_baseline_checkpoint(run_dir)
    gui_window.last_baseline_summary = {"run_dir": str(run_dir)}

    monkeypatch.setattr(
        gui_window_module.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(tmp_path / "out.onnx"), "ONNX Files (*.onnx)"),
    )

    def boom(**kwargs):
        raise RuntimeError("simulated export failure")

    monkeypatch.setattr(gui_window_module, "export_model_to_onnx", boom)

    gui_window.export_baseline_to_onnx()

    # The task-error path must surface the failure and leave the UI usable
    # (controls re-enabled, no task left running), not crash or hang.
    assert gui_window.run_state_badge.text() == "Error"
    assert gui_window.current_task is None
    assert gui_window.export_onnx_button.isEnabled()
    assert "Error: simulated export failure" in gui_window.run_log_text_edit.toPlainText()


def _check_projection_column(window, name: str) -> None:
    from PySide6.QtCore import Qt

    widget = window.baseline_projection_columns_list
    for index in range(widget.count()):
        if widget.item(index).text() == name:
            widget.item(index).setCheckState(Qt.CheckState.Checked)
            return
    raise AssertionError(f"column {name!r} not in picker")


PROJECTION_ROW_CAPTIONS = ("PVT Corner Columns", "Corner Projection Width")


def _row_is_hidden(window, caption: str) -> bool:
    """Whether a baseline form row is explicitly hidden, label included.

    isHidden() rather than isVisible(): the baseline tab may not be the current
    tab, which would make every row read as invisible regardless of this
    setting.
    """
    label, widget = window._baseline_form_rows[caption]
    assert label.isHidden() == widget.isHidden(), (
        f"{caption}: label and widget disagree, so a caption is left stranded"
    )
    return widget.isHidden()


def test_projection_controls_gate_on_model_type(gui_window) -> None:
    combo = gui_window.baseline_model_type_combo_box
    assert combo.findText("SpectraHydraProj") >= 0

    # Default SpectraNet: projection controls are hidden, not merely disabled.
    assert combo.currentText() == "SpectraNet"
    for caption in PROJECTION_ROW_CAPTIONS:
        assert _row_is_hidden(gui_window, caption)
    assert not gui_window.baseline_projection_columns_list.isEnabled()
    assert not gui_window.baseline_projection_dim_spin_box.isEnabled()

    combo.setCurrentText("SpectraHydraProj")
    for caption in PROJECTION_ROW_CAPTIONS:
        assert not _row_is_hidden(gui_window, caption)
    assert gui_window.baseline_projection_columns_list.isEnabled()
    assert gui_window.baseline_projection_dim_spin_box.isEnabled()

    combo.setCurrentText("SpectraHydra")
    assert _row_is_hidden(gui_window, "PVT Corner Columns")
    assert not gui_window.baseline_projection_columns_list.isEnabled()

    # SpectraTrunk is selectable and, like the non-projection models, hides the
    # corner-projection rows.
    assert combo.findText("SpectraTrunk") >= 0
    combo.setCurrentText("SpectraTrunk")
    for caption in PROJECTION_ROW_CAPTIONS:
        assert _row_is_hidden(gui_window, caption)
    assert not gui_window.baseline_projection_columns_list.isEnabled()
    assert not gui_window.baseline_projection_dim_spin_box.isEnabled()


def test_settings_shared_by_every_model_are_never_hidden(gui_window) -> None:
    """Only genuinely model-specific rows disappear.

    Hiding a shared setting would be worse than the problem being fixed: the
    control still governs the run, it just cannot be reached.
    """
    shared = [
        caption
        for caption in gui_window._baseline_form_rows
        if caption not in PROJECTION_ROW_CAPTIONS
    ]
    assert "Model Type" in shared and "Network Width" in shared and "Random Seed" in shared

    for model_type in gui_window_module.MODEL_TYPES:
        gui_window.baseline_model_type_combo_box.setCurrentText(model_type)
        for caption in shared:
            assert not _row_is_hidden(gui_window, caption), f"{caption} hidden for {model_type}"


def test_every_model_type_shows_the_rows_the_engine_says_it_reads(gui_window) -> None:
    """The form's visibility rule is the engine's, not a second copy of it.

    A model added to runner.PROJECTION_MODEL_TYPES must light up the corner rows
    without a matching GUI edit, or the two definitions drift.
    """
    from xfmr_v2.runner import PROJECTION_MODEL_TYPES, uses_corner_projection

    assert "SpectraHydraProj" in PROJECTION_MODEL_TYPES
    for model_type in gui_window_module.MODEL_TYPES:
        gui_window.baseline_model_type_combo_box.setCurrentText(model_type)
        expected_visible = uses_corner_projection(model_type)
        for caption in PROJECTION_ROW_CAPTIONS:
            assert _row_is_hidden(gui_window, caption) is not expected_visible, (
                f"{caption} visibility disagrees with the engine for {model_type}"
            )


def test_projection_columns_flow_from_scan_into_configs(
    gui_window, synthetic_dataset, tmp_path
) -> None:
    _configure_dataset_paths(gui_window, synthetic_dataset, tmp_path)
    gui_window.scan_dataset()

    # The picker lists the scanned dataset's active feature columns ('const' is
    # dropped as constant and therefore not offered).
    widget = gui_window.baseline_projection_columns_list
    assert [widget.item(i).text() for i in range(widget.count())] == ["x", "y"]

    gui_window.baseline_model_type_combo_box.setCurrentText("SpectraHydraProj")
    gui_window.baseline_projection_dim_spin_box.setValue(4)
    _check_projection_column(gui_window, "y")

    baseline_config = gui_window._build_baseline_train_config()
    assert baseline_config.model_type == "SpectraHydraProj"
    assert baseline_config.projection_columns == ["y"]
    assert baseline_config.projection_dim == 4



def test_projection_start_blocked_without_corner_columns(
    gui_window, synthetic_dataset, tmp_path, monkeypatch
) -> None:
    _configure_dataset_paths(gui_window, synthetic_dataset, tmp_path)
    gui_window.scan_dataset()
    gui_window.baseline_model_type_combo_box.setCurrentText("SpectraHydraProj")

    launched: list[dict] = []
    monkeypatch.setattr(
        gui_window_module,
        "run_training_workflow",
        lambda **kwargs: launched.append(kwargs) or {"status": "ok", "baseline": None, "transfer": None},
    )
    warnings: list[str] = []
    monkeypatch.setattr(gui_window, "_show_warning", lambda message: warnings.append(message))

    gui_window.start_baseline_training()
    assert not launched, "training must not start without corner columns selected"
    assert any("corner column" in message for message in warnings)

    # Selecting a corner column unblocks the run.
    _check_projection_column(gui_window, "y")
    gui_window.start_baseline_training()
    assert len(launched) == 1
    assert launched[0]["baseline_config"].projection_columns == ["y"]


def test_projection_form_roundtrip_and_suggestion_keeps_selection(gui_window) -> None:
    # Session restore happens before any scan: the restored names become visible
    # picker items and survive into the collected form payload.
    gui_window._apply_baseline_form(
        {
            "model_type": "SpectraHydraProj",
            "projection_columns": ["Temp_C", "VDD"],
            "projection_dim": 8,
        }
    )
    assert gui_window.baseline_model_type_combo_box.currentText() == "SpectraHydraProj"
    form = gui_window._collect_baseline_form()
    assert form["projection_columns"] == ["Temp_C", "VDD"]
    assert form["projection_dim"] == 8

    # Configs from suggest/search are built via asdict(TrainConfig(...)) and carry
    # the dataclass DEFAULTS (projection_columns=None, projection_dim=16) for the
    # non-projection model types they recommend. Applying one must not clear the
    # corner selection or reset the projection width (regression: the dim used to
    # snap back to 16 because 16 is not None).
    gui_window._apply_baseline_form(
        {"model_type": "SpectraHydra", "width": 128, "projection_columns": None, "projection_dim": 16}
    )
    form = gui_window._collect_baseline_form()
    assert form["projection_columns"] == ["Temp_C", "VDD"]
    assert form["projection_dim"] == 8


def test_scan_split_warnings_reach_the_run_log(gui_window, monkeypatch) -> None:
    """warnings.warn goes to stderr, which a packaged GUI never shows — and the
    crossing warning is the only signal for splintering that auto-fold cannot
    repair, so the scan must put it in the run log."""
    monkeypatch.setattr(
        gui_window_module,
        "build_and_scan_dataset",
        lambda **kwargs: {
            "status": "ok",
            "split_warnings": ["no design appears at more than 5 of them"],
            "preview_rows": [],
            "active_input_feature_names": ["x"],
            "dropped_input_feature_names": [],
            "dataset_name": "d",
            "schema_status": "Valid",
            "schema": {},
            "cache_summary": {"num_samples": 10, "status": "existing"},
            "cache_path": "c.npz",
            "frequency_count": 3,
            "frequency_range_ghz": [1.0, 3.0],
            "sweep_label": "Frequency (GHz)",
        },
    )
    gui_window._on_scan_completed(gui_window_module.build_and_scan_dataset())
    assert "no design appears at more than 5" in gui_window.run_log_text_edit.toPlainText()


def test_dataset_file_survives_a_session_roundtrip(gui_window, tmp_path) -> None:
    """The .npz file is the only data source. When it was not persisted, a
    restarted GUI forgot the dataset and kept only a stale cache path -- which is
    how a run ended up pointed at a leftover pytest directory."""
    dataset = tmp_path / "my_dataset.npz"
    dataset.write_bytes(b"")
    gui_window.dataset_file_edit.setText(str(dataset))

    payload = gui_window.collect_config_payload()
    assert payload["data_sources"]["dataset_file"] == str(dataset)

    gui_window.dataset_file_edit.setText("")
    gui_window.apply_config_payload(payload)
    assert gui_window.dataset_file_edit.text() == str(dataset)


def test_an_old_sessions_dataset_folder_is_reported_not_silently_dropped(gui_window, tmp_path) -> None:
    """Restoring must not look like the old dataset loaded when nothing did."""
    folder = tmp_path / "my_dataset"
    folder.mkdir()
    gui_window.apply_config_payload(
        {
            "data_sources": {
                "dataset_folder": str(folder),
                "input_feature_path": str(folder / "log.txt"),
                "output_dir": str(tmp_path / "out"),
                "run_name": "legacy_run",
            }
        }
    )

    log = gui_window.run_log_text_edit.toPlainText()
    assert str(folder) in log
    assert "--data-root" in log
    assert "--input-feature-path" in log
    assert gui_window.dataset_file_edit.text() == ""


def test_tests_never_write_the_real_gui_state(isolated_app_state) -> None:
    """Guard the guard: the autouse fixture must redirect every writable location
    away from <repo>/GUI/artifacts, or a test can clobber the developer's session."""
    from xfmr_v2 import gui_backend
    from xfmr_v2.licensing import client_config

    repo_artifacts = Path(gui_backend.__file__).resolve().parent.parent / "artifacts"
    gui_backend.save_last_session({"marker": "from-test"})

    written = gui_backend.current_runtime_paths().last_session_path
    assert written == isolated_app_state.last_session_path
    assert written.exists() and repo_artifacts not in written.parents
    assert repo_artifacts not in client_config.license_client_config_path().parents
    assert gui_backend.load_last_session()["marker"] == "from-test"


def test_per_channel_mae_cards_appear_from_the_run(gui_window) -> None:
    """The per-channel MAE is the number with physical units, so it must be visible
    without any interaction. The averaged card mixes dB/deg/decades and is titled
    accordingly so it is not mistaken for a comparable per-channel figure."""
    gui_window._update_baseline_progress(
        {
            "event": "evaluation_completed",
            "frequency_ghz": [1.0, 2.0],
            "frequency_mae": [0.1, 0.2],
            "average_evaluation_mae": 0.366261,
            "channel_mae_with_units": [
                "gain: 0.2671 dB", "phase: 0.8179 deg", "noise: 0.0138 decades",
            ],
        },
    )
    cards = gui_window._channel_metric_cards
    assert list(cards) == ["gain", "phase", "noise"]
    assert cards["gain"].value_label.text() == "0.2671 dB"
    assert cards["phase"].value_label.text() == "0.8179 deg"
    assert gui_window.metric_cards["average_mae"].value_label.text() == "0.366261"

    # A new run clears stale values rather than leaving the last run's numbers up.
    gui_window._reset_baseline_plots()
    assert cards["gain"].value_label.text() == "-"


def test_design_split_keys_in_old_configs_log_a_note_and_do_not_break(gui_window) -> None:
    """The design-level split was removed from the GUI (CLI-only now). Sessions and
    configs written before that still carry the keys; loading one must not fail,
    must not silently pretend the split will happen, and must not put the columns
    into the run config."""
    gui_window._apply_baseline_form(
        {"split_design_columns": ["CS_fF", "LD_pH"], "split_corner_columns": ["Temp_C"]}
    )
    log_text = gui_window.run_log_text_edit.toPlainText()
    assert "removed from the GUI" in log_text
    assert "CS_fF" in log_text and "Temp_C" in log_text
    assert "--split-design-columns" in log_text

    assert gui_window._build_baseline_train_config().split_design_columns is None
    assert gui_window._build_baseline_train_config().split_corner_columns is None

    # TrainConfig-default payloads (None) and split-off payloads ([]) stay silent.
    gui_window.run_log_text_edit.clear()
    gui_window._apply_baseline_form({"split_design_columns": None, "split_corner_columns": []})
    assert "removed from the GUI" not in gui_window.run_log_text_edit.toPlainText()


def test_datasets_without_channel_units_keep_the_single_card(gui_window) -> None:
    """The runner omits channel_mae_with_units when a dataset declares no units
    (e.g. raw Touchstone), so no per-channel cards should be invented."""
    gui_window._update_baseline_progress(
        {"event": "evaluation_completed", "frequency_ghz": [1.0],
         "frequency_mae": [0.1], "average_evaluation_mae": 0.5},
    )
    assert gui_window._channel_metric_cards == {}
    assert gui_window.metric_cards["average_mae"].value_label.text() == "0.500000"


def test_applying_a_corner_projection_suggestion_fills_the_picker(gui_window, tmp_path) -> None:
    """Accepting a PVT suggestion must leave the baseline form ready to train.

    The recommender now picks SpectraHydraProj for corner datasets, and that
    model refuses to start without corner columns. A recommendation that flipped
    the combo box but left the boxes unchecked would hand the user a form that
    only fails at Start Training.
    """
    from tests.test_suggest_corner_projection import _pvt_features, _write_cache
    from xfmr_v2.suggest import SuggestConfig, suggest_initial_settings

    features, names = _pvt_features()
    cache_path = _write_cache(tmp_path / "pvt_cache.npz", features, names)
    result = suggest_initial_settings(SuggestConfig(data_root=None, cache_path=str(cache_path)))

    gui_window.last_scan_result = {
        "active_input_feature_names": result["active_input_feature_names"],
        "dropped_input_feature_names": result["dropped_input_feature_names"],
        "frequency_count": result["diagnostics"]["frequency_point_count"],
    }
    gui_window._populate_projection_columns(result["active_input_feature_names"])
    gui_window.last_suggest_result = result
    gui_window.apply_suggested_settings()

    assert gui_window.baseline_model_type_combo_box.currentText() == "SpectraHydraProj"
    assert gui_window._selected_projection_columns() == ["Temp_C", "VDD"]
    assert gui_window.baseline_projection_columns_list.isEnabled()
    assert gui_window._validate_projection_settings()


def _revoke_test_seat(window) -> None:
    """Put the window in the unlicensed state a fresh install starts in.

    Set explicitly rather than assumed: a lease cached by the licensing tests can
    survive into this fixture, which would silently make the assertions vacuous.
    """
    from dataclasses import replace

    window.license_server_url_edit.setText("")
    window.license_lease_state = replace(
        window.license_lease_state,
        phase="unconfigured",
        badge_text="Unconfigured",
        server_url="",
        lease_id=None,
    )
    window._refresh_run_button_availability()


def test_a_disabled_start_button_says_why_it_is_disabled(gui_window) -> None:
    """The dead-end this closes: licensing fails closed, so out of the box the
    start buttons are disabled. A disabled button never emits clicked, so the
    explanatory dialog inside start_baseline_training is unreachable and the
    user is left with a greyed control and nothing telling them what to fix."""
    _revoke_test_seat(gui_window)
    assert not gui_window.start_baseline_button.isEnabled()

    expected = gui_window._start_blocked_reason()
    assert expected == gui_window._license_display_message()
    reason = gui_window.start_baseline_button.toolTip()
    assert reason, "a disabled start button must carry its reason as a tooltip"
    assert reason == expected
    assert "Acquire Seat" in reason


def test_the_start_button_tooltip_clears_once_a_seat_is_held(gui_window) -> None:
    _grant_test_seat(gui_window)
    gui_window._refresh_run_button_availability()

    assert gui_window.start_baseline_button.isEnabled()
    assert gui_window.start_baseline_button.toolTip() == ""
    assert gui_window._start_blocked_reason() is None


def test_a_running_task_is_named_in_the_start_button_tooltip(gui_window) -> None:
    _grant_test_seat(gui_window)
    gui_window.current_task_name = "training"
    gui_window._set_action_controls_enabled(False)

    assert not gui_window.start_baseline_button.isEnabled()
    assert "training task is running" in gui_window.start_baseline_button.toolTip()

    gui_window._set_action_controls_enabled(True)
    assert gui_window.start_baseline_button.isEnabled()
    assert gui_window.start_baseline_button.toolTip() == ""


def test_changing_the_dataset_file_invalidates_the_previous_scan(gui_window, tmp_path) -> None:
    """A scan describes one dataset.

    The badge already reset to Not Scanned, but last_scan_result stayed, so a
    run could start against a dataset that was never scanned and corner columns
    would validate against the previous dataset's column names.
    """
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    for path in (first, second):
        path.write_bytes(b"")

    gui_window.dataset_file_edit.setText(str(first))
    gui_window.last_scan_result = {
        "active_input_feature_names": ["Temp_C", "VDD"],
        "dropped_input_feature_names": [],
        "frequency_count": 8,
    }
    gui_window.dataset_schema_status_badge.set_status("Valid")

    gui_window.dataset_file_edit.setText(str(second))

    assert gui_window.last_scan_result is None
    assert gui_window.dataset_schema_status_badge.text() == "Not Scanned"


def test_starting_without_a_scan_explains_itself_instead_of_running(gui_window, tmp_path) -> None:
    """The click-time prerequisites stay dialogs, so they are not silent."""
    dataset = tmp_path / "d.npz"
    dataset.write_bytes(b"")
    gui_window.dataset_file_edit.setText(str(dataset))
    gui_window.model_output_folder_path_edit.setText(str(tmp_path / "out"))
    gui_window.run_name_edit.setText("r")
    _grant_test_seat(gui_window)
    gui_window._refresh_run_button_availability()
    assert gui_window.start_baseline_button.isEnabled()

    started = []
    gui_window._start_task = lambda *a, **k: started.append(True)
    gui_window.start_baseline_training()

    assert not started, "training must not start before the dataset is scanned"


def test_a_saved_transfer_setup_is_reported_not_silently_dropped(gui_window, tmp_path) -> None:
    """A session tuned for per-band transfer must not look like it still applies.

    The run now trains one model across the whole frequency range, which is a
    different model from the one the saved band count describes.
    """
    gui_window.apply_config_payload(
        {
            "data_sources": {"output_dir": str(tmp_path / "out"), "run_name": "legacy"},
            "transfer": {"enabled": True, "num_bands": 10, "iterations": 10},
        }
    )

    log = gui_window.run_log_text_edit.toPlainText()
    assert "self-transfer" in log
    assert "10 frequency bands" in log
    assert "run_self_transfer.py" in log


def test_a_config_without_transfer_settings_stays_quiet(gui_window, tmp_path) -> None:
    gui_window.apply_config_payload(
        {"data_sources": {"output_dir": str(tmp_path / "out"), "run_name": "clean"}}
    )
    assert "self-transfer" not in gui_window.run_log_text_edit.toPlainText()
