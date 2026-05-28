"""GUI interaction tests for the Surrogate Model Traning Suite."""

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
    return window


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
    assert gui_window._build_transfer_config().device == "cuda:1"


def _configure_dataset_paths(window, synthetic_dataset: dict[str, Path], tmp_path: Path) -> None:
    window.input_feature_path_edit.setText(str(synthetic_dataset["input_file"]))
    window.ground_truth_data_folder_path_edit.setText(str(synthetic_dataset["output_dir"]))
    window.model_output_folder_path_edit.setText(str(tmp_path / "gui_runs"))
    window.run_name_edit.setText("synthetic_gui")
    window._set_cache_path_value(str(synthetic_dataset["cache_path"]), manually_selected=True)


def test_window_scan_and_suggest_populate_preview_and_forms(
    gui_window,
    synthetic_dataset: dict[str, Path],
    tmp_path: Path,
) -> None:
    _configure_dataset_paths(gui_window, synthetic_dataset, tmp_path)

    gui_window.scan_dataset()
    assert gui_window.dataset_schema_status_badge.text() == "Valid"
    assert gui_window.detected_dataset_readme_value.text().endswith("README.md")
    assert gui_window.data_preview_table.rowCount() >= 10
    assert gui_window.scan_data_button.text() == "Scan Data"
    assert "[Scan] Loaded ground-truth sample" in gui_window.run_log_text_edit.toPlainText()
    assert gui_window.metric_cards["current_phase"].value_label.text() == "Scan"

    gui_window.run_suggest_initial_settings()
    assert gui_window.initial_suggestion_confidence_badge.text() in {"High", "Medium", "Low"}
    assert gui_window.initial_settings_table.rowCount() >= 10

    gui_window.apply_suggested_settings()
    baseline_config = gui_window.last_suggest_result["suggested_baseline_config"]
    transfer_config = gui_window.last_suggest_result["suggested_transfer_config"]
    assert gui_window.baseline_latent_dim_spin_box.value() == baseline_config["latent_dim"]
    assert gui_window.transfer_num_bands_spin_box.value() == transfer_config["num_bands"]


def test_data_source_browse_buttons_remain_visible(gui_window) -> None:
    assert gui_window.left_pane_container.minimumWidth() >= 560
    assert gui_window.browse_input_feature_button.width() >= 90
    assert gui_window.browse_ground_truth_data_folder_button.width() >= 90
    assert gui_window.browse_model_output_folder_button.width() >= 90


def test_window_exposes_separate_baseline_and_transfer_actions(gui_window) -> None:
    assert gui_window.start_baseline_button.text() == "Start Baseline Training"
    assert gui_window.start_transfer_button.text() == "Start Self-Transfer Learning"
    assert [gui_window.monitor_tabs.tabText(index) for index in range(gui_window.monitor_tabs.count())] == [
        "Baseline Monitor",
        "Transfer Results",
    ]
    transfer_tab = gui_window.monitor_tabs.widget(1)
    assert transfer_tab.layout().count() == 1
    assert not hasattr(gui_window, "transfer_training_loss_plot")
    assert not hasattr(gui_window, "transfer_average_mae_plot")
    assert not hasattr(gui_window, "transfer_band_mae_plot")


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

    assert gui_window.cache_path_edit.text().endswith("GUI_test_2.npz")


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

    assert gui_window.cache_path_edit.text() == str((tmp_path / "new_output" / "cache" / "GUI_test_1.npz").resolve())


def test_manual_cache_path_is_preserved_on_run_name_change(gui_window, tmp_path: Path) -> None:
    manual_cache_path = str(tmp_path / "manual_cache.npz")
    gui_window._set_cache_path_value(manual_cache_path, manually_selected=True)

    gui_window.run_name_edit.setText("GUI_test_2")

    assert gui_window.cache_path_edit.text() == manual_cache_path


def test_window_existing_transfer_base_updates_compatibility_status(
    gui_window,
    synthetic_dataset: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_dataset_paths(gui_window, synthetic_dataset, tmp_path)
    gui_window.scan_dataset()

    monkeypatch.setattr(
        gui_window_module,
        "check_transfer_compatibility",
        lambda **kwargs: {"status": "Compatible", "message": "Compatible synthetic baseline."},
    )

    gui_window.transfer_base_model_source_combo_box.setCurrentIndex(1)
    gui_window.transfer_base_run_path_edit.setText(str(tmp_path / "existing_baseline"))
    gui_window.validate_transfer_compatibility()

    assert gui_window.transfer_compatibility_status_badge.text() == "Compatible"
    assert "Compatible synthetic baseline" in gui_window.transfer_notes_label.text()


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
        transfer_base_run_dir,
        progress_callback=None,
        should_stop=None,
    ):
        assert baseline_config is not None
        assert transfer_config is None
        assert transfer_base_run_dir is None
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
                "runtime_seconds": 2.0,
            },
            "transfer": None,
        }

    monkeypatch.setattr(gui_window_module, "run_training_workflow", fake_run_training_workflow)
    gui_window.start_baseline_training()

    assert gui_window.last_workflow_summary["status"] == "ok"
    assert gui_window.run_state_badge.text() == "Completed"
    assert gui_window._baseline_epochs == [1, 2]
    assert gui_window.metric_cards["average_mae"].value_label.text() == "0.050000"
    assert gui_window.last_baseline_summary["run_dir"].endswith("baseline_run")


def test_window_transfer_training_uses_latest_baseline_from_session(
    gui_window,
    synthetic_dataset: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_dataset_paths(gui_window, synthetic_dataset, tmp_path)
    gui_window.scan_dataset()
    gui_window.last_baseline_summary = {"run_dir": str(tmp_path / "baseline_run")}
    gui_window.transfer_base_model_source_combo_box.setCurrentIndex(0)
    gui_window.transfer_num_bands_spin_box.setValue(3)
    gui_window.validate_transfer_compatibility()

    def fake_run_training_workflow(
        *,
        baseline_config,
        transfer_config,
        transfer_base_run_dir,
        progress_callback=None,
        should_stop=None,
    ):
        assert baseline_config is None
        assert transfer_config is not None
        assert transfer_base_run_dir == str(tmp_path / "baseline_run")
        emit_progress(
            progress_callback,
            event="baseline_metrics_ready",
            phase="transfer",
            base_average_mae=0.05,
            base_frequency_mae=[0.05, 0.04, 0.03],
            base_band_mae=[0.05, 0.04, 0.03],
            frequency_ghz=[2.0, 3.0, 4.0],
            message="Transfer baseline metrics ready.",
        )
        emit_progress(
            progress_callback,
            event="band_started",
            phase="transfer",
            transfer_iteration=1,
            direction="forward",
            band_index=1,
            num_bands=3,
            band_run_index=2,
            total_band_runs=5,
            message="Band training started.",
        )
        emit_progress(
            progress_callback,
            event="band_epoch_end",
            phase="transfer",
            transfer_iteration=1,
            direction="forward",
            band_index=1,
            num_bands=3,
            band_run_index=2,
            total_band_runs=5,
            epoch=1,
            total_epochs=2,
            train_loss=0.08,
            elapsed_seconds=5.0,
            eta_seconds=7.0,
            message="Band epoch 1 complete.",
        )
        emit_progress(
            progress_callback,
            event="iteration_completed",
            phase="transfer",
            transfer_iteration=1,
            total_iterations=1,
            average_mae=0.025,
            frequency_ghz=[2.0, 3.0, 4.0],
            frequency_mae=[0.03, 0.02, 0.01],
            band_mae=[0.03, 0.02, 0.01],
            elapsed_seconds=9.0,
            eta_seconds=0.0,
            message="Iteration completed.",
        )
        emit_progress(
            progress_callback,
            event="completed",
            phase="transfer",
            final_average_mae=0.025,
            elapsed_seconds=9.0,
            eta_seconds=0.0,
            message="Transfer complete.",
        )
        return {
            "status": "ok",
            "baseline": None,
            "transfer": {"run_dir": str(tmp_path / "transfer_run")},
        }

    monkeypatch.setattr(gui_window_module, "run_training_workflow", fake_run_training_workflow)
    gui_window.start_transfer_learning()

    assert gui_window.last_workflow_summary["status"] == "ok"
    assert gui_window.run_state_badge.text() == "Completed"
    assert gui_window._transfer_iteration_mae_x == [1]
    assert gui_window.metric_cards["elapsed"].value_label.text() == "9s"
    assert gui_window.metric_cards["eta"].value_label.text() == "0s"
