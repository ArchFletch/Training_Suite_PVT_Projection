"""PySide6 desktop GUI for the MLP training workflow."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .app_paths import current_runtime_paths
from .gui_backend import (
    build_and_scan_dataset,
    build_suggest_result,
    check_transfer_compatibility,
    default_run_name,
    export_model_to_onnx,
    list_available_devices,
    load_last_session,
    make_run_roots,
    run_training_workflow,
    save_last_session,
)
from .gui_theme import APP_THEME, apply_application_theme, configure_plot_widget, plot_color_cycle, status_colors
from .gui_workers import QtTaskExecutor
from .licensing import (
    LicenseClientError,
    LicenseLeaseController,
    LicenseLeaseState,
    LicenseStatus,
    format_utc_timestamp,
    normalize_server_url,
    test_license_connection,
)
from .runner import LOSS_FUNCTIONS, MODEL_TYPES, SCHEDULER_TYPES, TrainConfig, TransferConfig
from .search import SearchConfig


OBJECTIVE_LABELS = {
    "Balanced accuracy and time": "balanced",
    "Best accuracy": "best_accuracy",
    "Fastest acceptable": "fastest_acceptable",
}
OBJECTIVE_BY_CODE = {value: key for key, value in OBJECTIVE_LABELS.items()}


class _NoScrollSpinBox(QSpinBox):
    """QSpinBox that only responds to scroll wheel when already focused."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
        else:
            super().wheelEvent(event)


class _NoScrollDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that only responds to scroll wheel when already focused."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
        else:
            super().wheelEvent(event)


class _NoScrollComboBox(QComboBox):
    """QComboBox that only responds to scroll wheel when already focused."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
        else:
            super().wheelEvent(event)


class _LicenseStateBridge(QObject):
    state_changed = Signal(object)


class StatusBadge(QLabel):
    """Small rounded badge used for environment and run states."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(28)
        self.set_status(text)

    def set_status(self, status: str) -> None:
        color = status_colors().get(status, APP_THEME.muted_text)
        self.setText(status)
        self.setStyleSheet(
            f"""
            QLabel {{
                background: {color}22;
                color: {color};
                border: 1px solid {color}44;
                border-radius: 14px;
                padding: 4px 10px;
                font-weight: 700;
            }}
            """
        )


class CardFrame(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("CardFrame")
        self.setFrameShape(QFrame.Shape.NoFrame)


class MetricCard(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("CardFrame")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.setMinimumHeight(52)
        self.setMaximumHeight(58)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)
        title_label = QLabel(title)
        title_label.setObjectName("SectionSubtitle")
        title_label.setWordWrap(False)
        self.value_label = QLabel("-")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setStyleSheet("font-size: 10.5pt; font-weight: 700;")
        self.value_label.setMinimumWidth(76)
        layout.addWidget(title_label, 1)
        layout.addWidget(self.value_label, 0)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class CollapsibleSection(QWidget):
    def __init__(self, title: str, content: QWidget) -> None:
        super().__init__()
        self.content = content
        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.toggled.connect(self._on_toggled)
        self.content.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content)

    def _on_toggled(self, expanded: bool) -> None:
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.content.setVisible(expanded)


class MlpTrainingStudio(QMainWindow):
    """Main application window."""

    def __init__(self, *, executor: QtTaskExecutor | None = None) -> None:
        super().__init__()
        self.executor = executor or QtTaskExecutor()
        self.current_task = None
        self.license_connection_message = "Enter a license server URL to enable floating-seat checkout."
        self.license_server_status: LicenseStatus | None = None
        self.license_state_bridge = _LicenseStateBridge(self)
        self.license_controller = LicenseLeaseController(state_callback=self.license_state_bridge.state_changed.emit)
        self.license_state_bridge.state_changed.connect(self._apply_license_state)
        self.license_lease_state = self.license_controller.state
        self.last_scan_result: dict[str, Any] | None = None
        self.last_suggest_result: dict[str, Any] | None = None
        self.last_search_result: dict[str, Any] | None = None
        self.last_workflow_summary: dict[str, Any] | None = None
        self.last_baseline_summary: dict[str, Any] | None = None
        self.last_transfer_base_summary: dict[str, Any] | None = None
        self.last_onnx_export_path: str | None = None
        self.selected_search_full_config: dict[str, Any] | None = None
        self.search_row_configs: list[dict[str, Any]] = []
        self.current_task_name = "idle"
        self._cache_path_manually_selected = False
        self._setting_cache_path = False
        self._search_max_samples_autofill_value: int | None = None
        self._setting_search_max_samples = False
        self._controls_locked = False
        self._search_text_items: list[pg.TextItem] = []
        self._transfer_frequency_items: list[Any] = []

        self._baseline_epochs: list[float] = []
        self._baseline_train_losses: list[float] = []
        self._baseline_val_losses: list[float] = []
        self._transfer_iteration_mae_x: list[float] = []
        self._transfer_iteration_mae_y: list[float] = []
        self._transfer_frequency_history: list[list[float]] = []

        self.setWindowTitle("Surrogate Model Traning Suite")
        self.resize(1440, 920)
        self.setMinimumSize(1280, 800)

        self._build_ui()
        self._apply_default_values()
        self._load_last_session_if_available()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        root.addWidget(self._build_top_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(12)
        root.addWidget(splitter, 1)

        self.left_pane_container = self._build_left_pane()
        self.right_pane_container = self._build_right_pane()
        splitter.addWidget(self.left_pane_container)
        splitter.addWidget(self.right_pane_container)
        splitter.setStretchFactor(0, 8)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([760, 740])
        self.main_splitter = splitter

    def _build_top_bar(self) -> QWidget:
        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        layout = QHBoxLayout(top_bar)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(16)

        title_col = QVBoxLayout()
        title_label = QLabel("Surrogate Model Traning Suite")
        title_label.setObjectName("TopBarTitle")
        title_col.addWidget(title_label)
        layout.addLayout(title_col, 1)

        self.topbar_run_name = QLabel("No run configured")
        self.topbar_run_name.setObjectName("TopBarSubtitle")
        layout.addWidget(self.topbar_run_name)
        return top_bar

    def _build_left_pane(self) -> QWidget:
        container = QWidget()
        container.setMinimumWidth(660)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        content = QWidget()
        self.left_content = content
        self.left_layout = QVBoxLayout(content)
        self.left_layout.setContentsMargins(0, 0, 8, 0)
        self.left_layout.setSpacing(12)
        scroll.setWidget(content)

        self.left_layout.addWidget(self._build_license_card())
        self.left_layout.addWidget(self._build_data_sources_card())
        self.left_layout.addWidget(self._build_dataset_preview_card())
        self.left_layout.addWidget(self._build_suggest_card())
        self.left_layout.addWidget(self._build_training_settings_card())
        self.left_layout.addWidget(self._build_run_controls_card())
        self.left_layout.addStretch(1)
        return container

    def _build_right_pane(self) -> QWidget:
        container = QWidget()
        container.setMinimumWidth(500)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_metrics_card(), 0)

        # Monitor plots on top, run log below — both resizable via a vertical splitter.
        monitor_log_splitter = QSplitter(Qt.Orientation.Vertical)
        monitor_log_splitter.setChildrenCollapsible(False)
        monitor_log_splitter.setHandleWidth(10)
        monitor_log_splitter.addWidget(self._build_monitor_tabs())
        monitor_log_splitter.addWidget(self._build_run_log_card())
        monitor_log_splitter.setStretchFactor(0, 3)
        monitor_log_splitter.setStretchFactor(1, 1)
        monitor_log_splitter.setSizes([640, 220])
        layout.addWidget(monitor_log_splitter, 1)
        return container

    def _build_run_log_card(self) -> QWidget:
        card = CardFrame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(self._section_header("Run Log", "Live status, scan, and training messages"))

        self.run_log_text_edit = QPlainTextEdit()
        self.run_log_text_edit.setReadOnly(True)
        self.run_log_text_edit.setMaximumBlockCount(4000)
        self.run_log_text_edit.setMinimumHeight(90)
        self.run_log_text_edit.setStyleSheet(
            'QPlainTextEdit { font-family: "Cascadia Code", "Consolas", monospace; font-size: 9.5pt; }'
        )
        layout.addWidget(self.run_log_text_edit, 1)
        return card

    def _build_license_card(self) -> QWidget:
        card = CardFrame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(self._section_header("License", "Floating-seat server settings, seat checkout, and heartbeat health"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.license_server_url_edit = QLineEdit()
        self.license_server_url_edit.setPlaceholderText("http://license-host:27850")
        self.license_server_url_edit.textChanged.connect(self._on_license_server_url_changed)
        self.test_license_connection_button = self._make_button("Test Connection", secondary=True)
        self.acquire_license_seat_button = self._make_button("Acquire Seat")
        self.test_license_connection_button.clicked.connect(self.run_license_connection_test)
        self.acquire_license_seat_button.clicked.connect(self.acquire_license_seat)
        server_row = QWidget()
        server_row_layout = QHBoxLayout(server_row)
        server_row_layout.setContentsMargins(0, 0, 0, 0)
        server_row_layout.setSpacing(8)
        server_row_layout.addWidget(self.license_server_url_edit, 1)
        server_row_layout.addWidget(self.test_license_connection_button, 0)
        server_row_layout.addWidget(self.acquire_license_seat_button, 0)
        grid.addWidget(QLabel("Server URL"), 0, 0)
        grid.addWidget(server_row, 0, 1, 1, 2)

        self.license_server_status_badge = StatusBadge("Unconfigured")
        self.license_seat_state_badge = StatusBadge(self.license_lease_state.badge_text)
        self.license_company_value = QLabel("Not checked yet")
        self.license_window_value = QLabel("Not checked yet")
        self.license_seat_usage_value = QLabel("Not checked yet")
        self.license_lease_expires_value = QLabel("No active seat")

        self._add_form_row(grid, 1, "Server Status", self.license_server_status_badge)
        self._add_form_row(grid, 2, "Seat State", self.license_seat_state_badge)
        self._add_form_row(grid, 3, "Company", self.license_company_value)
        self._add_form_row(grid, 4, "License Window", self.license_window_value)
        self._add_form_row(grid, 5, "Seat Usage", self.license_seat_usage_value)
        self._add_form_row(grid, 6, "Lease Expires", self.license_lease_expires_value)
        layout.addLayout(grid)

        self.license_status_text = QLabel(self.license_connection_message)
        self.license_status_text.setWordWrap(True)
        layout.addWidget(self.license_status_text)
        return card

    def _build_data_sources_card(self) -> QWidget:
        card = CardFrame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(self._section_header("Data Sources", "Select your dataset folder and output location"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.dataset_folder_edit = QLineEdit()
        self.dataset_folder_edit.setPlaceholderText("Select the root folder containing your entire dataset")
        self.dataset_folder_edit.textChanged.connect(self._autofill_run_name_from_dataset_folder)
        self.browse_dataset_folder_button = self._make_button("Browse...", secondary=True)
        self.browse_dataset_folder_button.clicked.connect(self._browse_dataset_folder)
        self._add_path_row(grid, 0, "Dataset Folder", self.dataset_folder_edit, self.browse_dataset_folder_button)

        self.model_output_folder_path_edit = QLineEdit()
        self.model_output_folder_path_edit.setPlaceholderText("Select the output folder for runs, artifacts, and cache")
        self.model_output_folder_path_edit.textChanged.connect(self._on_output_folder_changed)
        self.browse_model_output_folder_button = self._make_button("Browse...", secondary=True)
        self.browse_model_output_folder_button.clicked.connect(self.browse_model_output_dir)
        self._add_path_row(grid, 1, "Output Folder", self.model_output_folder_path_edit, self.browse_model_output_folder_button)

        self.run_name_edit = QLineEdit()
        self.run_name_edit.textChanged.connect(self._on_run_name_changed)
        self._add_form_row(grid, 2, "Run Name", self.run_name_edit)

        # Advanced: legacy explicit path fields + cache control.
        advanced = QWidget()
        advanced_grid = QGridLayout(advanced)
        advanced_grid.setContentsMargins(0, 0, 0, 0)
        advanced_grid.setHorizontalSpacing(10)
        advanced_grid.setVerticalSpacing(8)

        self.input_feature_path_edit = QLineEdit()
        self.input_feature_path_edit.setPlaceholderText("Override: explicit input-feature file (optional)")
        self.input_feature_path_edit.textChanged.connect(self._autofill_run_name_and_cache)
        self.browse_input_feature_button = self._make_button("Browse...", secondary=True)
        self.browse_input_feature_button.clicked.connect(self.browse_input_feature_path)
        self._add_path_row(advanced_grid, 0, "Input-Feature File", self.input_feature_path_edit, self.browse_input_feature_button)

        self.ground_truth_data_folder_path_edit = QLineEdit()
        self.ground_truth_data_folder_path_edit.setPlaceholderText("Override: explicit ground-truth data folder (optional)")
        self.ground_truth_data_folder_path_edit.textChanged.connect(self._autofill_run_name_from_gt_dir)
        self.browse_ground_truth_data_folder_button = self._make_button("Browse...", secondary=True)
        self.browse_ground_truth_data_folder_button.clicked.connect(self.browse_ground_truth_data_dir)
        self._add_path_row(advanced_grid, 1, "Ground-Truth Data Folder", self.ground_truth_data_folder_path_edit, self.browse_ground_truth_data_folder_button)

        self.cache_path_edit = QLineEdit()
        self.cache_path_edit.setPlaceholderText("Auto-managed inside the output folder if left blank")
        self.cache_path_edit.textEdited.connect(self._on_cache_path_edited)
        self.browse_cache_button = self._make_button("Browse...", secondary=True)
        self.browse_cache_button.clicked.connect(self.browse_cache_path)
        self._add_path_row(advanced_grid, 2, "Cache File", self.cache_path_edit, self.browse_cache_button)

        self.advanced_paths_section = CollapsibleSection("Advanced", advanced)
        layout.addLayout(grid)

        # Schema status badge — right after the main fields, before Advanced.
        button_row = QHBoxLayout()
        self.dataset_schema_status_badge = StatusBadge("Not Scanned")
        button_row.addWidget(self.dataset_schema_status_badge)
        button_row.addStretch()
        self.scan_dataset_button = self._make_button("Scan Dataset")
        self.scan_dataset_button.clicked.connect(self.scan_dataset)
        button_row.addWidget(self.scan_dataset_button)
        layout.addLayout(button_row)

        layout.addWidget(self.advanced_paths_section)
        return card

    def _build_dataset_preview_card(self) -> QWidget:
        card = CardFrame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(self._section_header("Dataset Preview", "Detected schema details and cache-ready preview"))

        self.data_preview_table = self._make_table(["Item", "Value"], stretch_last=True)
        self.data_preview_table.verticalHeader().setVisible(False)
        self.data_preview_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.data_preview_table)
        self._fill_table(self.data_preview_table, [("Status", "Scan the dataset to populate this preview.")])
        return card

    def _build_suggest_card(self) -> QWidget:
        card = CardFrame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(self._section_header("Suggest Initial Settings", "Scan-only starter values for baseline and transfer"))

        header = QHBoxLayout()
        self.suggest_initial_settings_button = self._make_button("Suggest Initial Settings")
        self.suggest_initial_settings_button.clicked.connect(self.run_suggest_initial_settings)
        self.initial_suggestion_confidence_badge = StatusBadge("Not Scanned")
        self.initial_suggestion_status_text = QLabel("Run a scan-only suggestion after the dataset preview looks correct.")
        self.initial_suggestion_status_text.setWordWrap(True)
        header.addWidget(self.suggest_initial_settings_button)
        header.addWidget(self.initial_suggestion_confidence_badge)
        header.addStretch(1)
        layout.addLayout(header)
        layout.addWidget(self.initial_suggestion_status_text)

        diagnostics_label = QLabel("Diagnostics")
        diagnostics_label.setObjectName("SectionSubtitle")
        self.initial_dataset_diagnostics_table = self._make_table(["Metric", "Value"], stretch_last=True)
        layout.addWidget(diagnostics_label)
        layout.addWidget(self.initial_dataset_diagnostics_table)

        settings_label = QLabel("Suggested Settings")
        settings_label.setObjectName("SectionSubtitle")
        self.initial_settings_table = self._make_table(["Parameter", "Suggested", "Suggested Range", "Reason"], stretch_last=True)
        layout.addWidget(settings_label)
        layout.addWidget(self.initial_settings_table)

        warnings_label = QLabel("Warnings")
        warnings_label.setObjectName("SectionSubtitle")
        self.initial_suggestion_warnings_box = QPlainTextEdit()
        self.initial_suggestion_warnings_box.setReadOnly(True)
        self.initial_suggestion_warnings_box.setFixedHeight(96)
        layout.addWidget(warnings_label)
        layout.addWidget(self.initial_suggestion_warnings_box)

        buttons = QHBoxLayout()
        self.apply_initial_settings_button = self._make_button("Use Suggested Settings", secondary=True)
        self.apply_initial_settings_button.clicked.connect(self.apply_suggested_settings)
        buttons.addWidget(self.apply_initial_settings_button)
        layout.addLayout(buttons)
        return card

    def _build_training_settings_card(self) -> QWidget:
        card = CardFrame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(self._section_header("Training Settings", "Editable baseline and transfer hyperparameters"))

        device_grid = QGridLayout()
        device_grid.setHorizontalSpacing(10)
        device_grid.setVerticalSpacing(8)
        self.training_device_combo_box = _NoScrollComboBox()
        self.training_device_combo_box.setToolTip(
            "Compute unit used for baseline and transfer training. Detected automatically when the app starts."
        )
        self._populate_device_choices()
        self._add_form_row(device_grid, 0, "Training Device", self.training_device_combo_box)
        layout.addLayout(device_grid)

        self.training_tabs = QTabWidget()
        layout.addWidget(self.training_tabs)

        self.training_tabs.addTab(self._build_baseline_tab(), "Baseline Training")
        self.training_tabs.addTab(self._build_transfer_tab(), "Self-Transfer Learning")
        return card

    def _populate_device_choices(self) -> None:
        """Auto-detect compute devices and fill the Training Device dropdown.

        Called once while the window is built so the user never has to trigger a
        manual environment check. The first CUDA device (when present) is selected
        by default, matching the previous auto-select behavior.
        """
        previous = self.training_device_combo_box.currentData()
        self.training_device_combo_box.blockSignals(True)
        self.training_device_combo_box.clear()
        for device in list_available_devices():
            self.training_device_combo_box.addItem(device["label"], device["id"])
        if previous is not None:
            restored = self.training_device_combo_box.findData(previous)
            if restored >= 0:
                self.training_device_combo_box.setCurrentIndex(restored)
        self.training_device_combo_box.blockSignals(False)

    def _current_device_id(self) -> str | None:
        """Return the selected device string (e.g. ``"cuda:0"``) for a run config."""
        return self.training_device_combo_box.currentData()

    def _set_device_selection(self, device_id: str | None) -> None:
        """Select ``device_id`` when it is still available; otherwise leave the default."""
        if not device_id:
            return
        index = self.training_device_combo_box.findData(device_id)
        if index >= 0:
            self.training_device_combo_box.setCurrentIndex(index)

    def _build_baseline_tab(self) -> QWidget:
        tab = QWidget()
        grid = QGridLayout(tab)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.baseline_model_type_combo_box = _NoScrollComboBox()
        self.baseline_model_type_combo_box.addItems(list(MODEL_TYPES))
        self.baseline_model_type_combo_box.setCurrentText("FlatMLP")
        self.baseline_epochs_spin_box = self._make_int_spin(1, 5000, 300)
        self.baseline_batch_size_spin_box = self._make_int_spin(1, 4096, 16)
        self.baseline_learning_rate_spin_box = self._make_float_spin(1e-6, 1.0, 1e-4, decimals=6, step=1e-5, scientific=True)
        self.baseline_weight_decay_spin_box = self._make_float_spin(0.0, 1.0, 1e-4, decimals=6, step=1e-5, scientific=True)
        self.baseline_width_spin_box = self._make_int_spin(16, 8192, 512)
        self.baseline_depth_spin_box = self._make_int_spin(1, 20, 5)
        self.baseline_loss_function_combo_box = _NoScrollComboBox()
        self.baseline_loss_function_combo_box.addItems(list(LOSS_FUNCTIONS))
        self.baseline_loss_function_combo_box.setCurrentText("rmse")
        self.baseline_scheduler_combo_box = _NoScrollComboBox()
        self.baseline_scheduler_combo_box.addItems(list(SCHEDULER_TYPES))
        self.baseline_scheduler_combo_box.setCurrentText("plateau")
        self.baseline_train_fraction_spin_box = self._make_float_spin(0.05, 0.95, 0.80, decimals=3, step=0.01)
        self.baseline_validation_fraction_spin_box = self._make_float_spin(0.0, 0.90, 0.10, decimals=3, step=0.01)
        self.baseline_seed_spin_box = self._make_int_spin(0, 1000000, 42)
        self.restore_recommended_baseline_button = self._make_button("Restore Recommended", secondary=True)
        self.restore_recommended_baseline_button.clicked.connect(self._restore_recommended_baseline)
        self.ctle_preset_button = self._make_button("CTLE Defaults", secondary=True)
        self.ctle_preset_button.clicked.connect(self._apply_ctle_preset)

        fields = [
            ("Model Type", self.baseline_model_type_combo_box),
            ("Full Training Epochs", self.baseline_epochs_spin_box),
            ("Batch Size", self.baseline_batch_size_spin_box),
            ("Learning Rate", self.baseline_learning_rate_spin_box),
            ("Weight Decay", self.baseline_weight_decay_spin_box),
            ("Network Width", self.baseline_width_spin_box),
            ("Network Depth", self.baseline_depth_spin_box),
            ("Loss Function", self.baseline_loss_function_combo_box),
            ("LR Scheduler", self.baseline_scheduler_combo_box),
            ("Train Fraction", self.baseline_train_fraction_spin_box),
            ("Validation Fraction", self.baseline_validation_fraction_spin_box),
            ("Random Seed", self.baseline_seed_spin_box),
        ]
        for row, (label, widget) in enumerate(fields):
            self._add_form_row(grid, row, label, widget)
        grid.addWidget(self.restore_recommended_baseline_button, len(fields), 1)
        grid.addWidget(self.ctle_preset_button, len(fields), 2)
        return tab

    def _build_transfer_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.enable_transfer_learning_checkbox = QCheckBox("Enable Frequency-Domain Transfer Learning")
        self.enable_transfer_learning_checkbox.setChecked(True)
        self.enable_transfer_learning_checkbox.hide()
        self.enable_transfer_learning_checkbox.toggled.connect(self._refresh_transfer_controls_enabled)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.transfer_base_model_source_combo_box = _NoScrollComboBox()
        self.transfer_base_model_source_combo_box.addItems(["Use baseline from this session", "Choose existing run..."])
        self.transfer_base_model_source_combo_box.currentIndexChanged.connect(self._refresh_transfer_source_controls)

        self.transfer_base_run_path_edit = QLineEdit()
        self.transfer_base_run_path_edit.setPlaceholderText("Select an existing baseline run directory")
        self.transfer_base_run_path_edit.textChanged.connect(self.validate_transfer_compatibility)
        self.browse_transfer_base_run_button = self._make_button("Browse...", secondary=True)
        self.browse_transfer_base_run_button.clicked.connect(self.browse_transfer_base_run)

        self.transfer_compatibility_status_badge = StatusBadge("Not Checked")
        self.transfer_notes_label = QLabel("Run baseline training first, or choose an existing baseline run.")
        self.transfer_notes_label.setWordWrap(True)

        self.transfer_num_bands_spin_box = self._make_int_spin(1, 512, 10)
        self.transfer_num_bands_spin_box.valueChanged.connect(self._refresh_transfer_note_text)
        self.transfer_iterations_spin_box = self._make_int_spin(1, 100, 10)
        self.transfer_epochs_spin_box = self._make_int_spin(1, 5000, 100)
        self.transfer_batch_size_spin_box = self._make_int_spin(1, 4096, 64)
        self.transfer_learning_rate_spin_box = self._make_float_spin(1e-6, 1.0, 5e-5, decimals=6, step=1e-5, scientific=True)
        self.transfer_weight_decay_spin_box = self._make_float_spin(0.0, 1.0, 0.0, decimals=6, step=1e-5, scientific=True)
        self.transfer_weight_decay_spin_box.setSpecialValueText("Use baseline")
        self.transfer_seed_spin_box = self._make_int_spin(0, 1000000, 42)

        self._add_form_row(grid, 0, "Base Model Source", self.transfer_base_model_source_combo_box)
        self._add_path_row(grid, 1, "Baseline Run Folder", self.transfer_base_run_path_edit, self.browse_transfer_base_run_button)
        self._add_form_row(grid, 2, "Compatibility Status", self.transfer_compatibility_status_badge)
        self._add_form_row(grid, 3, "Number of Frequency Bands", self.transfer_num_bands_spin_box)
        self._add_form_row(grid, 4, "Transfer Iterations", self.transfer_iterations_spin_box)
        self._add_form_row(grid, 5, "Epochs per Transfer Stage", self.transfer_epochs_spin_box)
        self._add_form_row(grid, 6, "Batch Size", self.transfer_batch_size_spin_box)
        self._add_form_row(grid, 7, "Learning Rate", self.transfer_learning_rate_spin_box)
        self._add_form_row(grid, 8, "Weight Decay", self.transfer_weight_decay_spin_box)
        self._add_form_row(grid, 9, "Random Seed", self.transfer_seed_spin_box)
        layout.addLayout(grid)
        layout.addWidget(self.transfer_notes_label)
        return tab

    def _build_run_controls_card(self) -> QWidget:
        card = CardFrame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(self._section_header("Run Controls", "Launch baseline or self-transfer runs, stop gracefully, or export summaries"))

        buttons = QHBoxLayout()
        self.start_baseline_button = self._make_button("Start Baseline Training")
        self.start_transfer_button = self._make_button("Start Self-Transfer Learning")
        self.stop_training_button = self._make_button("Stop", secondary=True)
        self.stop_training_button.setEnabled(False)
        self.open_output_folder_button = self._make_button("Open Output Folder", secondary=True)
        self.export_run_summary_button = self._make_button("Export Summary", secondary=True)
        self.export_onnx_button = self._make_button("Export to ONNX", secondary=True)
        self.start_baseline_button.clicked.connect(self.start_baseline_training)
        self.start_transfer_button.clicked.connect(self.start_transfer_learning)
        self.stop_training_button.clicked.connect(self.stop_current_task)
        self.open_output_folder_button.clicked.connect(self.open_output_folder)
        self.export_run_summary_button.clicked.connect(self.export_run_summary)
        self.export_onnx_button.clicked.connect(self.export_baseline_to_onnx)
        buttons.addWidget(self.start_baseline_button)
        buttons.addWidget(self.start_transfer_button)
        buttons.addWidget(self.stop_training_button)
        buttons.addWidget(self.open_output_folder_button)
        buttons.addWidget(self.export_run_summary_button)
        buttons.addWidget(self.export_onnx_button)
        layout.addLayout(buttons)

        status_row = QHBoxLayout()
        self.run_progress_bar = QProgressBar()
        self.run_progress_bar.setRange(0, 100)
        self.run_progress_bar.setValue(0)
        self.run_progress_bar.setFormat("%p%")
        self.run_state_badge = StatusBadge("Idle")
        status_row.addWidget(self.run_progress_bar, 1)
        status_row.addWidget(self.run_state_badge)
        layout.addLayout(status_row)
        return card

    def _build_metrics_card(self) -> QWidget:
        card = CardFrame()
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.addWidget(self._section_header("Current Metrics", "Phase-aware live progress across scan, suggestion, baseline, and self-transfer"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        metric_titles = [
            ("current_phase", "Current Phase"),
            ("current_progress", "Primary Progress"),
            ("secondary_progress", "Secondary Progress"),
            ("best_metric", "Best Metric"),
            ("train_loss", "Train Loss"),
            ("validation_loss", "Validation Loss"),
            ("average_mae", "Average MAE"),
            ("elapsed", "Elapsed"),
            ("eta", "ETA"),
        ]
        self.metric_cards: dict[str, MetricCard] = {}
        for index, (key, title) in enumerate(metric_titles):
            card_widget = MetricCard(title)
            self.metric_cards[key] = card_widget
            grid.addWidget(card_widget, index // 3, index % 3)
        layout.addLayout(grid)
        return card

    def _build_monitor_tabs(self) -> QWidget:
        tabs = QTabWidget()
        tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        tabs.setMinimumHeight(200)

        baseline_tab = QWidget()
        baseline_layout = QGridLayout(baseline_tab)
        baseline_layout.setContentsMargins(12, 12, 12, 12)
        baseline_layout.setHorizontalSpacing(12)
        baseline_layout.setVerticalSpacing(12)
        self.baseline_loss_plot = pg.PlotWidget()
        self.baseline_frequency_mae_plot = pg.PlotWidget()
        self.baseline_loss_plot.setMinimumHeight(160)
        self.baseline_frequency_mae_plot.setMinimumHeight(160)
        baseline_layout.addWidget(self.baseline_loss_plot, 0, 0)
        baseline_layout.addWidget(self.baseline_frequency_mae_plot, 0, 1)
        baseline_layout.setColumnStretch(0, 1)
        baseline_layout.setColumnStretch(1, 1)
        baseline_layout.setRowStretch(0, 1)
        tabs.addTab(baseline_tab, "Baseline Monitor")

        transfer_tab = QWidget()
        transfer_layout = QVBoxLayout(transfer_tab)
        transfer_layout.setContentsMargins(12, 12, 12, 12)
        transfer_layout.setSpacing(10)
        self.transfer_frequency_mae_plot = pg.PlotWidget(transfer_tab)
        self.transfer_frequency_mae_plot.setMinimumHeight(200)
        transfer_layout.addWidget(self.transfer_frequency_mae_plot, 1)
        tabs.addTab(transfer_tab, "Transfer Results")

        # Test Samples tab — shows prediction vs ground truth for test samples.
        test_samples_scroll = QScrollArea()
        test_samples_scroll.setWidgetResizable(True)
        test_samples_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        test_samples_content = QWidget()
        self.test_samples_layout = QVBoxLayout(test_samples_content)
        self.test_samples_layout.setContentsMargins(12, 12, 12, 12)
        self.test_samples_layout.setSpacing(12)
        self.test_samples_placeholder = QLabel(
            "Test sample plots will appear here after training completes.\n"
            "Each plot compares predicted (dashed) vs true (solid) curves."
        )
        self.test_samples_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.test_samples_placeholder.setStyleSheet("color: #888; font-size: 11pt; padding: 40px;")
        self.test_samples_layout.addWidget(self.test_samples_placeholder)
        self.test_samples_layout.addStretch()
        test_samples_scroll.setWidget(test_samples_content)
        self._test_sample_plots: list[pg.PlotWidget] = []
        tabs.addTab(test_samples_scroll, "Test Samples")

        self.monitor_tabs = tabs
        self._reset_baseline_plots()
        self._reset_transfer_plots()
        return tabs

    # ------------------------------------------------------------------
    # Defaults and persistence
    # ------------------------------------------------------------------
    def _apply_default_values(self) -> None:
        self.model_output_folder_path_edit.setText(str(self._default_output_dir().resolve()))
        self._set_cache_path_value("", manually_selected=False)
        self.run_name_edit.setText("mlp_run")
        self.dataset_schema_status_badge.set_status("Not Scanned")
        self.initial_suggestion_confidence_badge.set_status("Not Scanned")
        self.transfer_compatibility_status_badge.set_status("Not Checked")
        self.license_server_status_badge.set_status("Unconfigured")
        self.license_seat_state_badge.set_status(self.license_lease_state.badge_text)
        self._set_metric_defaults()
        self._refresh_transfer_controls_enabled()
        self._refresh_transfer_source_controls()
        self._refresh_license_display()
        self._update_topbar_run_name()
        self.append_log("Ready. Select dataset paths and scan the data to begin.")

    def _load_last_session_if_available(self) -> None:
        payload = load_last_session()
        if not payload:
            return
        try:
            self.apply_config_payload(payload)
        except Exception as exc:  # pragma: no cover - defensive startup path
            self.append_log(f"Could not restore the last GUI session: {exc}")

    def collect_config_payload(self) -> dict[str, Any]:
        return {
            "device": self._current_device_id(),
            "data_sources": {
                "input_feature_path": self.input_feature_path_edit.text().strip(),
                "ground_truth_data_dir": self.ground_truth_data_folder_path_edit.text().strip(),
                "output_dir": self.model_output_folder_path_edit.text().strip(),
                "run_name": self.run_name_edit.text().strip(),
                "cache_path": self.cache_path_edit.text().strip(),
                "cache_path_manually_selected": self._cache_path_manually_selected,
            },
            "baseline": self._collect_baseline_form(),
            "transfer": {
                "enabled": True,
                "base_model_source": self.transfer_base_model_source_combo_box.currentIndex(),
                "base_run_dir": self.transfer_base_run_path_edit.text().strip(),
                **self._collect_transfer_form(),
            },
            "licensing": {
                "server_url": self.license_server_url_edit.text().strip(),
                "last_status": self._current_license_status_summary(),
            },
        }

    def apply_config_payload(self, payload: dict[str, Any]) -> None:
        self._set_device_selection(payload.get("device"))
        data_sources = payload.get("data_sources", {})
        self.input_feature_path_edit.setText(str(data_sources.get("input_feature_path", "")))
        self.ground_truth_data_folder_path_edit.setText(str(data_sources.get("ground_truth_data_dir", "")))
        output_dir = data_sources.get("output_dir", data_sources.get("model_output_dir", self.model_output_folder_path_edit.text()))
        self.model_output_folder_path_edit.setText(str(output_dir))
        run_name = str(data_sources.get("run_name", self.run_name_edit.text()))
        self.run_name_edit.setText(run_name)
        cache_path = str(data_sources.get("cache_path", self.cache_path_edit.text())).strip()
        manually_selected = data_sources.get("cache_path_manually_selected")
        if manually_selected is None:
            manually_selected = self._infer_cache_path_manually_selected(cache_path, run_name=run_name)
        self._set_cache_path_value(cache_path, manually_selected=bool(manually_selected))
        if not self._cache_path_manually_selected:
            self._sync_auto_cache_path()

        baseline = payload.get("baseline", {})
        self._apply_baseline_form(baseline)

        transfer = payload.get("transfer", {})
        self.enable_transfer_learning_checkbox.setChecked(True)
        self.transfer_base_model_source_combo_box.setCurrentIndex(int(transfer.get("base_model_source", 0)))
        self.transfer_base_run_path_edit.setText(str(transfer.get("base_run_dir", "")))
        self._apply_transfer_form(transfer)
        self._refresh_transfer_controls_enabled()
        self._refresh_transfer_source_controls()
        self._apply_license_payload(payload.get("licensing", {}))
        self._update_topbar_run_name()

    def _apply_license_payload(self, payload: dict[str, Any]) -> None:
        server_url = normalize_server_url(str(payload.get("server_url", "")))
        stored_status = payload.get("last_status")
        self.license_server_url_edit.setText(server_url)
        if isinstance(stored_status, dict) and stored_status:
            try:
                self.license_server_status = LicenseStatus.from_payload(stored_status)
            except Exception:
                self.license_server_status = None
        else:
            self.license_server_status = None

        if not server_url:
            self.license_connection_message = "Enter a license server URL to enable floating-seat checkout."
            self.license_controller.clear_configuration()
            self._refresh_license_display()
            return

        self.license_connection_message = "Saved server URL restored. Click Acquire Seat to request a seat."
        self._refresh_license_display()
        self.acquire_license_seat()

    def _current_license_status_summary(self) -> dict[str, Any] | None:
        effective_status = self.license_lease_state.last_status or self.license_server_status
        if effective_status is None:
            return None
        return effective_status.to_payload()

    def _run_license_status_check(
        self,
        *,
        server_url: str,
        progress_callback=None,
        should_stop=None,
    ) -> dict[str, Any]:
        del progress_callback, should_stop
        try:
            status = test_license_connection(server_url)
        except LicenseClientError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "status": status}

    def _run_license_checkout(
        self,
        *,
        server_url: str,
        progress_callback=None,
        should_stop=None,
    ) -> LicenseLeaseState:
        del progress_callback, should_stop
        return self.license_controller.connect_and_checkout(server_url)

    def _on_license_status_checked(self, result: dict[str, Any]) -> None:
        if not result.get("ok"):
            self.license_server_status_badge.set_status("Error")
            self.license_connection_message = str(result.get("error") or "Could not reach the license server.")
            self.append_log(f"License connection test failed: {self.license_connection_message}")
            self._refresh_license_display()
            return
        status = result["status"]
        self.license_server_status = status
        self.license_server_status_badge.set_status("Connected")
        company = status.company_name or "the configured server"
        self.license_connection_message = f"Connected to {company}. {status.seat_summary}."
        self.append_log(f"License server is reachable: {self.license_connection_message}")
        self._refresh_license_display()

    def _on_license_checkout_completed(self, state: LicenseLeaseState) -> None:
        if state.phase == "checked_out":
            self.append_log("License checkout granted. Background heartbeats are running.")
        elif state.phase == "checkout_denied":
            self.append_log(f"License checkout denied: {state.message}")
        elif state.phase == "error":
            self.append_log(f"License checkout failed: {state.message}")
        self._refresh_license_display()

    @Slot(object)
    def _apply_license_state(self, state: LicenseLeaseState) -> None:
        previous = self.license_lease_state
        self.license_lease_state = state
        if state.last_status is not None:
            self.license_server_status = state.last_status
        if state.phase == "checking":
            self.license_server_status_badge.set_status("Checking")
        elif state.phase == "error":
            self.license_server_status_badge.set_status("Error")
        elif state.server_url and state.last_status is not None:
            self.license_server_status_badge.set_status("Connected")
        elif not state.server_url:
            self.license_server_status_badge.set_status("Unconfigured")
        self._refresh_license_display()
        self._log_license_transition(previous, state)

    def _log_license_transition(self, previous: LicenseLeaseState, current: LicenseLeaseState) -> None:
        if previous.phase == current.phase and previous.heartbeat_failures == current.heartbeat_failures:
            return
        if current.phase == "heartbeat_warning":
            self.append_log(current.message)
        elif current.phase == "license_required":
            self.append_log(current.message)
        elif current.phase == "released" and previous.phase != "released":
            self.append_log(current.message)

    def _on_license_server_url_changed(self, text: str) -> None:
        configured_url = normalize_server_url(text)
        active_url = normalize_server_url(self.license_lease_state.server_url)
        if not configured_url:
            self.license_server_status_badge.set_status("Unconfigured")
            self.license_connection_message = "Enter a license server URL to enable floating-seat checkout."
        elif configured_url != active_url:
            self.license_server_status_badge.set_status("Not Checked")
            self.license_connection_message = "Click Test Connection or Acquire Seat to use this server."
        self._refresh_license_display()

    def _refresh_license_display(self) -> None:
        effective_status = self.license_lease_state.last_status or self.license_server_status
        self.license_seat_state_badge.set_status(self.license_lease_state.badge_text)
        self.license_company_value.setText(effective_status.company_name if effective_status and effective_status.company_name else "Not checked yet")
        self.license_window_value.setText(effective_status.window_summary if effective_status else "Not checked yet")
        self.license_seat_usage_value.setText(effective_status.seat_summary if effective_status else "Not checked yet")
        self.license_lease_expires_value.setText(format_utc_timestamp(self.license_lease_state.expires_at) or "No active seat")
        self.license_status_text.setText(self._license_display_message())
        self._refresh_run_button_availability()

    def _license_display_message(self) -> str:
        configured_url = normalize_server_url(self.license_server_url_edit.text())
        active_url = normalize_server_url(self.license_lease_state.server_url)
        if active_url and configured_url != active_url:
            return "Server URL changed. Click Acquire Seat to reconnect before starting another run."
        if self.license_lease_state.phase != "unconfigured":
            return self.license_lease_state.message
        return self.license_connection_message

    def _license_allows_new_runs(self) -> bool:
        configured_url = normalize_server_url(self.license_server_url_edit.text())
        active_url = normalize_server_url(self.license_lease_state.server_url)
        if not configured_url and not active_url:
            return True
        if configured_url != active_url:
            return False
        return self.license_lease_state.can_start_runs

    def _ensure_license_ready_for_training(self) -> bool:
        if self._license_allows_new_runs():
            return True
        self._show_warning(self._license_display_message())
        return False

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def scan_dataset(self) -> None:
        payload = self._require_data_paths()
        if payload is None:
            return
        # Log which paths are being used so we can debug path issues.
        self.append_log(f"Scan paths: dataset_root={payload.get('dataset_root', '')!r}, "
                        f"input_feature={payload.get('input_feature_path', '')!r}, "
                        f"gt_dir={payload.get('ground_truth_data_dir', '')!r}")
        self.last_scan_result = None
        self.dataset_schema_status_badge.set_status("Scanning")
        self._fill_table(self.data_preview_table, [("Status", "Building cache from the dataset folder and scanning...")])
        self._start_task(
            build_and_scan_dataset,
            kwargs={
                **payload,
                "train_frac": self.baseline_train_fraction_spin_box.value(),
                "val_frac": self.baseline_validation_fraction_spin_box.value(),
                "seed": self.baseline_seed_spin_box.value(),
                "max_samples": None,
                # Auto-managed cache: rebuild from the folder. Manually-selected cache:
                # reuse it if present (only build when missing).
                "overwrite": not self._cache_path_manually_selected,
            },
            task_name="scan",
            busy_state="Scanning",
            on_result=self._on_scan_completed,
        )

    def run_suggest_initial_settings(self) -> None:
        payload = self._require_data_paths()
        if payload is None:
            return
        self.initial_suggestion_confidence_badge.set_status("Checking")
        self.initial_suggestion_status_text.setText("Computing scan-only recommendations...")
        self._start_task(
            build_suggest_result,
            kwargs={
                **payload,
                "seed": self.baseline_seed_spin_box.value(),
                "train_frac": self.baseline_train_fraction_spin_box.value(),
                "val_frac": self.baseline_validation_fraction_spin_box.value(),
                "max_samples": None,
            },
            task_name="suggest",
            busy_state="Suggesting",
            on_result=self._on_suggest_completed,
        )

    def start_baseline_training(self) -> None:
        if not self._ensure_license_ready_for_training():
            return
        payload = self._require_data_paths()
        if payload is None:
            return
        if self.last_scan_result is None:
            self._show_warning("Please scan the dataset before starting training.")
            return
        if not self._validate_split_fractions():
            return

        self._reset_baseline_plots()
        self._reset_transfer_plots()
        self.last_workflow_summary = None
        self.last_transfer_base_summary = None

        self._start_task(
            run_training_workflow,
            kwargs={
                "baseline_config": self._build_baseline_train_config(),
                "transfer_config": None,
                "transfer_base_run_dir": None,
            },
            task_name="training",
            busy_state="Training",
            on_result=self._on_training_workflow_completed,
        )

    def start_transfer_learning(self) -> None:
        if not self._ensure_license_ready_for_training():
            return
        payload = self._require_data_paths()
        if payload is None:
            return
        if self.last_scan_result is None:
            self._show_warning("Please scan the dataset before starting self-transfer learning.")
            return
        if not self._validate_transfer_ready():
            return

        self._reset_transfer_plots()
        self.last_workflow_summary = None
        self.last_transfer_base_summary = None
        transfer_base_run_dir = (
            self.transfer_base_run_path_edit.text().strip()
            if self.transfer_base_model_source_combo_box.currentIndex() == 1
            else self._current_baseline_run_dir()
        )

        self._start_task(
            run_training_workflow,
            kwargs={
                "baseline_config": None,
                "transfer_config": self._build_transfer_config(),
                "transfer_base_run_dir": transfer_base_run_dir,
            },
            task_name="training",
            busy_state="Transfer",
            on_result=self._on_training_workflow_completed,
        )

    def start_training(self) -> None:
        self.start_baseline_training()

    def stop_current_task(self) -> None:
        if self.current_task is None:
            return
        self.current_task.stop()
        self.append_log("Stop requested. Waiting for the current stage to exit cleanly...")
        self.run_state_badge.set_status("Stopped")

    def _browse_dataset_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", self.dataset_folder_edit.text())
        if path:
            self.dataset_folder_edit.setText(path)

    def browse_input_feature_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Input-Feature File", self.input_feature_path_edit.text(), "Text Files (*.txt);;All Files (*)")
        if path:
            self.input_feature_path_edit.setText(path)

    def browse_ground_truth_data_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Ground-Truth Data Folder", self.ground_truth_data_folder_path_edit.text())
        if path:
            self.ground_truth_data_folder_path_edit.setText(path)

    def browse_model_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.model_output_folder_path_edit.text())
        if path:
            self.model_output_folder_path_edit.setText(path)

    def browse_cache_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Select Cache File", self.cache_path_edit.text(), "NumPy Cache (*.npz)")
        if path:
            self._set_cache_path_value(path, manually_selected=True)

    def browse_transfer_base_run(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Baseline Run Folder", self.transfer_base_run_path_edit.text())
        if path:
            self.transfer_base_run_path_edit.setText(path)
            self.validate_transfer_compatibility()

    def open_output_folder(self) -> None:
        candidate = self._preferred_output_path()
        if candidate is None:
            self._show_warning("No output folder is available yet.")
            return
        path = Path(candidate)
        if not path.exists():
            self._show_warning("The selected output path does not exist yet.")
            return
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except OSError as exc:
            self._show_warning(f"Could not open the output folder: {exc}")

    def export_run_summary(self) -> None:
        payload = {
            "scan": self.last_scan_result,
            "suggest": self.last_suggest_result,
            "search": self.last_search_result,
            "workflow": self.last_workflow_summary,
        }
        default_path = self._default_dialog_root() / f"{self.run_name_edit.text().strip() or 'mlp_run'}_summary.json"
        path, _ = QFileDialog.getSaveFileName(self, "Export Summary", str(default_path), "JSON Files (*.json)")
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.append_log(f"Exported GUI summary to {path}")

    def export_baseline_to_onnx(self) -> None:
        """Export the most recent baseline checkpoint to ONNX (e.g. for MATLAB)."""
        run_dir = self._current_baseline_run_dir()
        if not run_dir:
            self._show_warning("No baseline run is available yet. Complete a baseline training run first.")
            return
        checkpoint_path = Path(run_dir) / "best_model.pt"
        if not checkpoint_path.exists():
            self._show_warning(f"No checkpoint was found at {checkpoint_path}.")
            return
        default_path = checkpoint_path.with_suffix(".onnx")
        path, _ = QFileDialog.getSaveFileName(self, "Export to ONNX", str(default_path), "ONNX Files (*.onnx)")
        if not path:
            return
        self.append_log(f"Exporting baseline checkpoint to ONNX: {checkpoint_path}")
        self._start_task(
            export_model_to_onnx,
            kwargs={"checkpoint_path": str(checkpoint_path), "out_path": path},
            task_name="export",
            busy_state="Exporting",
            on_result=self._on_onnx_export_completed,
        )

    def _on_onnx_export_completed(self, result: dict[str, Any]) -> None:
        onnx_path = result.get("onnx_path", "")
        meta_path = result.get("meta_path", "")
        self.last_onnx_export_path = onnx_path or None
        self.append_log(f"Exported ONNX model to {onnx_path}")
        if meta_path:
            self.append_log(f"Wrote metadata sidecar to {meta_path}")

    def run_license_connection_test(self) -> None:
        server_url = normalize_server_url(self.license_server_url_edit.text())
        if not server_url:
            self._show_warning("Enter a license server URL first.")
            return
        self.license_server_status_badge.set_status("Checking")
        self.license_connection_message = "Testing connectivity to the license server..."
        self._refresh_license_display()
        self._start_task(
            self._run_license_status_check,
            kwargs={"server_url": server_url},
            task_name="license_test",
            busy_state="Checking",
            on_result=self._on_license_status_checked,
        )

    def acquire_license_seat(self) -> None:
        server_url = normalize_server_url(self.license_server_url_edit.text())
        if not server_url:
            self.license_server_status = None
            self.license_connection_message = "Enter a license server URL to enable floating-seat checkout."
            self.license_controller.clear_configuration()
            self._refresh_license_display()
            return
        self._start_task(
            self._run_license_checkout,
            kwargs={"server_url": server_url},
            task_name="license_checkout",
            busy_state="Checking",
            on_result=self._on_license_checkout_completed,
        )

    def validate_transfer_compatibility(self) -> None:
        if self.transfer_base_model_source_combo_box.currentIndex() == 0:
            if self._current_baseline_run_dir():
                self.transfer_compatibility_status_badge.set_status("Compatible")
                self.transfer_notes_label.setText("Self-transfer will use the baseline produced in this session.")
            else:
                self.transfer_compatibility_status_badge.set_status("Not Checked")
                self.transfer_notes_label.setText("Run baseline training first, or choose an existing baseline run.")
            return
        if self.last_scan_result is None or not self.transfer_base_run_path_edit.text().strip():
            self.transfer_compatibility_status_badge.set_status("Not Checked")
            self.transfer_notes_label.setText("Select a baseline run and scan the dataset to validate compatibility.")
            return
        result = check_transfer_compatibility(
            base_run_dir=self.transfer_base_run_path_edit.text().strip(),
            cache_path=self._ensure_cache_path(),
            input_feature_path=self.input_feature_path_edit.text().strip() or None,
            ground_truth_data_dir=self.ground_truth_data_folder_path_edit.text().strip() or None,
        )
        self.transfer_compatibility_status_badge.set_status(result["status"])
        self.transfer_notes_label.setText(result["message"])

    def apply_suggested_settings(self) -> None:
        if self.last_suggest_result is None:
            self._show_warning("No suggestion is available yet.")
            return
        self._apply_baseline_form(self.last_suggest_result["suggested_baseline_config"])
        self._apply_transfer_form(self.last_suggest_result["suggested_transfer_config"])
        self.append_log("Applied the scan-only suggested settings to the editable forms.")

    def prepare_quick_search_from_suggestion(self) -> None:
        if self.last_suggest_result is None:
            self._show_warning("Run Suggest Initial Settings first.")
            return
        self._focus_training_settings()
        self.append_log("Suggested settings are ready for baseline review.")

    def apply_recommended_search_result(self) -> None:
        config = self.selected_search_full_config
        if config is None:
            self._show_warning("Select a quick-search result first.")
            return
        self._apply_baseline_form(config)
        self.append_log("Applied the selected quick-search baseline configuration to the editable form.")

    # ------------------------------------------------------------------
    # Task plumbing
    # ------------------------------------------------------------------
    def _start_task(
        self,
        function,
        *,
        kwargs: dict[str, Any],
        task_name: str,
        busy_state: str,
        on_result,
    ) -> None:
        if self.current_task is not None:
            self._show_warning("A task is already running. Please stop it or wait for it to finish.")
            return
        self.current_task_name = task_name
        self.run_state_badge.set_status(busy_state)
        self.stop_training_button.setEnabled(task_name in {"search", "training", "suggest", "scan"})
        self._set_action_controls_enabled(False)
        handle = self.executor.start(
            function,
            kwargs=kwargs,
            on_progress=self._handle_progress_event,
            on_result=on_result,
            on_error=self._handle_task_error,
            on_finished=self._handle_task_finished,
        )
        self.current_task = None if handle.completed else handle

    def _handle_task_error(self, message: str, traceback_text: str) -> None:
        self.run_state_badge.set_status("Error")
        if self.current_task_name == "scan":
            self.dataset_schema_status_badge.set_status("Error")
            pass
        self.append_log(f"Error: {message}")
        self._show_warning(f"{message}\n\n{traceback_text}")

    def _force_clear_task_state(self) -> None:
        """Reset task bookkeeping so a new task can start immediately.

        This is needed when an error or result callback wants to chain into a
        follow-up task.  The Qt
        ``finished`` signal travels through ``thread.quit → thread.finished``
        and arrives asynchronously, so ``current_task`` is still set when the
        error/result callback fires.  Clearing it here avoids the "A task is
        already running" guard in ``_start_task``.

        The old thread is kept alive until it finishes quitting — dropping
        the reference too early causes "QThread destroyed while still running".
        """
        old_task = self.current_task
        if old_task is not None and old_task.thread is not None:
            thread = old_task.thread
            # Disconnect the old finished handler so it doesn't clobber the
            # new task's bookkeeping when it finally fires.
            try:
                thread.finished.disconnect(self._handle_task_finished)
            except (RuntimeError, TypeError):
                pass
            # Keep old thread alive until it actually finishes, then clean up.
            if not hasattr(self, "_retiring_threads"):
                self._retiring_threads: list[QThread] = []
            self._retiring_threads.append(thread)
            thread.finished.connect(lambda t=thread: self._retire_thread(t))
        self.current_task = None
        self.current_task_name = "idle"
        self._set_action_controls_enabled(True)
        self.stop_training_button.setEnabled(False)

    def _retire_thread(self, thread: QThread) -> None:
        """Remove a finished thread from the retirement list."""
        try:
            self._retiring_threads.remove(thread)
        except ValueError:
            pass

    def _handle_task_finished(self) -> None:
        self.current_task = None
        self.current_task_name = "idle"
        self._set_action_controls_enabled(True)
        self.stop_training_button.setEnabled(False)
        if self.run_state_badge.text() in {"Training", "Transfer", "Searching", "Scanning", "Suggesting", "Checking", "Exporting"}:
            self.run_state_badge.set_status("Idle")

    # ------------------------------------------------------------------
    # Progress and results
    # ------------------------------------------------------------------
    def _handle_progress_event(self, payload: dict[str, Any]) -> None:
        payload = self._normalize_progress_payload(payload)
        line = self._format_progress_line(payload)
        if line:
            self.append_log(line)

        phase = payload.get("phase")
        if phase == "scan":
            self._update_scan_progress(payload)
            return
        if phase == "suggest":
            self._update_suggest_progress(payload)
            return
        if phase == "search" or payload.get("trial_index") is not None:
            self._update_search_progress(payload)
            return
        if phase == "baseline":
            self._update_baseline_progress(payload)
            return
        if phase == "transfer":
            self._update_transfer_progress(payload)

    def _on_scan_completed(self, result: dict[str, Any]) -> None:
        self.last_scan_result = result
        self._sweep_label = result.get("sweep_label", "Frequency (GHz)")
        self.dataset_schema_status_badge.set_status(result.get("schema_status", "Valid"))
        self._fill_table(self.data_preview_table, result["preview_rows"])
        self._set_cache_path_value(result["cache_path"], manually_selected=self._cache_path_manually_selected)
        self.append_log(f"Dataset scan completed for {result['dataset_name']}.")
        self._reset_baseline_plots()
        self._reset_transfer_plots()
        self.validate_transfer_compatibility()
        self._refresh_transfer_note_text()
        self.run_progress_bar.setValue(100)
        self.run_state_badge.set_status("Completed")

    def _on_suggest_completed(self, result: dict[str, Any]) -> None:
        self.last_suggest_result = result
        self.initial_suggestion_confidence_badge.set_status(result["confidence"].title())
        self.initial_suggestion_status_text.setText(result["confidence_reason"])
        self._fill_table(
            self.initial_dataset_diagnostics_table,
            [(self._prettify_key(key), self._stringify(value)) for key, value in result["diagnostics"].items()],
        )
        self._fill_table(self.initial_settings_table, self._suggestion_rows(result))
        self.initial_suggestion_warnings_box.setPlainText("\n".join(result["warnings"]))
        self.append_log("Initial settings suggestion is ready.")
        self.run_progress_bar.setValue(100)
        self.run_state_badge.set_status("Completed")

    def _on_search_completed(self, result: dict[str, Any]) -> None:
        if not hasattr(self, "search_results_table"):
            self.append_log("Quick search results are not shown because quick search has been removed from the GUI.")
            return
        self.last_search_result = result
        self.search_row_configs = [trial["recommended_full_config"] for trial in result["trial_results"]]
        self.selected_search_full_config = self.search_row_configs[0] if self.search_row_configs else None
        rows = [self._search_row(trial) for trial in result["trial_results"]]
        self._fill_table(self.search_results_table, rows)
        self._fill_table(self.search_ranking_table_mirror, rows)
        if result["recommended_trial"] is not None:
            trial = result["recommended_trial"]
            self.recommended_config_summary.setText(
                f"Recommended: {trial['label']} | Avg MAE {trial['average_val_mae']:.6f} | "
                f"Best val loss {trial['best_val_loss']:.6f} | Runtime {self._format_seconds(trial['runtime_seconds'])}"
            )
            self.search_results_table.selectRow(0)
        self._plot_search_tradeoff(result["trial_results"])
        self.monitor_tabs.setCurrentIndex(1)
        self.run_progress_bar.setValue(100)
        self.run_state_badge.set_status("Completed")
        self.append_log("Quick search completed.")

    def _on_training_workflow_completed(self, result: dict[str, Any]) -> None:
        self.last_workflow_summary = result
        status = result.get("status", "ok")
        if status == "stopped":
            self.run_state_badge.set_status("Stopped")
            self.append_log("Training workflow stopped.")
            return

        self.run_state_badge.set_status("Completed")
        self.run_progress_bar.setValue(100)
        baseline = result.get("baseline")
        transfer = result.get("transfer")
        if baseline:
            self.last_baseline_summary = baseline
            self._apply_baseline_summary(baseline)
            self.append_log(f"Baseline run saved to {baseline['run_dir']}")
            if baseline.get("test_sample_data"):
                self._populate_test_sample_plots(baseline)
        if transfer:
            self.append_log(f"Self-transfer run saved to {transfer['run_dir']}")
        if baseline and transfer:
            self.append_log("Baseline training and self-transfer learning completed.")
        elif baseline:
            self.append_log("Baseline training completed.")
        elif transfer:
            self.append_log("Self-transfer learning completed.")
        # Switch to Test Samples tab if we have test plots, otherwise show the training monitor.
        if baseline and baseline.get("test_sample_data"):
            self.monitor_tabs.setCurrentIndex(2)  # Test Samples tab
        elif transfer is not None:
            self.monitor_tabs.setCurrentIndex(1)
        else:
            self.monitor_tabs.setCurrentIndex(0)
        self.validate_transfer_compatibility()

    def _update_scan_progress(self, payload: dict[str, Any]) -> None:
        event = payload.get("event")
        self.run_state_badge.set_status("Scanning")
        self.metric_cards["current_phase"].set_value("Scan")

        stage_labels = {
            "started": "Starting",
            "schema_ready": "Schema Ready",
            "input_rows_loaded": "Input Rows Ready",
            "cache_build_started": "Building Cache",
            "cache_overwrite_started": "Rebuilding Cache",
            "cache_progress": "Building Cache",
            "cache_existing": "Using Existing Cache",
            "cache_write_started": "Saving Cache",
            "cache_saved": "Cache Saved",
            "split_loading_started": "Preparing Splits",
            "completed": "Completed",
        }
        self.metric_cards["current_progress"].set_value(stage_labels.get(str(event), "Scanning"))

        if event == "started":
            self.run_progress_bar.setValue(0)
            self.metric_cards["secondary_progress"].set_value("Preparing dataset paths")
            return

        if event == "schema_ready":
            self.run_progress_bar.setValue(max(self.run_progress_bar.value(), 5))
            dataset_name = payload.get("dataset_name")
            if dataset_name:
                self.metric_cards["secondary_progress"].set_value(str(dataset_name))
            return

        if event == "input_rows_loaded":
            total_samples = payload.get("total_samples")
            self.run_progress_bar.setValue(max(self.run_progress_bar.value(), 10))
            if total_samples is not None:
                self.metric_cards["secondary_progress"].set_value(f"{int(total_samples)} input rows loaded")
            return

        if event == "cache_build_started":
            self.run_progress_bar.setValue(max(self.run_progress_bar.value(), 15))
            total_samples = payload.get("total_samples")
            if total_samples is not None:
                self.metric_cards["secondary_progress"].set_value(f"0/{int(total_samples)} ground-truth files")
            return

        if event == "cache_overwrite_started":
            self.run_progress_bar.setValue(max(self.run_progress_bar.value(), 12))
            cache_path = payload.get("cache_path")
            if cache_path:
                cache_name = Path(str(cache_path)).name
                self.metric_cards["secondary_progress"].set_value(cache_name)
            return

        if event == "cache_progress":
            completed_samples = payload.get("completed_samples")
            total_samples = payload.get("total_samples")
            if completed_samples is not None and total_samples:
                scan_progress = 15 + int(70.0 * int(completed_samples) / max(int(total_samples), 1))
                self.run_progress_bar.setValue(max(self.run_progress_bar.value(), min(scan_progress, 85)))
                self.metric_cards["secondary_progress"].set_value(f"{int(completed_samples)}/{int(total_samples)} ground-truth files")
            return

        if event == "cache_existing":
            self.run_progress_bar.setValue(max(self.run_progress_bar.value(), 80))
            total_samples = payload.get("total_samples")
            frequency_count = payload.get("frequency_count")
            if total_samples is not None and frequency_count is not None:
                self.metric_cards["secondary_progress"].set_value(f"{int(total_samples)} samples | {int(frequency_count)} freq")
            return

        if event == "cache_write_started":
            self.run_progress_bar.setValue(max(self.run_progress_bar.value(), 90))
            cache_path = payload.get("cache_path")
            if cache_path:
                self.metric_cards["secondary_progress"].set_value(Path(str(cache_path)).name)
            return

        if event == "cache_saved":
            self.run_progress_bar.setValue(max(self.run_progress_bar.value(), 95))
            cache_path = payload.get("cache_path")
            if cache_path:
                self.metric_cards["secondary_progress"].set_value(Path(str(cache_path)).name)
            return

        if event == "split_loading_started":
            self.run_progress_bar.setValue(max(self.run_progress_bar.value(), 97))
            self.metric_cards["secondary_progress"].set_value("Preparing train / val / test loaders")
            return

        if event == "completed":
            self.run_progress_bar.setValue(100)
            dataset_name = payload.get("dataset_name")
            frequency_count = payload.get("frequency_count")
            if dataset_name and frequency_count is not None:
                self.metric_cards["secondary_progress"].set_value(f"{dataset_name} | {int(frequency_count)} freq")

    def _update_suggest_progress(self, payload: dict[str, Any]) -> None:
        event = payload.get("event")
        self.run_state_badge.set_status("Suggesting")
        progress_by_event = {
            "started": 5,
            "cache_ready": 25,
            "diagnostics_started": 45,
            "recommendation_started": 70,
            "diagnostics_ready": 85,
            "completed": 100,
        }
        self.run_progress_bar.setValue(progress_by_event.get(event, self.run_progress_bar.value()))
        self.metric_cards["current_phase"].set_value("Suggest")
        self.metric_cards["current_progress"].set_value(event.replace("_", " ").title() if isinstance(event, str) else "-")
        if event == "diagnostics_ready":
            diagnostics = payload.get("diagnostics", {})
            self.metric_cards["secondary_progress"].set_value(f"{diagnostics.get('num_samples_train', '-')} train samples")

    def _update_search_progress(self, payload: dict[str, Any]) -> None:
        if not hasattr(self, "search_results_table"):
            return
        event = payload.get("event")
        trial_index = int(payload.get("trial_index", 0)) if payload.get("trial_index") is not None else 0
        trial_count = int(payload.get("trial_count", 0)) if payload.get("trial_count") is not None else 0
        self.run_state_badge.set_status("Searching")
        self.metric_cards["current_phase"].set_value("Quick Search")
        if trial_index and trial_count:
            self.metric_cards["current_progress"].set_value(f"Trial {trial_index}/{trial_count}")
        if payload.get("trial_label"):
            self.metric_cards["secondary_progress"].set_value(str(payload["trial_label"]))

        if event == "trial_started" and trial_index and trial_count:
            self._set_search_trial_progress(trial_index, trial_count, 0.0)
            self.metric_cards["secondary_progress"].set_value(f"{payload.get('trial_label', 'trial')} | Starting")
            return

        if payload.get("phase") == "baseline" and trial_index:
            label = str(payload.get("trial_label", "trial"))
            if event == "started":
                self._set_search_trial_progress(trial_index, trial_count, 0.02)
                self.metric_cards["secondary_progress"].set_value(f"{label} | Baseline setup")
                return
            if event == "cache_check_started":
                self._set_search_trial_progress(trial_index, trial_count, 0.06)
                self.metric_cards["secondary_progress"].set_value(f"{label} | Checking cache")
                return
            if event == "cache_ready":
                self._set_search_trial_progress(trial_index, trial_count, 0.12)
                self.metric_cards["secondary_progress"].set_value(f"{label} | Cache ready")
                return
            if event == "split_loading_started":
                self._set_search_trial_progress(trial_index, trial_count, 0.18)
                self.metric_cards["secondary_progress"].set_value(f"{label} | Loading split")
                return
            if event == "data_ready":
                self._set_search_trial_progress(trial_index, trial_count, 0.26)
                train_samples = payload.get("train_samples")
                val_samples = payload.get("val_samples")
                if None not in (train_samples, val_samples):
                    self.metric_cards["secondary_progress"].set_value(f"{label} | {train_samples} train / {val_samples} val")
                else:
                    self.metric_cards["secondary_progress"].set_value(f"{label} | Data ready")
                return
            if event == "model_ready":
                self._set_search_trial_progress(trial_index, trial_count, 0.32)
                parameter_count = payload.get("parameter_count")
                if parameter_count is not None:
                    self.metric_cards["secondary_progress"].set_value(f"{label} | Model ready ({int(parameter_count):,} params)")
                else:
                    self.metric_cards["secondary_progress"].set_value(f"{label} | Model ready")
                return
            if event == "epoch_end":
                epoch = int(payload["epoch"])
                total_epochs = int(payload["total_epochs"])
                self._set_search_trial_progress(trial_index, trial_count, 0.32 + 0.63 * epoch / max(total_epochs, 1))
                self.metric_cards["secondary_progress"].set_value(f"{label} | Epoch {epoch}/{total_epochs}")
                self.metric_cards["train_loss"].set_value(f"{payload['train_loss']:.6f}")
                self.metric_cards["validation_loss"].set_value(f"{payload['val_loss']:.6f}")
                self.metric_cards["best_metric"].set_value(f"Best val {payload['best_val_loss']:.6f}")
                self.metric_cards["elapsed"].set_value(self._format_seconds(payload.get("elapsed_seconds")))
                self.metric_cards["eta"].set_value(self._format_seconds(payload.get("eta_seconds")))
                return
            if event == "evaluation_completed":
                self._set_search_trial_progress(trial_index, trial_count, 0.97)
                average_mae = payload.get("average_evaluation_mae", payload.get("average_val_mae"))
                if average_mae is not None:
                    self.metric_cards["average_mae"].set_value(f"{average_mae:.6f}")
                self.metric_cards["secondary_progress"].set_value(f"{label} | Evaluating")
                return
            if event == "completed":
                self._set_search_trial_progress(trial_index, trial_count, 1.0)
                return
        elif event == "trial_completed":
            self._set_search_trial_progress(trial_index, trial_count, 1.0)
            self.metric_cards["best_metric"].set_value(f"Val {payload['best_val_loss']:.6f}")
            self.metric_cards["average_mae"].set_value(f"{payload['average_val_mae']:.6f}")

    def _set_search_trial_progress(self, trial_index: int, trial_count: int, within_trial: float) -> None:
        if trial_index <= 0 or trial_count <= 0:
            return
        overall_progress = ((trial_index - 1) + min(max(within_trial, 0.0), 1.0)) / max(trial_count, 1)
        self.run_progress_bar.setValue(int(100.0 * overall_progress))

    def _update_baseline_progress(self, payload: dict[str, Any]) -> None:
        event = payload.get("event")
        if payload.get("trial_index") is not None:
            return
        self.run_state_badge.set_status("Training")
        self.metric_cards["current_phase"].set_value("Baseline")
        if event == "started":
            self.run_progress_bar.setValue(0)
            self.metric_cards["current_progress"].set_value("Starting")
        elif event == "data_ready":
            train_samples = payload.get("train_samples")
            val_samples = payload.get("val_samples")
            test_samples = payload.get("test_samples")
            if None not in (train_samples, val_samples, test_samples):
                self.metric_cards["secondary_progress"].set_value(f"{train_samples} train / {val_samples} val / {test_samples} test")
            else:
                self.metric_cards["secondary_progress"].set_value("Data ready")
        elif event == "epoch_end":
            if None in (payload.get("epoch"), payload.get("total_epochs"), payload.get("train_loss"), payload.get("val_loss")):
                self.metric_cards["current_progress"].set_value("Epoch completed")
                return
            epoch = int(payload["epoch"])
            total_epochs = int(payload["total_epochs"])
            self._baseline_epochs.append(epoch)
            self._baseline_train_losses.append(float(payload["train_loss"]))
            self._baseline_val_losses.append(float(payload["val_loss"]))
            self.baseline_train_curve.setData(self._baseline_epochs, self._baseline_train_losses)
            self.baseline_val_curve.setData(self._baseline_epochs, self._baseline_val_losses)
            self.metric_cards["current_progress"].set_value(f"Epoch {epoch}/{total_epochs}")
            self.metric_cards["secondary_progress"].set_value(f"Best epoch {payload['best_epoch']}")
            self.metric_cards["best_metric"].set_value(f"Best val {payload['best_val_loss']:.6f}")
            self.metric_cards["train_loss"].set_value(f"{payload['train_loss']:.6f}")
            self.metric_cards["validation_loss"].set_value(f"{payload['val_loss']:.6f}")
            self.metric_cards["elapsed"].set_value(self._format_seconds(payload.get("elapsed_seconds")))
            self.metric_cards["eta"].set_value(self._format_seconds(payload.get("eta_seconds")))
            self.run_progress_bar.setValue(int(100.0 * epoch / max(total_epochs, 1)))
        elif event == "evaluation_completed":
            self.baseline_frequency_curve.setData(payload.get("frequency_ghz", []), payload.get("frequency_mae", []))
            channel_labels = payload.get("channel_mae_with_units")
            if channel_labels:
                self.metric_cards["average_mae"].set_value(" | ".join(channel_labels))
            else:
                average_mae = payload.get("average_evaluation_mae", payload.get("average_test_mae"))
                if average_mae is not None:
                    self.metric_cards["average_mae"].set_value(f"{average_mae:.6f}")
        elif event == "completed":
            self.run_progress_bar.setValue(100)
            self.run_state_badge.set_status("Completed")
        elif event == "stopped":
            self.run_state_badge.set_status("Stopped")

    def _update_transfer_progress(self, payload: dict[str, Any]) -> None:
        event = payload.get("event")
        self.run_state_badge.set_status("Transfer")
        self.metric_cards["current_phase"].set_value("Transfer")
        if event == "data_ready":
            self.metric_cards["current_progress"].set_value("Preparing")
            train_samples = payload.get("train_samples")
            test_samples = payload.get("test_samples")
            effective_frequency_count = payload.get("effective_frequency_count", payload.get("frequency_count"))
            trimmed_frequency_count = int(payload.get("trimmed_frequency_count", 0) or 0)
            if None not in (train_samples, test_samples):
                progress_text = f"{train_samples} train / {test_samples} test"
                if effective_frequency_count is not None:
                    progress_text += f" | {effective_frequency_count} freq"
                    if trimmed_frequency_count > 0:
                        progress_text += f" ({trimmed_frequency_count} trimmed)"
                self.metric_cards["secondary_progress"].set_value(progress_text)
            else:
                self.metric_cards["secondary_progress"].set_value("Transfer data ready")
        elif event == "baseline_metrics_ready":
            self.last_transfer_base_summary = payload
            base_average = payload.get("base_average_mae", payload.get("average_mae"))
            if base_average is not None:
                self.metric_cards["best_metric"].set_value(f"Base MAE {base_average:.6f}")
                self.metric_cards["average_mae"].set_value(f"{base_average:.6f}")
            self._plot_transfer_base_metrics(payload)
        elif event == "iteration_started":
            if payload.get("transfer_iteration") is not None and payload.get("total_iterations") is not None:
                self.metric_cards["current_progress"].set_value(f"Iteration {payload['transfer_iteration']}/{payload['total_iterations']}")
            else:
                self.metric_cards["current_progress"].set_value("Iteration started")
        elif event == "band_started":
            self.metric_cards["secondary_progress"].set_value(self._transfer_band_context(payload))
        elif event == "band_epoch_end":
            if None in (payload.get("epoch"), payload.get("total_epochs"), payload.get("train_loss")):
                self.metric_cards["current_progress"].set_value("Band epoch completed")
                return
            epoch = int(payload["epoch"])
            total_epochs = int(payload["total_epochs"])
            self.metric_cards["current_progress"].set_value(f"T={payload.get('transfer_iteration', 0)} | Band epoch {epoch}/{total_epochs}")
            self.metric_cards["secondary_progress"].set_value(self._transfer_band_context(payload))
            self.metric_cards["train_loss"].set_value(f"{payload['train_loss']:.6f}")
            if payload.get("elapsed_seconds") is not None:
                self.metric_cards["elapsed"].set_value(self._format_seconds(payload.get("elapsed_seconds")))
            if payload.get("eta_seconds") is not None:
                self.metric_cards["eta"].set_value(self._format_seconds(payload.get("eta_seconds")))
            total_band_runs = int(payload.get("total_band_runs", 0))
            band_run_index = int(payload.get("band_run_index", 0))
            if total_band_runs:
                base_progress = (band_run_index - 1) / max(total_band_runs, 1)
                within_band = epoch / max(total_epochs, 1) / max(total_band_runs, 1)
                self.run_progress_bar.setValue(int(100.0 * min(base_progress + within_band, 1.0)))
        elif event == "iteration_completed":
            if None in (payload.get("transfer_iteration"), payload.get("average_mae"), payload.get("total_iterations")):
                self.metric_cards["current_progress"].set_value("Iteration completed")
                return
            transfer_iteration = int(payload["transfer_iteration"])
            self._transfer_iteration_mae_x.append(transfer_iteration)
            self._transfer_iteration_mae_y.append(float(payload["average_mae"]))
            self._transfer_frequency_history.append(list(payload.get("frequency_mae", [])))
            self._plot_transfer_iteration_metrics(payload)
            self.metric_cards["average_mae"].set_value(f"{payload['average_mae']:.6f}")
            self.metric_cards["current_progress"].set_value(f"Iteration {transfer_iteration}/{payload['total_iterations']} complete")
            if payload.get("elapsed_seconds") is not None:
                self.metric_cards["elapsed"].set_value(self._format_seconds(payload.get("elapsed_seconds")))
            if payload.get("eta_seconds") is not None:
                self.metric_cards["eta"].set_value(self._format_seconds(payload.get("eta_seconds")))
            self.run_progress_bar.setValue(int(100.0 * transfer_iteration / max(int(payload["total_iterations"]), 1)))
        elif event == "completed":
            self.run_progress_bar.setValue(100)
            final_average = payload.get("final_average_mae", payload.get("average_mae"))
            if final_average is not None:
                self.metric_cards["best_metric"].set_value(f"Final MAE {final_average:.6f}")
            if payload.get("elapsed_seconds") is not None:
                self.metric_cards["elapsed"].set_value(self._format_seconds(payload.get("elapsed_seconds")))
            if payload.get("eta_seconds") is not None:
                self.metric_cards["eta"].set_value(self._format_seconds(payload.get("eta_seconds")))
            self.run_state_badge.set_status("Completed")
        elif event == "stopped":
            if payload.get("elapsed_seconds") is not None:
                self.metric_cards["elapsed"].set_value(self._format_seconds(payload.get("elapsed_seconds")))
            if payload.get("eta_seconds") is not None:
                self.metric_cards["eta"].set_value(self._format_seconds(payload.get("eta_seconds")))
            self.run_state_badge.set_status("Stopped")

    # ------------------------------------------------------------------
    # Form helpers
    # ------------------------------------------------------------------
    def _collect_baseline_form(self) -> dict[str, Any]:
        return {
            "model_type": self.baseline_model_type_combo_box.currentText(),
            "epochs": self.baseline_epochs_spin_box.value(),
            "batch_size": self.baseline_batch_size_spin_box.value(),
            "learning_rate": float(self.baseline_learning_rate_spin_box.value()),
            "weight_decay": float(self.baseline_weight_decay_spin_box.value()),
            "width": self.baseline_width_spin_box.value(),
            "depth": self.baseline_depth_spin_box.value(),
            "loss_function": self.baseline_loss_function_combo_box.currentText(),
            "scheduler": self.baseline_scheduler_combo_box.currentText(),
            "train_frac": float(self.baseline_train_fraction_spin_box.value()),
            "val_frac": float(self.baseline_validation_fraction_spin_box.value()),
            "seed": self.baseline_seed_spin_box.value(),
        }

    def _apply_baseline_form(self, payload: dict[str, Any]) -> None:
        if not payload:
            return
        model_type = payload.get("model_type", "FlatMLP")
        if model_type in MODEL_TYPES:
            self.baseline_model_type_combo_box.setCurrentText(model_type)
        self.baseline_epochs_spin_box.setValue(int(payload.get("epochs", self.baseline_epochs_spin_box.value())))
        self.baseline_batch_size_spin_box.setValue(int(payload.get("batch_size", self.baseline_batch_size_spin_box.value())))
        self.baseline_learning_rate_spin_box.setValue(float(payload.get("learning_rate", self.baseline_learning_rate_spin_box.value())))
        self.baseline_weight_decay_spin_box.setValue(float(payload.get("weight_decay", self.baseline_weight_decay_spin_box.value())))
        self.baseline_width_spin_box.setValue(int(payload.get("width", self.baseline_width_spin_box.value())))
        self.baseline_depth_spin_box.setValue(int(payload.get("depth", self.baseline_depth_spin_box.value())))
        loss_fn = payload.get("loss_function", "rmse")
        if loss_fn in LOSS_FUNCTIONS:
            self.baseline_loss_function_combo_box.setCurrentText(loss_fn)
        sched = payload.get("scheduler", "plateau")
        if sched in SCHEDULER_TYPES:
            self.baseline_scheduler_combo_box.setCurrentText(sched)
        self.baseline_train_fraction_spin_box.setValue(float(payload.get("train_frac", self.baseline_train_fraction_spin_box.value())))
        self.baseline_validation_fraction_spin_box.setValue(float(payload.get("val_frac", self.baseline_validation_fraction_spin_box.value())))
        self.baseline_seed_spin_box.setValue(int(payload.get("seed", self.baseline_seed_spin_box.value())))

    def _collect_transfer_form(self) -> dict[str, Any]:
        return {
            "num_bands": self.transfer_num_bands_spin_box.value(),
            "iterations": self.transfer_iterations_spin_box.value(),
            "transfer_epochs": self.transfer_epochs_spin_box.value(),
            "batch_size": self.transfer_batch_size_spin_box.value(),
            "learning_rate": float(self.transfer_learning_rate_spin_box.value()),
            "weight_decay": None if self.transfer_weight_decay_spin_box.value() <= 0.0 else float(self.transfer_weight_decay_spin_box.value()),
            "seed": self.transfer_seed_spin_box.value(),
        }

    def _apply_transfer_form(self, payload: dict[str, Any]) -> None:
        if not payload:
            return
        self.transfer_num_bands_spin_box.setValue(int(payload.get("num_bands", self.transfer_num_bands_spin_box.value())))
        self.transfer_iterations_spin_box.setValue(int(payload.get("iterations", self.transfer_iterations_spin_box.value())))
        self.transfer_epochs_spin_box.setValue(int(payload.get("transfer_epochs", self.transfer_epochs_spin_box.value())))
        self.transfer_batch_size_spin_box.setValue(int(payload.get("batch_size", self.transfer_batch_size_spin_box.value())))
        self.transfer_learning_rate_spin_box.setValue(float(payload.get("learning_rate", self.transfer_learning_rate_spin_box.value())))
        weight_decay = payload.get("weight_decay", None)
        self.transfer_weight_decay_spin_box.setValue(0.0 if weight_decay in (None, "") else float(weight_decay))
        self.transfer_seed_spin_box.setValue(int(payload.get("seed", self.transfer_seed_spin_box.value())))
        self._refresh_transfer_note_text()

    def _build_baseline_train_config(self) -> TrainConfig:
        roots = make_run_roots(self.model_output_folder_path_edit.text().strip(), self.run_name_edit.text().strip())
        form = self._collect_baseline_form()
        dataset_folder = self.dataset_folder_edit.text().strip()
        input_feat = self.input_feature_path_edit.text().strip()
        gt_dir = self.ground_truth_data_folder_path_edit.text().strip()
        # When the new single-folder field is set, use it as data_root.
        # Otherwise fall back to legacy fields.
        if dataset_folder:
            data_root = dataset_folder
            input_feat = ""
            gt_dir = ""
        else:
            data_root = gt_dir if not input_feat else None
        return TrainConfig(
            data_root=data_root,
            input_feature_path=input_feat or None,
            ground_truth_data_dir=gt_dir if input_feat else None,
            cache_path=self._ensure_cache_path(),
            output_dir=roots["baseline"],
            model_type=form["model_type"],
            seed=form["seed"],
            batch_size=form["batch_size"],
            epochs=form["epochs"],
            learning_rate=form["learning_rate"],
            weight_decay=form["weight_decay"],
            train_frac=form["train_frac"],
            val_frac=form["val_frac"],
            width=form["width"],
            depth=form["depth"],
            loss_function=form["loss_function"],
            scheduler=form["scheduler"],
            use_amp=form.get("use_amp", True),
            max_samples=None,
            device=self._current_device_id(),
        )

    def _build_transfer_config(self) -> TransferConfig:
        roots = make_run_roots(self.model_output_folder_path_edit.text().strip(), self.run_name_edit.text().strip())
        form = self._collect_transfer_form()
        base_run_dir = self.transfer_base_run_path_edit.text().strip() if self.transfer_base_model_source_combo_box.currentIndex() == 1 else ""
        return TransferConfig(
            base_run_dir=base_run_dir,
            cache_path=self._ensure_cache_path(),
            output_dir=roots["transfer"],
            seed=form["seed"],
            num_bands=form["num_bands"],
            iterations=form["iterations"],
            transfer_epochs=form["transfer_epochs"],
            batch_size=form["batch_size"],
            learning_rate=form["learning_rate"],
            weight_decay=form["weight_decay"],
            use_amp=True,
            device=self._current_device_id(),
        )

    def _build_search_config(self) -> SearchConfig:
        raise RuntimeError("Quick search has been removed from the GUI.")

    def _on_search_max_samples_changed(self, value: int) -> None:
        return

    def _set_search_max_samples_value(self, value: int, *, autofill: bool) -> None:
        return

    def _apply_search_max_samples_default(self, total_samples: int) -> None:
        return

    # ------------------------------------------------------------------
    # UI state helpers
    # ------------------------------------------------------------------
    def _refresh_search_controls_enabled(self) -> None:
        return

    def _refresh_transfer_controls_enabled(self) -> None:
        enabled = self.enable_transfer_learning_checkbox.isChecked() and not self._controls_locked
        for widget in (
            self.transfer_base_model_source_combo_box,
            self.transfer_base_run_path_edit,
            self.browse_transfer_base_run_button,
            self.transfer_num_bands_spin_box,
            self.transfer_iterations_spin_box,
            self.transfer_epochs_spin_box,
            self.transfer_batch_size_spin_box,
            self.transfer_learning_rate_spin_box,
            self.transfer_weight_decay_spin_box,
            self.transfer_seed_spin_box,
        ):
            widget.setEnabled(enabled)
        if not enabled:
            self.transfer_compatibility_status_badge.set_status("Not Checked")
            self.transfer_notes_label.setText("Run baseline training first, or choose an existing baseline run.")
        self._refresh_transfer_source_controls()

    def _refresh_transfer_source_controls(self) -> None:
        using_existing = self.transfer_base_model_source_combo_box.currentIndex() == 1
        self.transfer_base_run_path_edit.setEnabled(using_existing and self.enable_transfer_learning_checkbox.isChecked() and not self._controls_locked)
        self.browse_transfer_base_run_button.setEnabled(using_existing and self.enable_transfer_learning_checkbox.isChecked() and not self._controls_locked)
        if not using_existing and self.enable_transfer_learning_checkbox.isChecked():
            if self._current_baseline_run_dir():
                self.transfer_compatibility_status_badge.set_status("Compatible")
                self.transfer_notes_label.setText("Self-transfer will use the baseline produced in this session.")
            else:
                self.transfer_compatibility_status_badge.set_status("Not Checked")
                self.transfer_notes_label.setText("Run baseline training first, or choose an existing baseline run.")
        elif not self.enable_transfer_learning_checkbox.isChecked():
            self.transfer_compatibility_status_badge.set_status("Not Checked")
            self.transfer_notes_label.setText("Run baseline training first, or choose an existing baseline run.")
        else:
            self.validate_transfer_compatibility()

    def _refresh_transfer_note_text(self) -> None:
        if self.last_scan_result is None:
            return
        frequency_count = int(self.last_scan_result["frequency_count"])
        num_bands = self.transfer_num_bands_spin_box.value()
        if num_bands <= 0:
            return
        if num_bands > frequency_count:
            points_per_band = 0
            suffix = "This band count is not valid because it exceeds the detected frequency count."
        elif frequency_count % num_bands == 0:
            points_per_band = frequency_count // num_bands
            suffix = "Compatible split. No frequency points will be trimmed."
        else:
            trimmed_frequency_count = frequency_count % num_bands
            points_per_band = (frequency_count - trimmed_frequency_count) // num_bands
            suffix = (
                f"The last {trimmed_frequency_count} frequency point(s) will be discarded during "
                f"transfer so each band uses {points_per_band} points."
            )
        if self.transfer_base_model_source_combo_box.currentIndex() == 0 and self.enable_transfer_learning_checkbox.isChecked():
            prefix = (
                "Self-transfer will use the baseline from this session."
                if self._current_baseline_run_dir()
                else "Run baseline training first, or choose an existing baseline run."
            )
        elif self.enable_transfer_learning_checkbox.isChecked():
            prefix = self.transfer_notes_label.text().split(". ")[0] + "."
        else:
            prefix = "Run baseline training first, or choose an existing baseline run."
        self.transfer_notes_label.setText(f"{prefix} {frequency_count} frequency points detected, about {points_per_band:.1f} per band. {suffix}")

    def _update_start_button_text(self) -> None:
        return

    def _refresh_run_button_availability(self) -> None:
        start_enabled = (not self._controls_locked) and self._license_allows_new_runs()
        self.start_baseline_button.setEnabled(start_enabled)
        self.start_transfer_button.setEnabled(start_enabled)

    def _set_action_controls_enabled(self, enabled: bool) -> None:
        widgets = [
            self.training_device_combo_box,
            self.suggest_initial_settings_button,
            self.start_baseline_button,
            self.start_transfer_button,
            self.open_output_folder_button,
            self.export_run_summary_button,
            self.export_onnx_button,
            self.apply_initial_settings_button,
            self.restore_recommended_baseline_button,
            self.ctle_preset_button,
            self.input_feature_path_edit,
            self.browse_input_feature_button,
            self.ground_truth_data_folder_path_edit,
            self.browse_ground_truth_data_folder_button,
            self.model_output_folder_path_edit,
            self.browse_model_output_folder_button,
            self.run_name_edit,
            self.cache_path_edit,
            self.browse_cache_button,
            self.training_tabs,
            self.browse_transfer_base_run_button,
            self.license_server_url_edit,
            self.test_license_connection_button,
            self.acquire_license_seat_button,
        ]
        self._controls_locked = not enabled
        for widget in widgets:
            widget.setEnabled(enabled)
        self._refresh_transfer_controls_enabled()
        self._refresh_run_button_availability()

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------
    def _reset_baseline_plots(self) -> None:
        self._baseline_epochs.clear()
        self._baseline_train_losses.clear()
        self._baseline_val_losses.clear()
        self.baseline_loss_plot.clear()
        self.baseline_frequency_mae_plot.clear()
        configure_plot_widget(self.baseline_loss_plot, title="Training Loss vs Epoch", x_label="Epoch", y_label="Loss")
        sweep = getattr(self, "_sweep_label", "Frequency (GHz)")
        configure_plot_widget(self.baseline_frequency_mae_plot, title=f"MAE over {sweep}", x_label=sweep, y_label="MAE")
        colors = plot_color_cycle()
        self.baseline_train_curve = self.baseline_loss_plot.plot([], [], pen=pg.mkPen(colors[0], width=2.5), name="Train")
        self.baseline_val_curve = self.baseline_loss_plot.plot([], [], pen=pg.mkPen(colors[1], width=2.5), name="Validation")
        self.baseline_frequency_curve = self.baseline_frequency_mae_plot.plot([], [], pen=pg.mkPen(colors[2], width=2.5), name="MAE")

    def _reset_search_plot(self) -> None:
        self.search_tradeoff_plot.clear()
        configure_plot_widget(self.search_tradeoff_plot, title="Accuracy vs Runtime", x_label="Runtime (s)", y_label="Average Validation MAE")
        self._search_text_items.clear()

    def _reset_transfer_plots(self) -> None:
        self._transfer_iteration_mae_x.clear()
        self._transfer_iteration_mae_y.clear()
        self._transfer_frequency_history.clear()
        self._transfer_frequency_items.clear()
        self.transfer_frequency_mae_plot.clear()
        sweep = getattr(self, "_sweep_label", "Frequency (GHz)")
        configure_plot_widget(self.transfer_frequency_mae_plot, title=f"MAE over {sweep} by Transfer Iteration", x_label=sweep, y_label="MAE")

    def _plot_search_tradeoff(self, trials: list[dict[str, Any]]) -> None:
        self._reset_search_plot()
        colors = plot_color_cycle()
        scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen(APP_THEME.text, width=0.6), brush=pg.mkBrush(colors[0]))
        scatter.addPoints([{"pos": (trial["runtime_seconds"], trial["average_val_mae"])} for trial in trials])
        self.search_tradeoff_plot.addItem(scatter)
        for trial in trials:
            label = pg.TextItem(text=str(trial["rank"]), color=APP_THEME.text, anchor=(0, 1))
            label.setPos(trial["runtime_seconds"], trial["average_val_mae"])
            self.search_tradeoff_plot.addItem(label)
            self._search_text_items.append(label)

    def _plot_transfer_base_metrics(self, payload: dict[str, Any]) -> None:
        colors = plot_color_cycle()
        if payload.get("frequency_ghz") and payload.get("base_frequency_mae"):
            item = self.transfer_frequency_mae_plot.plot(
                payload["frequency_ghz"],
                payload["base_frequency_mae"],
                pen=pg.mkPen(colors[4], width=2.0, style=Qt.PenStyle.DashLine),
                name="Base",
            )
            self._transfer_frequency_items.append(item)

    def _plot_transfer_iteration_metrics(self, payload: dict[str, Any]) -> None:
        colors = plot_color_cycle()
        index = max(len(self._transfer_frequency_history) - 1, 0)
        color = colors[index % len(colors)]
        if payload.get("frequency_ghz") and payload.get("frequency_mae"):
            item = self.transfer_frequency_mae_plot.plot(
                payload["frequency_ghz"],
                payload["frequency_mae"],
                pen=pg.mkPen(color, width=2.2),
                name=f"T={payload['transfer_iteration']}",
            )
            self._transfer_frequency_items.append(item)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def _section_header(self, title: str, subtitle: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("SectionTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("SectionSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        return widget

    def _make_button(self, text: str, *, secondary: bool = False) -> QPushButton:
        button = QPushButton(text)
        if secondary:
            button.setProperty("secondary", "true")
            button.style().unpolish(button)
            button.style().polish(button)
        return button

    def _make_int_spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = _NoScrollSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _make_float_spin(
        self,
        minimum: float,
        maximum: float,
        value: float,
        *,
        decimals: int,
        step: float,
        scientific: bool = False,
    ) -> QDoubleSpinBox:
        spin = _NoScrollDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        if scientific:
            spin.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
        return spin

    def _make_table(self, headers: list[str], *, stretch_last: bool) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = table.horizontalHeader()
        header.setStretchLastSection(stretch_last)
        if not stretch_last:
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        return table

    def _add_form_row(self, layout: QGridLayout, row: int, label_text: str, widget: QWidget) -> None:
        layout.addWidget(QLabel(label_text), row, 0)
        layout.addWidget(widget, row, 1, 1, 2)

    def _add_path_row(self, layout: QGridLayout, row: int, label_text: str, edit: QLineEdit, button: QPushButton) -> None:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        edit.setMinimumWidth(0)
        edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.setMinimumWidth(112)
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        row_layout.addWidget(edit, 1)
        row_layout.addWidget(button, 0)
        layout.addWidget(QLabel(label_text), row, 0)
        layout.addWidget(row_widget, row, 1, 1, 2)

    def _fill_table(self, table: QTableWidget, rows: list[tuple[Any, ...]]) -> None:
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for col_index, value in enumerate(row):
                item = QTableWidgetItem(self._stringify(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(row_index, col_index, item)
        table.resizeRowsToContents()

    def _stringify(self, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.6f}" if abs(value) < 1000 else f"{value:.3f}"
        if isinstance(value, (list, tuple)):
            return ", ".join(self._stringify(item) for item in value)
        return str(value)

    def _prettify_key(self, key: str) -> str:
        return key.replace("_", " ").title()

    def _suggestion_rows(self, result: dict[str, Any]) -> list[tuple[str, str, str, str]]:
        rows: list[tuple[str, str, str, str]] = []
        baseline = result["suggested_baseline_config"]
        baseline_ranges = result["suggested_baseline_ranges"]
        baseline_rationale = result["baseline_rationale"]
        for name in ["model_type", "width", "depth", "batch_size", "learning_rate", "weight_decay", "epochs"]:
            range_info = baseline_ranges.get(name, {})
            rows.append((f"Baseline: {self._prettify_key(name)}", self._stringify(baseline[name]), self._stringify(range_info.get("candidates", [])), baseline_rationale.get(name, "")))
        transfer = result["suggested_transfer_config"]
        transfer_ranges = result["suggested_transfer_ranges"]
        transfer_rationale = result["transfer_rationale"]
        for name in ["num_bands", "iterations", "transfer_epochs", "batch_size", "learning_rate"]:
            range_info = transfer_ranges.get(name, {})
            rows.append((f"Transfer: {self._prettify_key(name)}", self._stringify(transfer[name]), self._stringify(range_info.get("candidates", [])), transfer_rationale.get(name, "")))
        return rows

    def _search_row(self, trial: dict[str, Any]) -> tuple[str, ...]:
        config = trial["recommended_full_config"]
        return (
            str(trial["rank"]),
            str(config["width"]),
            str(config["depth"]),
            f"{config['learning_rate']:.1e}",
            str(config["batch_size"]),
            f"{trial['best_val_loss']:.6f}",
            f"{trial['average_val_mae']:.6f}",
            self._format_seconds(trial["runtime_seconds"]),
            f"{trial['score']:.4f}",
        )

    def _on_search_selection_changed(self) -> None:
        row = self.search_results_table.currentRow()
        self.selected_search_full_config = self.search_row_configs[row] if 0 <= row < len(self.search_row_configs) else None

    def _focus_training_settings(self) -> None:
        self.training_tabs.setCurrentIndex(0)
        self.monitor_tabs.setCurrentIndex(0)

    def _restore_recommended_baseline(self) -> None:
        if self.selected_search_full_config is not None:
            self._apply_baseline_form(self.selected_search_full_config)
            self.append_log("Restored the selected quick-search baseline configuration.")
            return
        if self.last_suggest_result is not None:
            self._apply_baseline_form(self.last_suggest_result["suggested_baseline_config"])
            self.append_log("Restored the scan-only baseline recommendation.")
            return
        self._show_warning("No recommended baseline configuration is available yet.")

    def _apply_ctle_preset(self) -> None:
        from .runner import TrainConfig
        config = TrainConfig.preset("CTLE")
        self._apply_baseline_form(asdict(config))
        self.append_log("Applied CTLE preset (notebook-validated defaults).")

    def _validate_split_fractions(self) -> bool:
        train_frac = float(self.baseline_train_fraction_spin_box.value())
        val_frac = float(self.baseline_validation_fraction_spin_box.value())
        if not 0.0 < train_frac < 1.0:
            self._show_warning("Train fraction must be between 0 and 1.")
            return False
        if not 0.0 <= val_frac < 1.0:
            self._show_warning("Validation fraction must be between 0 and 1.")
            return False
        if train_frac + val_frac >= 1.0:
            self._show_warning("Train fraction plus validation fraction must leave room for a test split.")
            return False
        return True

    def _validate_transfer_ready(self) -> bool:
        if self.last_scan_result is None:
            self._show_warning("Scan the dataset before starting self-transfer learning.")
            return False
        frequency_count = int(self.last_scan_result["frequency_count"])
        num_bands = self.transfer_num_bands_spin_box.value()
        if num_bands > frequency_count:
            self._show_warning(
                f"The selected number of frequency bands ({num_bands}) exceeds the detected frequency count ({frequency_count})."
            )
            return False
        if self.transfer_base_model_source_combo_box.currentIndex() == 0 and self._current_baseline_run_dir() is None:
            self._show_warning("Run baseline training first, or choose an existing baseline run for self-transfer learning.")
            return False
        if self.transfer_base_model_source_combo_box.currentIndex() == 1 and self.transfer_compatibility_status_badge.text() != "Compatible":
            self._show_warning("The selected existing baseline run is not compatible with the current dataset/cache.")
            return False
        return True

    def _preferred_output_path(self) -> str | None:
        if self.last_workflow_summary:
            if self.last_workflow_summary.get("transfer"):
                return self.last_workflow_summary["transfer"]["run_dir"]
            if self.last_workflow_summary.get("baseline"):
                return self.last_workflow_summary["baseline"]["run_dir"]
        return self.model_output_folder_path_edit.text().strip() or None

    def _current_baseline_run_dir(self) -> str | None:
        if self.last_baseline_summary and self.last_baseline_summary.get("run_dir"):
            return str(self.last_baseline_summary["run_dir"])
        if self.last_workflow_summary and self.last_workflow_summary.get("baseline"):
            return str(self.last_workflow_summary["baseline"].get("run_dir", "")) or None
        return None

    def _default_cache_path(self, run_name: str | None = None) -> str:
        active_run_name = (run_name or self.run_name_edit.text().strip() or "mlp_run").strip() or "mlp_run"
        return str((self._output_root_path() / "cache" / f"{active_run_name}.npz").resolve())

    def _legacy_default_cache_path(self, run_name: str) -> str:
        return str((Path("artifacts/cache") / f"{run_name}.npz").resolve())

    def _infer_cache_path_manually_selected(self, cache_path: str, *, run_name: str) -> bool:
        normalized_path = cache_path.strip()
        if not normalized_path:
            return False
        auto_managed_paths = {self._default_cache_path(run_name)}
        if current_runtime_paths().mode == "source":
            auto_managed_paths.update(
                {
                    str(Path("artifacts/cache/gui_session_cache.npz").resolve()),
                    self._legacy_default_cache_path(run_name),
                }
            )
        return str(Path(normalized_path).resolve()) not in auto_managed_paths

    def _set_cache_path_value(self, cache_path: str, *, manually_selected: bool) -> None:
        self._setting_cache_path = True
        try:
            self.cache_path_edit.setText(cache_path)
        finally:
            self._setting_cache_path = False
        self._cache_path_manually_selected = bool(cache_path.strip()) and manually_selected

    def _sync_auto_cache_path(self) -> None:
        if self._cache_path_manually_selected:
            return
        self._set_cache_path_value(self._default_cache_path(), manually_selected=False)

    def _ensure_cache_path(self) -> str:
        if self._cache_path_manually_selected:
            cache_path = self.cache_path_edit.text().strip()
            if cache_path:
                return cache_path
        cache_path = self._default_cache_path()
        if self.cache_path_edit.text().strip() != cache_path:
            self._set_cache_path_value(cache_path, manually_selected=False)
        return cache_path

    def _require_data_paths(self) -> dict[str, str] | None:
        dataset_folder = self.dataset_folder_edit.text().strip()
        input_feature_path = self.input_feature_path_edit.text().strip()
        ground_truth_data_dir = self.ground_truth_data_folder_path_edit.text().strip()
        model_output_dir = self.model_output_folder_path_edit.text().strip()

        # Need at least one data source: the new single-folder field OR the legacy fields.
        if not dataset_folder and not input_feature_path and not ground_truth_data_dir:
            self._show_warning("Select a dataset folder first.")
            return None
        if not model_output_dir:
            self._show_warning("Select the output folder first.")
            return None
        if not self.run_name_edit.text().strip():
            self._show_warning("Provide a run name before starting.")
            return None
        Path(model_output_dir).mkdir(parents=True, exist_ok=True)
        return {
            "dataset_root": dataset_folder,
            "input_feature_path": input_feature_path,
            "ground_truth_data_dir": ground_truth_data_dir,
            "cache_path": self._ensure_cache_path(),
        }

    def _autofill_run_name_and_cache(self) -> None:
        input_feature_path = self.input_feature_path_edit.text().strip()
        if input_feature_path and not self.dataset_folder_edit.text().strip():
            self.run_name_edit.setText(default_run_name(input_feature_path))
        self._sync_auto_cache_path()
        self._update_topbar_run_name()

    def _autofill_run_name_from_dataset_folder(self) -> None:
        """Auto-fill the run name and reset scan status when the dataset folder changes."""
        dataset_dir = self.dataset_folder_edit.text().strip()
        if dataset_dir:
            folder_name = Path(dataset_dir).name or Path(dataset_dir).stem
            self.run_name_edit.setText(folder_name.replace(" ", "_").lower())
        # Clear legacy fields when a dataset folder is selected to avoid confusion.
        if dataset_dir:
            self.input_feature_path_edit.clear()
            self.ground_truth_data_folder_path_edit.clear()
        if dataset_dir:
            self.dataset_schema_status_badge.set_status("Not Scanned")
        self._sync_auto_cache_path()
        self._update_topbar_run_name()

    def _autofill_run_name_from_gt_dir(self) -> None:
        # When the input-feature path is empty (cadence_csv), derive the run
        # name from the ground-truth folder instead.
        if not self.input_feature_path_edit.text().strip() and not self.dataset_folder_edit.text().strip():
            gt_dir = self.ground_truth_data_folder_path_edit.text().strip()
            if gt_dir:
                folder_name = Path(gt_dir).name or Path(gt_dir).stem
                self.run_name_edit.setText(folder_name.replace(" ", "_").lower())
        self._sync_auto_cache_path()
        self._update_topbar_run_name()

    def _on_run_name_changed(self, _text: str) -> None:
        self._sync_auto_cache_path()
        self._update_topbar_run_name()

    def _on_output_folder_changed(self, _text: str) -> None:
        self._sync_auto_cache_path()
        self._update_topbar_run_name()

    def _output_root_path(self) -> Path:
        output_dir = self.model_output_folder_path_edit.text().strip()
        return Path(output_dir) if output_dir else self._default_output_dir()

    def _default_output_dir(self) -> Path:
        return current_runtime_paths().default_output_dir

    def _default_dialog_root(self) -> Path:
        output_dir = self.model_output_folder_path_edit.text().strip()
        return Path(output_dir) if output_dir else self._default_output_dir()

    def _on_cache_path_edited(self, text: str) -> None:
        if self._setting_cache_path:
            return
        self._cache_path_manually_selected = bool(text.strip())

    def _normalize_progress_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        nested = payload.get("data")
        if not isinstance(nested, dict) or not nested:
            return payload
        normalized = dict(nested)
        normalized.update(payload)
        normalized.pop("data", None)
        return normalized

    def _apply_baseline_summary(self, summary: dict[str, Any]) -> None:
        history = summary.get("history", [])
        if history:
            self._baseline_epochs = [int(entry["epoch"]) for entry in history if "epoch" in entry]
            self._baseline_train_losses = [float(entry["train_loss"]) for entry in history if "train_loss" in entry]
            self._baseline_val_losses = [float(entry["val_loss"]) for entry in history if "val_loss" in entry]
            self.baseline_train_curve.setData(self._baseline_epochs, self._baseline_train_losses)
            self.baseline_val_curve.setData(self._baseline_epochs, self._baseline_val_losses)
        channel_labels = summary.get("channel_mae_with_units")
        if channel_labels:
            self.metric_cards["average_mae"].set_value(" | ".join(channel_labels))
        else:
            average_mae = summary.get("average_evaluation_mae", summary.get("average_test_mae"))
            if average_mae is not None:
                self.metric_cards["average_mae"].set_value(f"{float(average_mae):.6f}")
        best_val_loss = summary.get("best_val_loss")
        if best_val_loss is not None:
            self.metric_cards["best_metric"].set_value(f"Best val {float(best_val_loss):.6f}")
        runtime_seconds = summary.get("runtime_seconds")
        if runtime_seconds is not None:
            self.metric_cards["elapsed"].set_value(self._format_seconds(runtime_seconds))

    def _populate_test_sample_plots(self, summary: dict[str, Any]) -> None:
        """Display prediction-vs-truth curves for test samples after training."""
        # Clear previous plots.
        for widget in self._test_sample_plots:
            widget.setParent(None)
            widget.deleteLater()
        self._test_sample_plots.clear()
        self.test_samples_placeholder.setVisible(False)

        test_data = summary.get("test_sample_data")
        if not test_data:
            self.test_samples_placeholder.setText("No test sample data available.")
            self.test_samples_placeholder.setVisible(True)
            return

        freq_ghz = test_data["freq_ghz"]
        channel_names = test_data["channel_names"]
        samples = test_data["samples"]  # list of {"index", "pred", "true"}

        insert_pos = 0  # insert before the stretch at the end

        for sample in samples:
            sample_idx = sample["index"]
            pred = sample["pred"]   # list of lists: [channel][freq]
            true = sample["true"]

            label = QLabel(f"Test Sample #{sample_idx}")
            label.setStyleSheet("font-weight: bold; font-size: 11pt; margin-top: 8px;")
            self.test_samples_layout.insertWidget(insert_pos, label)
            self._test_sample_plots.append(label)
            insert_pos += 1

            # Create a grid of plots, one per channel.
            num_channels = len(pred)
            ncols = min(4, num_channels)
            nrows = (num_channels + ncols - 1) // ncols
            plot_height = 250
            grid_widget = QWidget()
            grid_widget.setMinimumHeight(plot_height * nrows + 20)
            grid_layout = QGridLayout(grid_widget)
            grid_layout.setContentsMargins(0, 0, 0, 0)
            grid_layout.setSpacing(8)

            for ch_idx in range(num_channels):
                row, col = divmod(ch_idx, ncols)
                pw = pg.PlotWidget()
                pw.setMinimumHeight(plot_height)
                ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx}"
                sweep = getattr(self, "_sweep_label", "Frequency (GHz)")
                configure_plot_widget(pw, title=ch_name, x_label=sweep, y_label="Value")

                true_pen = pg.mkPen(color="#2196F3", width=2)
                pred_pen = pg.mkPen(color="#FF5722", width=2, style=Qt.PenStyle.DashLine)
                pw.plot(freq_ghz, true[ch_idx], pen=true_pen, name="True")
                pw.plot(freq_ghz, pred[ch_idx], pen=pred_pen, name="Pred")

                pw.addLegend()

                grid_layout.addWidget(pw, row, col)
                self._test_sample_plots.append(pw)

            self.test_samples_layout.insertWidget(insert_pos, grid_widget)
            self._test_sample_plots.append(grid_widget)
            insert_pos += 1

    def _update_topbar_run_name(self) -> None:
        run_name = self.run_name_edit.text().strip() or "Unconfigured run"
        output_dir = self.model_output_folder_path_edit.text().strip() or "no output folder selected"
        self.topbar_run_name.setText(f"{run_name} | {output_dir}")

    def _set_metric_defaults(self) -> None:
        for key, value in {
            "current_phase": "Idle",
            "current_progress": "-",
            "secondary_progress": "-",
            "best_metric": "-",
            "train_loss": "-",
            "validation_loss": "-",
            "average_mae": "-",
            "elapsed": "-",
            "eta": "-",
        }.items():
            self.metric_cards[key].set_value(value)

    def _transfer_band_context(self, payload: dict[str, Any]) -> str:
        return (
            f"T={int(payload.get('transfer_iteration', 0))} | "
            f"{str(payload.get('direction', 'transfer')).title()} band "
            f"{int(payload.get('band_index', 0)) + 1}/{int(payload.get('num_bands', 0))}"
        )

    def _format_progress_line(self, payload: dict[str, Any]) -> str:
        event = payload.get("event", "")
        message = str(payload.get("message", "")).strip()
        phase = str(payload.get("phase", "")).title() or "Task"
        if payload.get("phase") == "scan":
            completed_samples = payload.get("completed_samples")
            total_samples = payload.get("total_samples")
            if event == "cache_progress" and completed_samples is not None and total_samples is not None:
                return f"[Scan] Loaded ground-truth sample {completed_samples}/{total_samples}"
            return f"[Scan] {message}" if message else ""
        if event == "epoch_end" and payload.get("phase") == "baseline" and payload.get("trial_index") is None:
            epoch = payload.get("epoch")
            total_epochs = payload.get("total_epochs")
            train_loss = payload.get("train_loss")
            val_loss = payload.get("val_loss")
            best_val = payload.get("best_val_loss")
            if None not in (epoch, total_epochs, train_loss, val_loss, best_val):
                return f"[{phase}] Epoch {epoch}/{total_epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_val={best_val:.6f}"
            return f"[{phase}] {message or 'Epoch completed.'}"
        if event == "band_epoch_end":
            epoch = payload.get("epoch")
            total_epochs = payload.get("total_epochs")
            train_loss = payload.get("train_loss")
            if None not in (epoch, total_epochs, train_loss):
                return f"[Transfer] {self._transfer_band_context(payload)} epoch {epoch}/{total_epochs} train_loss={train_loss:.6f}"
            return f"[Transfer] {message or 'Band epoch completed.'}"
        if event == "trial_completed":
            if None not in (
                payload.get("trial_index"),
                payload.get("trial_count"),
                payload.get("trial_label"),
                payload.get("average_val_mae"),
                payload.get("runtime_seconds"),
            ):
                return f"[Search] Trial {payload['trial_index']}/{payload['trial_count']} ({payload['trial_label']}) avg_val_mae={payload['average_val_mae']:.6f} runtime={self._format_seconds(payload['runtime_seconds'])}"
            return f"[Search] {message or 'Trial completed.'}"
        if event == "iteration_completed":
            if payload.get("transfer_iteration") is not None and payload.get("average_mae") is not None:
                return f"[Transfer] Iteration {payload['transfer_iteration']} completed with average_mae={payload['average_mae']:.6f}"
            return f"[Transfer] {message or 'Iteration completed.'}"
        return f"[{phase}] {message}" if message else ""

    def append_log(self, text: str) -> None:
        self.run_log_text_edit.appendPlainText(text)
        self.run_log_text_edit.verticalScrollBar().setValue(self.run_log_text_edit.verticalScrollBar().maximum())

    def _show_warning(self, text: str) -> None:
        QMessageBox.warning(self, "Surrogate Model Traning Suite", text)

    def _format_seconds(self, value: Any) -> str:
        if value in (None, "", "-"):
            return "-"
        total_seconds = int(float(value))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {seconds:02d}s"
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        try:
            save_last_session(self.collect_config_payload())
        except Exception:  # pragma: no cover - best effort on close
            pass
        if self.current_task is not None and self.current_task.thread is not None and self.current_task.thread.isRunning():
            self.current_task.stop()
            self.current_task.thread.quit()
            self.current_task.thread.wait(3000)
        self.license_controller.shutdown()
        super().closeEvent(event)


def create_application() -> tuple[QApplication, MlpTrainingStudio]:
    app = QApplication.instance() or QApplication([])
    apply_application_theme(app)
    window = MlpTrainingStudio()
    return app, window
