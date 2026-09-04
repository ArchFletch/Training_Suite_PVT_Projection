"""PySide6 desktop GUI for the MLP training workflow."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import QObject, QThread, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
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
    QListWidget,
    QListWidgetItem,
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
    QVBoxLayout,
    QWidget,
)

from .app_paths import current_runtime_paths
from .gui_backend import (
    build_and_scan_dataset,
    build_suggest_result,
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
    LICENSE_REQUIRED_MESSAGE,
    LicenseClientError,
    LicenseLeaseController,
    LicenseLeaseState,
    LicenseStatus,
    format_utc_timestamp,
    normalize_server_url,
    test_license_connection,
)
from .runner import (
    LOSS_FUNCTIONS,
    MODEL_TYPES,
    PROJECTION_MODEL_TYPES,
    SCHEDULER_TYPES,
    TrainConfig,
    canonical_model_type,
    uses_corner_projection,
)
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


class MlpTrainingStudio(QMainWindow):
    """Main application window."""

    def __init__(self, *, executor: QtTaskExecutor | None = None) -> None:
        super().__init__()
        self.executor = executor or QtTaskExecutor()
        self.current_task = None
        self.license_connection_message = LICENSE_REQUIRED_MESSAGE
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
        self.last_onnx_export_path: str | None = None
        self.selected_search_full_config: dict[str, Any] | None = None
        self.search_row_configs: list[dict[str, Any]] = []
        self.current_task_name = "idle"
        # Display-only: the cache path the engine last used. Always derived from
        # the output folder and run name, never chosen by the user.
        self._cache_path_value = ""
        self._search_max_samples_autofill_value: int | None = None
        self._setting_search_max_samples = False
        self._controls_locked = False
        self._search_text_items: list[pg.TextItem] = []

        self._baseline_epochs: list[float] = []
        self._baseline_train_losses: list[float] = []
        self._baseline_val_losses: list[float] = []
        self.setWindowTitle("Surrogate Model Training Suite")
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
        title_label = QLabel("Surrogate Model Training Suite")
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
        layout.addWidget(self._section_header("Data Sources", "Select your dataset file and output location"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.dataset_file_edit = QLineEdit()
        self.dataset_file_edit.setPlaceholderText("Select the .npz dataset file")
        self.dataset_file_edit.setToolTip(
            "The dataset is one .npz file holding features, targets and frequency_hz.\n"
            "Build one from raw simulation output with the dataset preparation tools, "
            "or point at a bundle.npz produced by the PVT data-generation flow.\n\n"
            "Raw folder layouts (log.txt + .sNp, Cadence CSV) are still supported from "
            "the command line via train_baseline.py --data-root."
        )
        self.dataset_file_edit.textChanged.connect(self._autofill_run_name_from_dataset_file)
        self.browse_dataset_file_button = self._make_button("Browse...", secondary=True)
        self.browse_dataset_file_button.clicked.connect(self.browse_dataset_file)
        self._add_path_row(grid, 0, "Dataset File", self.dataset_file_edit, self.browse_dataset_file_button)

        self.model_output_folder_path_edit = QLineEdit()
        self.model_output_folder_path_edit.setPlaceholderText("Select the output folder for runs, artifacts, and cache")
        self.model_output_folder_path_edit.textChanged.connect(self._on_output_folder_changed)
        self.browse_model_output_folder_button = self._make_button("Browse...", secondary=True)
        self.browse_model_output_folder_button.clicked.connect(self.browse_model_output_dir)
        self._add_path_row(grid, 1, "Output Folder", self.model_output_folder_path_edit, self.browse_model_output_folder_button)

        self.run_name_edit = QLineEdit()
        self.run_name_edit.textChanged.connect(self._on_run_name_changed)
        self._add_form_row(grid, 2, "Run Name", self.run_name_edit)

        # The explicit input-feature / ground-truth / cache overrides that used to
        # live behind an "Advanced" disclosure are gone. One .npz names the whole
        # dataset, so the two path overrides had nothing left to override, and the
        # cache is derived from the output folder and run name. All three remain
        # available from the command line (train_baseline.py --input-feature-path /
        # --ground-truth-data-dir / --cache-path).
        layout.addLayout(grid)

        button_row = QHBoxLayout()
        self.dataset_schema_status_badge = StatusBadge("Not Scanned")
        button_row.addWidget(self.dataset_schema_status_badge)
        button_row.addStretch()
        self.scan_dataset_button = self._make_button("Scan Dataset")
        self.scan_dataset_button.clicked.connect(self.scan_dataset)
        button_row.addWidget(self.scan_dataset_button)
        layout.addLayout(button_row)
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
        layout.addWidget(self._section_header("Suggest Initial Settings", "Scan-only starter values for baseline training"))

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
        layout.addWidget(self._section_header("Training Settings", "Editable baseline hyperparameters"))

        device_grid = QGridLayout()
        device_grid.setHorizontalSpacing(10)
        device_grid.setVerticalSpacing(8)
        self.training_device_combo_box = _NoScrollComboBox()
        self.training_device_combo_box.setToolTip(
            "Compute unit used for training. Detected automatically when the app starts."
        )
        self._populate_device_choices()
        self._add_form_row(device_grid, 0, "Training Device", self.training_device_combo_box)
        layout.addLayout(device_grid)

        self.training_tabs = QTabWidget()
        layout.addWidget(self.training_tabs)

        self.training_tabs.addTab(self._build_baseline_tab(), "Baseline Training")
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
        self.baseline_model_type_combo_box.setCurrentText("SpectraNet")
        self.baseline_model_type_combo_box.setToolTip(
            "SpectraNet: one dense network from inputs to the whole spectrum.\n"
            "SpectraHydra: shared encoder with one output head per channel.\n"
            "SpectraHydraProj: SpectraHydra plus a learned projection of the "
            "checked PVT Corner Columns.\n"
            "SpectraTrunk: one weight-shared residual trunk evaluated per "
            "frequency point (context MLP + Fourier frequency embedding), so "
            "outputs are forced to be smooth functions of frequency instead of "
            "independent output slots. Width is the trunk width and Depth the "
            "number of residual blocks; its reference recipe used width 512, "
            "depth 4, batch 32, AdamW around 1e-3 with the cosine scheduler."
        )
        self.baseline_model_type_combo_box.currentTextChanged.connect(
            self._refresh_model_settings_visibility
        )
        # SpectraHydraProj settings: which input-feature columns are PVT corner
        # conditions (fed to the learned projection) and the embedding width.
        # The list is populated with the dataset's active feature names after a
        # scan; until then it shows whatever a restored session selected.
        self._projection_column_selection: list[str] = []
        self.baseline_projection_columns_list = QListWidget()
        self.baseline_projection_columns_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.baseline_projection_columns_list.setMaximumHeight(96)
        self.baseline_projection_columns_list.setToolTip(
            "Check the PVT corner/condition columns (e.g. temperature, supply, process "
            "one-hots). They feed SpectraHydraProj's learned corner projection. Scan "
            "the dataset to list its input-feature columns."
        )
        self.baseline_projection_columns_list.itemChanged.connect(self._on_projection_column_toggled)
        self.baseline_projection_dim_spin_box = self._make_int_spin(1, 256, 16)
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
        self.baseline_train_fraction_spin_box.setToolTip(
            "Fraction of the dataset used for training. Validation takes its own "
            "fraction below and whatever is left over becomes the internal test "
            "fold.\n\n"
            "Setting Train + Validation to exactly 1.0 (e.g. 0.900 / 0.100) keeps no "
            "test fold at all: those rows go into training instead, and the internal "
            "MAE cards stay blank because there is nothing left to measure. Worth it "
            "when you judge the model by other means — on one measured dataset it was "
            "about 19% better."
        )
        self.baseline_validation_fraction_spin_box = self._make_float_spin(0.0, 0.90, 0.10, decimals=3, step=0.01)
        self.baseline_seed_spin_box = self._make_int_spin(0, 1000000, 42)
        self.restore_recommended_baseline_button = self._make_button("Restore Recommended", secondary=True)
        self.restore_recommended_baseline_button.clicked.connect(self._restore_recommended_baseline)

        fields = [
            ("Model Type", self.baseline_model_type_combo_box),
            ("PVT Corner Columns", self.baseline_projection_columns_list),
            ("Corner Projection Width", self.baseline_projection_dim_spin_box),
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
        # Keep every row addressable by its caption so model-specific settings
        # can be shown and hidden as label+widget pairs.
        self._baseline_form_rows: dict[str, tuple[QLabel, QWidget]] = {}
        for row, (label, widget) in enumerate(fields):
            self._baseline_form_rows[label] = (self._add_form_row(grid, row, label, widget), widget)
        grid.addWidget(self.restore_recommended_baseline_button, len(fields), 1)
        self._refresh_model_settings_visibility()
        return tab

    def _build_run_controls_card(self) -> QWidget:
        card = CardFrame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(self._section_header("Run Controls", "Launch a baseline run, stop gracefully, or export summaries"))

        buttons = QHBoxLayout()
        self.start_baseline_button = self._make_button("Start Baseline Training")
        self.stop_training_button = self._make_button("Stop", secondary=True)
        self.stop_training_button.setEnabled(False)
        self.open_output_folder_button = self._make_button("Open Output Folder", secondary=True)
        self.export_run_summary_button = self._make_button("Export Summary", secondary=True)
        self.export_onnx_button = self._make_button("Export to ONNX", secondary=True)
        self.start_baseline_button.clicked.connect(self.start_baseline_training)
        self.stop_training_button.clicked.connect(self.stop_current_task)
        self.open_output_folder_button.clicked.connect(self.open_output_folder)
        self.export_run_summary_button.clicked.connect(self.export_run_summary)
        self.export_onnx_button.clicked.connect(self.export_baseline_to_onnx)
        buttons.addWidget(self.start_baseline_button)
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

    def _update_channel_metric_cards(self, labels: list[str]) -> None:
        """Show one MAE card per ground-truth channel, in the run's channel order.

        ``labels`` are the runner's ready-formatted "gain: 0.2671 dB" strings; the
        runner emits them only when the dataset declares channel units, so datasets
        without units keep the single averaged card and nothing is created here.
        """
        for label in labels:
            name, _, value = str(label).partition(":")
            name, value = name.strip(), value.strip()
            if not name or not value:
                continue
            card = self._channel_metric_cards.get(name)
            if card is None:
                card = MetricCard(f"MAE {name}")
                self._channel_metric_cards[name] = card
                position = len(self._channel_metric_cards) - 1
                self._channel_metric_grid.addWidget(
                    card, self._channel_metric_row + position // 3, position % 3
                )
            card.set_value(value)

    def _reset_channel_metric_cards(self) -> None:
        """Clear stale per-channel values when a new run starts.

        The cards themselves are kept so the layout does not jump between runs on
        the same dataset; only a dataset with different channels replaces them.
        """
        for card in self._channel_metric_cards.values():
            card.set_value("-")

    def _build_metrics_card(self) -> QWidget:
        card = CardFrame()
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.addWidget(self._section_header("Current Metrics", "Phase-aware live progress across scan, suggestion, and baseline training"))

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
            ("average_mae", "Average MAE (mixed units)"),
            ("elapsed", "Elapsed"),
            ("eta", "ETA"),
        ]
        self.metric_cards: dict[str, MetricCard] = {}
        for index, (key, title) in enumerate(metric_titles):
            card_widget = MetricCard(title)
            self.metric_cards[key] = card_widget
            grid.addWidget(card_widget, index // 3, index % 3)
        # Per-channel MAE is the number that carries physical units, so it is shown
        # outright rather than hidden behind a toggle. The cards are created from the
        # run's own channel list because channel count and names vary by dataset.
        self._channel_metric_row = (len(metric_titles) + 2) // 3
        self._channel_metric_cards: dict[str, MetricCard] = {}
        self._channel_metric_grid = grid
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
        return tabs

    # ------------------------------------------------------------------
    # Defaults and persistence
    # ------------------------------------------------------------------
    def _apply_default_values(self) -> None:
        self.model_output_folder_path_edit.setText(str(self._default_output_dir().resolve()))
        self._set_cache_path_value("")
        self.run_name_edit.setText("mlp_run")
        self.dataset_schema_status_badge.set_status("Not Scanned")
        self.initial_suggestion_confidence_badge.set_status("Not Scanned")
        self.license_server_status_badge.set_status("Unconfigured")
        self.license_seat_state_badge.set_status(self.license_lease_state.badge_text)
        self._set_metric_defaults()
        self._refresh_license_display()
        self._update_topbar_run_name()
        self.append_log("Ready. Select the .npz dataset file and scan the data to begin.")

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
                # The .npz dataset file is the only data source; without it a
                # restored session silently forgets which dataset was loaded. The
                # cache path is not saved: it is derived from the output folder and
                # run name, both of which are.
                "dataset_file": self.dataset_file_edit.text().strip(),
                "output_dir": self.model_output_folder_path_edit.text().strip(),
                "run_name": self.run_name_edit.text().strip(),
            },
            "baseline": self._collect_baseline_form(),
            "licensing": {
                "server_url": self.license_server_url_edit.text().strip(),
                "last_status": self._current_license_status_summary(),
            },
        }

    def apply_config_payload(self, payload: dict[str, Any]) -> None:
        self._set_device_selection(payload.get("device"))
        data_sources = payload.get("data_sources", {})
        self.dataset_file_edit.setText(str(data_sources.get("dataset_file", "")))
        output_dir = data_sources.get("output_dir", data_sources.get("model_output_dir", self.model_output_folder_path_edit.text()))
        self.model_output_folder_path_edit.setText(str(output_dir))
        run_name = str(data_sources.get("run_name", self.run_name_edit.text()))
        self.run_name_edit.setText(run_name)
        self._note_removed_data_source_fields(data_sources)
        self._sync_auto_cache_path()

        baseline = payload.get("baseline", {})
        self._apply_baseline_form(baseline)

        self._note_removed_transfer_settings(payload.get("transfer"))
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
            self.license_connection_message = LICENSE_REQUIRED_MESSAGE
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
            self.license_connection_message = LICENSE_REQUIRED_MESSAGE
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
        # Fail closed. There is no "no server configured, so anything goes" branch:
        # that made an untouched License Server field equivalent to an unlimited
        # licence. A run requires a seat checked out from the server currently
        # named in the field.
        configured_url = normalize_server_url(self.license_server_url_edit.text())
        active_url = normalize_server_url(self.license_lease_state.server_url)
        if not configured_url or not active_url:
            return False
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
                # The cache is always auto-managed now, so a scan always rebuilds
                # it from the selected dataset file rather than trusting a stale one.
                "overwrite": True,
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
        if not self._validate_projection_settings():
            return

        self._reset_baseline_plots()
        self.last_workflow_summary = None

        self._start_task(
            run_training_workflow,
            kwargs={
                "baseline_config": self._build_baseline_train_config(),
                # The engine still runs baseline-only, transfer-only, or both.
                # The GUI only ever asks for baseline; self-transfer moved to
                # run_self_transfer.py.
                "transfer_config": None,
            },
            task_name="training",
            busy_state="Training",
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

    def browse_dataset_file(self) -> None:
        start = self.dataset_file_edit.text().strip() or str(self._default_dialog_root())
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset File", start, "Dataset Arrays (*.npz);;All Files (*)"
        )
        if path:
            self.dataset_file_edit.setText(path)

    def browse_model_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.model_output_folder_path_edit.text())
        if path:
            self.model_output_folder_path_edit.setText(path)

    def open_output_folder(self) -> None:
        candidate = self._preferred_output_path()
        if candidate is None:
            self._show_warning("No output folder is available yet.")
            return
        path = Path(candidate)
        if not path.exists():
            self._show_warning("The selected output path does not exist yet.")
            return
        # QDesktopServices picks the right opener per platform (xdg-open on Linux,
        # open on macOS, ShellExecute on Windows) instead of the Windows-only
        # os.startfile.
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            self._show_warning(f"Could not open the output folder: {path}")

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
            self.license_connection_message = LICENSE_REQUIRED_MESSAGE
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

    def apply_suggested_settings(self) -> None:
        if self.last_suggest_result is None:
            self._show_warning("No suggestion is available yet.")
            return
        self._apply_baseline_form(self.last_suggest_result["suggested_baseline_config"])
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
        elif self.current_task_name == "suggest":
            # Don't leave the confidence badge frozen at "Checking".
            self.initial_suggestion_confidence_badge.set_status("Error")
            self.initial_suggestion_status_text.setText("Suggestion failed; see the log for details.")
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
        if self.run_state_badge.text() in {"Training", "Searching", "Scanning", "Suggesting", "Checking", "Exporting"}:
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

    def _on_scan_completed(self, result: dict[str, Any]) -> None:
        if result.get("status") == "stopped":
            self.dataset_schema_status_badge.set_status("Not Scanned")
            self.run_state_badge.set_status("Stopped")
            self.append_log("Dataset scan stopped before completion.")
            self._refresh_run_button_availability()
            return
        self.last_scan_result = result
        self._sweep_label = result.get("sweep_label", "Frequency (GHz)")
        self.dataset_schema_status_badge.set_status(result.get("schema_status", "Valid"))
        for message in result.get("split_warnings", []):
            self.append_log(f"[Split] {message}")
        scanned_columns = list(result.get("active_input_feature_names", []))
        self._populate_projection_columns(scanned_columns)
        self._fill_table(self.data_preview_table, result["preview_rows"])
        self._set_cache_path_value(result["cache_path"])
        self.append_log(f"Dataset scan completed for {result['dataset_name']}.")
        self._reset_baseline_plots()
        self.run_progress_bar.setValue(100)
        self.run_state_badge.set_status("Completed")
        self._refresh_run_button_availability()

    def _on_suggest_completed(self, result: dict[str, Any]) -> None:
        if result.get("status") == "stopped":
            self.initial_suggestion_confidence_badge.set_status("Stopped")
            self.initial_suggestion_status_text.setText("Suggestion stopped before completion.")
            self.run_state_badge.set_status("Stopped")
            self.append_log("Initial settings suggestion stopped before completion.")
            return
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
        if result.get("status") == "stopped":
            self.run_state_badge.set_status("Stopped")
            self.append_log("Quick search stopped before completion.")
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
        if baseline:
            self.last_baseline_summary = baseline
            self._apply_baseline_summary(baseline)
            self.append_log(f"Baseline run saved to {baseline['run_dir']}")
            if baseline.get("test_sample_data"):
                self._populate_test_sample_plots(baseline)
            self.append_log("Baseline training completed.")
        # Show the test samples when the run produced any, else the training monitor.
        self.monitor_tabs.setCurrentIndex(1 if baseline and baseline.get("test_sample_data") else 0)

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
            # The averaged card mixes channel units (dB / deg / decades); the
            # per-channel cards beside it carry the numbers that mean something.
            average_mae = payload.get("average_evaluation_mae", payload.get("average_test_mae"))
            if average_mae is not None:
                self.metric_cards["average_mae"].set_value(f"{average_mae:.6f}")
            self._update_channel_metric_cards(list(payload.get("channel_mae_with_units") or []))
        elif event == "completed":
            self.run_progress_bar.setValue(100)
            self.run_state_badge.set_status("Completed")
        elif event == "stopped":
            self.run_state_badge.set_status("Stopped")

    # ------------------------------------------------------------------
    # Form helpers
    # ------------------------------------------------------------------
    def _collect_baseline_form(self) -> dict[str, Any]:
        return {
            "model_type": self.baseline_model_type_combo_box.currentText(),
            "projection_columns": self._selected_projection_columns(),
            "projection_dim": self.baseline_projection_dim_spin_box.value(),
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
        # Normalize legacy names so forms saved before the model rename still
        # select the right entry in the combo box.
        model_type = canonical_model_type(payload.get("model_type", "SpectraNet"))
        if model_type in MODEL_TYPES:
            self.baseline_model_type_combo_box.setCurrentText(model_type)
        # Projection fields are only applied from a payload that actually selects
        # the projection model. Suggest/search configs are built via
        # asdict(TrainConfig(...)) and therefore always carry the dataclass
        # defaults (projection_columns=None, projection_dim=16) for the other
        # model types — applying those would silently clear or reset a user's
        # corner setup.
        if uses_corner_projection(model_type):
            projection_columns = payload.get("projection_columns")
            if projection_columns is not None:
                self._set_projection_columns_selection([str(name) for name in projection_columns])
            projection_dim = payload.get("projection_dim")
            if projection_dim is not None:
                self.baseline_projection_dim_spin_box.setValue(int(projection_dim))
        # The design-level split was removed from the GUI (the datasets this GUI
        # targets put each design at one corner, where the row-level split is
        # already design-disjoint). Configs written before that — or by the CLI,
        # which still supports it — may carry the keys; ignore them rather than
        # fail the restore, but say something when one is non-empty, because the
        # split it asked for will not happen and silence would look like it did.
        split_columns = [
            str(name)
            for key in ("split_design_columns", "split_corner_columns")
            for name in (payload.get(key) or [])
        ]
        if split_columns:
            self.append_log(
                "This configuration set a design-level split "
                f"(columns: {', '.join(split_columns)}). That option has been "
                "removed from the GUI, so this run will use the row-level split. "
                "It is still available from the command line via train_baseline.py "
                "--split-design-columns / --split-corner-columns."
            )
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
        # The External Eval Set field was removed from the GUI. Configs written
        # before that still carry the key; ignore it rather than fail the whole
        # restore, but say something when it held a real path, because it used to
        # change the split (it merged the test fold into training) and silently
        # dropping it would quietly retrain on less data.
        eval_dataset_path = payload.get("eval_dataset_path")
        if eval_dataset_path:
            self.append_log(
                "This configuration set an External Eval Set "
                f"({eval_dataset_path}). That field has been removed from the GUI; "
                "the option is still available from the command line via "
                "train_baseline.py --eval-dataset-path. To reproduce the split it "
                "used, set Train Fraction 0.900 and Validation Fraction 0.100."
            )

    # ------------------------------------------------------------------
    # PVT corner-column picker (SpectraHydraProj)
    # ------------------------------------------------------------------
    def _selected_projection_columns(self) -> list[str]:
        return list(self._projection_column_selection)

    def _on_projection_column_toggled(self, _item: QListWidgetItem) -> None:
        # Recompute from widget state so the selection stays in dataset column
        # order regardless of the order boxes were clicked in.
        widget = self.baseline_projection_columns_list
        self._projection_column_selection = [
            widget.item(index).text()
            for index in range(widget.count())
            if widget.item(index).checkState() == Qt.CheckState.Checked
        ]

    def _rebuild_projection_column_items(self, names: list[str]) -> None:
        widget = self.baseline_projection_columns_list
        widget.blockSignals(True)
        widget.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if name in self._projection_column_selection
                else Qt.CheckState.Unchecked
            )
            widget.addItem(item)
        widget.blockSignals(False)

    def _populate_projection_columns(self, names: list[str]) -> None:
        """Fill the corner-column picker with the scanned dataset's feature names.

        The checked selection survives repopulation by name. Selected names that
        do not exist in the newly scanned dataset are unselected with a log line
        instead of failing later at training time.
        """
        missing = [name for name in self._projection_column_selection if name not in names]
        if missing:
            self.append_log(
                "PVT corner column(s) not present in the scanned dataset were unselected: "
                + ", ".join(missing)
            )
        self._projection_column_selection = [
            name for name in self._projection_column_selection if name in names
        ]
        self._rebuild_projection_column_items(names)

    def _set_projection_columns_selection(self, names: list[str]) -> None:
        """Programmatically select corner columns (session restore / config load).

        Before a dataset scan the picker has no items, so restored names are
        appended as items to keep the pending selection visible.
        """
        self._projection_column_selection = [str(name) for name in names]
        widget = self.baseline_projection_columns_list
        existing = [widget.item(index).text() for index in range(widget.count())]
        items = existing + [name for name in self._projection_column_selection if name not in existing]
        self._rebuild_projection_column_items(items)

    # Baseline settings that only some model types use, by row caption. A row
    # absent from this map belongs to every model and is always shown. The model
    # type sets come from the engine (runner.PROJECTION_MODEL_TYPES), so adding a
    # model there cannot leave the form showing the wrong controls.
    @property
    def _model_specific_baseline_rows(self) -> dict[str, tuple[str, ...]]:
        return {
            "PVT Corner Columns": PROJECTION_MODEL_TYPES,
            "Corner Projection Width": PROJECTION_MODEL_TYPES,
        }

    def _refresh_model_settings_visibility(self) -> None:
        """Show only the settings the selected model actually reads.

        The corner-projection rows used to be merely disabled, so every model
        displayed two controls that did nothing for it. Rows are hidden as
        label+widget pairs, and stay disabled as well as hidden so a hidden
        control cannot be reached by keyboard focus or a stale programmatic
        enable.
        """
        model_type = canonical_model_type(self.baseline_model_type_combo_box.currentText())
        for caption, (label, widget) in self._baseline_form_rows.items():
            applies_to = self._model_specific_baseline_rows.get(caption)
            visible = applies_to is None or model_type in applies_to
            label.setVisible(visible)
            widget.setVisible(visible)
            if applies_to is not None:
                widget.setEnabled(visible)

    def _validate_projection_settings(self) -> bool:
        model_type = canonical_model_type(self.baseline_model_type_combo_box.currentText())
        if not uses_corner_projection(model_type):
            return True
        known: set[str] = set()
        if self.last_scan_result is not None:
            known = set(self.last_scan_result.get("active_input_feature_names", [])) | set(
                self.last_scan_result.get("dropped_input_feature_names", [])
            )

        selected = self._selected_projection_columns()
        if not selected:
            self._show_warning(
                f"{model_type} needs at least one PVT corner column. Check the "
                "corner/condition columns (e.g. temperature, supply, or process "
                "one-hots) in the Baseline tab's PVT Corner Columns list."
            )
            return False
        unknown = [name for name in selected if name not in known]
        if known and unknown:
            self._show_warning(
                "These PVT corner columns are not input features of the scanned "
                "dataset: " + ", ".join(unknown) + ". Re-select the corner columns "
                "for this dataset."
            )
            return False
        return True

    def _build_baseline_train_config(self) -> TrainConfig:
        roots = make_run_roots(self.model_output_folder_path_edit.text().strip(), self.run_name_edit.text().strip())
        form = self._collect_baseline_form()
        # One .npz names the whole dataset, so it IS the data root; the explicit
        # input-feature / ground-truth overrides are command-line only now.
        return TrainConfig(
            data_root=self.dataset_file_edit.text().strip() or None,
            input_feature_path=None,
            ground_truth_data_dir=None,
            cache_path=self._ensure_cache_path(),
            output_dir=roots["baseline"],
            model_type=form["model_type"],
            projection_columns=form["projection_columns"] or None,
            projection_dim=form["projection_dim"],
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

    def _update_start_button_text(self) -> None:
        return

    def _start_blocked_reason(self) -> str | None:
        """Why the start buttons are disabled, or None when they are enabled.

        This covers only the two conditions that DISABLE the buttons. The
        remaining prerequisites (a dataset file, a completed scan, valid split
        fractions, corner columns) are checked when the button is clicked and
        already explain themselves in a dialog.

        A disabled button never emits clicked, so those dialogs are unreachable
        and the user is left with a dead control and no stated reason. This text
        becomes the tooltip, which Qt still shows on a disabled widget.
        """
        if self._controls_locked:
            running = self.current_task_name if self.current_task_name != "idle" else "background"
            return f"A {running} task is running. Wait for it to finish, or press Stop."
        if not self._license_allows_new_runs():
            return self._license_display_message()
        return None

    def _refresh_run_button_availability(self) -> None:
        blocked_reason = self._start_blocked_reason()
        start_enabled = blocked_reason is None
        self.start_baseline_button.setEnabled(start_enabled)
        # Qt shows a tooltip on a disabled widget, which is the only channel left
        # for telling the user what to fix.
        self.start_baseline_button.setToolTip("" if start_enabled else blocked_reason)

    def _set_action_controls_enabled(self, enabled: bool) -> None:
        widgets = [
            self.training_device_combo_box,
            self.suggest_initial_settings_button,
            self.start_baseline_button,
            self.open_output_folder_button,
            self.export_run_summary_button,
            self.export_onnx_button,
            self.apply_initial_settings_button,
            self.restore_recommended_baseline_button,
            self.dataset_file_edit,
            self.browse_dataset_file_button,
            self.model_output_folder_path_edit,
            self.browse_model_output_folder_button,
            self.run_name_edit,
            self.training_tabs,
            self.license_server_url_edit,
            self.test_license_connection_button,
            self.acquire_license_seat_button,
        ]
        self._controls_locked = not enabled
        for widget in widgets:
            widget.setEnabled(enabled)
        self._refresh_run_button_availability()

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------
    def _reset_baseline_plots(self) -> None:
        self._reset_channel_metric_cards()
        # A run whose internal test fold was merged into training never emits an
        # average MAE, so a stale value from the previous run would otherwise stay
        # on the card and read as if it belonged to this one.
        self.metric_cards["average_mae"].set_value("-")
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

    def _add_form_row(self, layout: QGridLayout, row: int, label_text: str, widget: QWidget) -> QLabel:
        """Add a label/widget row and return the label.

        Callers that may need to hide the row keep the label: hiding only the
        widget would leave its caption behind, captioning the row below it.
        """
        label = QLabel(label_text)
        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1, 1, 2)
        return label

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

    def _validate_split_fractions(self) -> bool:
        train_frac = float(self.baseline_train_fraction_spin_box.value())
        val_frac = float(self.baseline_validation_fraction_spin_box.value())
        if not 0.0 < train_frac < 1.0:
            self._show_warning("Train fraction must be between 0 and 1.")
            return False
        if not 0.0 <= val_frac < 1.0:
            self._show_warning("Validation fraction must be between 0 and 1.")
            return False
        if train_frac + val_frac > 1.0:
            self._show_warning("Train fraction plus validation fraction cannot exceed 1.0.")
            return False
        return True

    def _preferred_output_path(self) -> str | None:
        if self.last_workflow_summary:
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

    def _set_cache_path_value(self, cache_path: str) -> None:
        """Record the cache file the engine reported, for display only.

        The Cache File field is gone: the cache is always derived from the output
        folder and run name, so there is nothing for a user to choose and nothing
        to keep in sync with a widget.
        """
        self._cache_path_value = cache_path.strip()

    def _sync_auto_cache_path(self) -> None:
        self._set_cache_path_value(self._default_cache_path())

    def _ensure_cache_path(self) -> str:
        cache_path = self._default_cache_path()
        self._set_cache_path_value(cache_path)
        return cache_path

    def _require_data_paths(self) -> dict[str, str] | None:
        dataset_file = self.dataset_file_edit.text().strip()
        model_output_dir = self.model_output_folder_path_edit.text().strip()

        if not dataset_file:
            self._show_warning("Select a .npz dataset file first.")
            return None
        # Catch a missing or mistyped path here rather than inside a worker
        # thread, where it would surface as a task traceback.
        if not Path(dataset_file).is_file():
            self._show_warning(f"The dataset file does not exist:\n{dataset_file}")
            return None
        if not model_output_dir:
            self._show_warning("Select the output folder first.")
            return None
        if not self.run_name_edit.text().strip():
            self._show_warning("Provide a run name before starting.")
            return None
        Path(model_output_dir).mkdir(parents=True, exist_ok=True)
        return {
            "dataset_root": dataset_file,
            "input_feature_path": "",
            "ground_truth_data_dir": "",
            "cache_path": self._ensure_cache_path(),
        }

    def _autofill_run_name_from_dataset_file(self) -> None:
        """Name the run after the dataset file and mark it unscanned.

        A generic bundle name (cache.npz, bundle.npz) says nothing about which
        dataset it is, so those fall back to the containing folder's name.
        """
        dataset_file = self.dataset_file_edit.text().strip()
        if dataset_file:
            path = Path(dataset_file)
            stem = path.stem
            if stem.lower() in ("cache", "bundle", "dataset", "data"):
                stem = path.parent.name or stem
            self.run_name_edit.setText(stem.replace(" ", "_").lower())
            self.dataset_schema_status_badge.set_status("Not Scanned")
        # A scan describes one dataset. Keeping the previous result would let a
        # run start against a dataset the badge says was never scanned, and would
        # validate corner columns against the old dataset's column names.
        self.last_scan_result = None
        self._sync_auto_cache_path()
        self._update_topbar_run_name()
        self._refresh_run_button_availability()

    def _note_removed_transfer_settings(self, transfer: Any) -> None:
        """Report a saved self-transfer setup that this GUI no longer runs.

        Frequency-domain self-transfer was removed from the GUI; the engine still
        implements it and run_self_transfer.py still drives it. A restored session
        may carry a tuned band setup, and dropping it silently would look like the
        run still trains per-band when it now trains one model over the whole
        frequency range.
        """
        if not isinstance(transfer, dict) or not transfer:
            return
        bands = transfer.get("num_bands")
        detail = f" ({bands} frequency bands)" if bands else ""
        self.append_log(
            f"This configuration set up frequency-domain self-transfer learning{detail}. "
            "That workflow has been removed from the GUI, so this run will train a single "
            "baseline model across the whole frequency range. Self-transfer is still "
            "available from the command line via run_self_transfer.py."
        )

    def _note_removed_data_source_fields(self, data_sources: dict[str, Any]) -> None:
        """Report saved values for fields this GUI no longer has.

        A session or config written before the .npz-only data loading may carry a
        dataset folder or an explicit input-feature / ground-truth / cache path.
        Restoring silently would look like the old dataset had loaded, when in
        fact no dataset is selected at all.
        """
        folder = str(data_sources.get("dataset_folder", "")).strip()
        if folder and not self.dataset_file_edit.text().strip():
            self.append_log(
                f"This session used the dataset folder {folder}. The GUI now loads one "
                ".npz dataset file, so no dataset is selected — choose the .npz for this "
                "dataset. Raw folder layouts are still supported by train_baseline.py "
                "--data-root."
            )
        removed = {
            "an explicit input-feature file": ("input_feature_path", "--input-feature-path"),
            "an explicit ground-truth folder": ("ground_truth_data_dir", "--ground-truth-data-dir"),
            "a custom cache file": ("cache_path", "--cache-path"),
        }
        for description, (key, flag) in removed.items():
            value = str(data_sources.get(key, "")).strip()
            # A saved auto-managed cache path is not a user choice, so restoring
            # without it changes nothing worth reporting.
            if key == "cache_path" and (not value or not data_sources.get("cache_path_manually_selected")):
                continue
            if value:
                self.append_log(
                    f"This session set {description} ({value}). That field has been removed "
                    f"from the GUI; it is still available via train_baseline.py {flag}."
                )

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
        average_mae = summary.get("average_evaluation_mae", summary.get("average_test_mae"))
        if average_mae is not None:
            self.metric_cards["average_mae"].set_value(f"{float(average_mae):.6f}")
        self._update_channel_metric_cards(list(summary.get("channel_mae_with_units") or []))
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
        return f"[{phase}] {message}" if message else ""

    def append_log(self, text: str) -> None:
        self.run_log_text_edit.appendPlainText(text)
        self.run_log_text_edit.verticalScrollBar().setValue(self.run_log_text_edit.verticalScrollBar().maximum())

    def _show_warning(self, text: str) -> None:
        QMessageBox.warning(self, "Surrogate Model Training Suite", text)

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
            # Workers poll should_stop between batches/samples, so give them time
            # to unwind. Destroying a QThread that is still running aborts the
            # whole process, so fall back to terminate() as the lesser evil.
            if not self.current_task.thread.wait(10000):
                self.current_task.thread.terminate()
                self.current_task.thread.wait(2000)
        self.license_controller.shutdown()
        super().closeEvent(event)


def create_application() -> tuple[QApplication, MlpTrainingStudio]:
    app = QApplication.instance() or QApplication([])
    apply_application_theme(app)
    window = MlpTrainingStudio()
    return app, window
