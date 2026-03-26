"""Dataset Wizard dialog for PySide6.

A multi-step dialog that:
1. Scans a user-selected directory.
2. Shows all detected files with LLM-suggested (or heuristic) role assignments.
3. Lets the user confirm, edit roles, and select feature/ID columns.
4. Produces a DatasetConfig for the data loading pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .dataset_classifier import (
    ROLE_DESCRIPTIONS,
    ROLES,
    ClassificationResult,
    RoleAssignment,
    classify_dataset,
)
from .dataset_scanner import ScanResult, scan_directory
from .dataset_wizard_core import DatasetConfig, build_config, config_to_dict


# ---------------------------------------------------------------------------
# Background worker for scanning + classification
# ---------------------------------------------------------------------------

class _ScanWorker(QThread):
    """Run scan + classify in a background thread."""

    finished = Signal(object, object)  # (ScanResult, ClassificationResult)
    error = Signal(str)

    def __init__(self, directory: str, api_key: str | None, parent=None):
        super().__init__(parent)
        self._directory = directory
        self._api_key = api_key

    def run(self):
        try:
            scan = scan_directory(self._directory, max_depth=2)
            classification = classify_dataset(scan, api_key=self._api_key)
            self.finished.emit(scan, classification)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Wizard Dialog
# ---------------------------------------------------------------------------

class DatasetWizardDialog(QDialog):
    """Three-step wizard: Select Folder → Review Assignments → Confirm Config."""

    # Emitted when the user accepts the final config.
    config_ready = Signal(object)  # DatasetConfig

    def __init__(self, parent=None, api_key: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Dataset Wizard")
        self.setMinimumSize(900, 620)
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

        self._scan: ScanResult | None = None
        self._classification: ClassificationResult | None = None
        self._config: DatasetConfig | None = None
        self._worker: _ScanWorker | None = None

        self._build_ui()

    # ---------------------------------------------------------------
    # UI construction
    # ---------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)

        # Stacked pages.
        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_page_select())     # 0
        self._pages.addWidget(self._build_page_review())      # 1
        self._pages.addWidget(self._build_page_confirm())      # 2
        outer.addWidget(self._pages, 1)

        # Navigation buttons.
        nav = QHBoxLayout()
        self._back_btn = QPushButton("Back")
        self._back_btn.clicked.connect(self._go_back)
        self._next_btn = QPushButton("Scan && Classify")
        self._next_btn.clicked.connect(self._go_next)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        nav.addWidget(self._back_btn)
        nav.addStretch(1)
        nav.addWidget(self._cancel_btn)
        nav.addWidget(self._next_btn)
        outer.addLayout(nav)

        self._update_nav()

    # --- Page 0: Select folder ---

    def _build_page_select(self) -> QWidget:
        page = QWidget()
        vbox = QVBoxLayout(page)

        title = QLabel("Step 1: Select Dataset Folder")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        vbox.addWidget(title)

        desc = QLabel(
            "Choose the root folder that contains your entire dataset — "
            "input parameters, simulation results, layout files, etc.  "
            "The wizard will scan everything inside and figure out what's what."
        )
        desc.setWordWrap(True)
        vbox.addWidget(desc)

        row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Enter or browse to dataset directory...")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_dir)
        row.addWidget(self._dir_edit, 1)
        row.addWidget(browse_btn)
        vbox.addLayout(row)

        self._scan_status = QLabel("")
        vbox.addWidget(self._scan_status)
        vbox.addStretch(1)
        return page

    # --- Page 1: Review assignments ---

    def _build_page_review(self) -> QWidget:
        page = QWidget()
        vbox = QVBoxLayout(page)

        title = QLabel("Step 2: Review File Roles")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        vbox.addWidget(title)

        desc = QLabel(
            "Each file or file group has been assigned a role.  "
            "Review the assignments below and change any that are incorrect."
        )
        desc.setWordWrap(True)
        vbox.addWidget(desc)

        self._summary_label = QLabel("")
        self._summary_label.setWordWrap(True)
        vbox.addWidget(self._summary_label)

        # Table: Path | Summary | Role (dropdown) | Confidence
        self._review_table = QTableWidget()
        self._review_table.setColumnCount(4)
        self._review_table.setHorizontalHeaderLabels(["Path", "Description", "Role", "Confidence"])
        self._review_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self._review_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._review_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._review_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self._review_table.setColumnWidth(2, 160)
        self._review_table.setColumnWidth(3, 80)
        self._review_table.verticalHeader().setVisible(False)
        self._review_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        vbox.addWidget(self._review_table, 1)

        # Detail panel for selected row.
        detail_group = QGroupBox("Selected File Details")
        detail_layout = QVBoxLayout(detail_group)
        self._detail_text = QTextEdit()
        self._detail_text.setReadOnly(True)
        self._detail_text.setMaximumHeight(120)
        detail_layout.addWidget(self._detail_text)
        vbox.addWidget(detail_group)

        self._review_table.currentCellChanged.connect(self._on_review_row_changed)
        return page

    # --- Page 2: Confirm config ---

    def _build_page_confirm(self) -> QWidget:
        page = QWidget()
        vbox = QVBoxLayout(page)

        title = QLabel("Step 3: Confirm Configuration")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        vbox.addWidget(title)

        desc = QLabel(
            "Review the final dataset configuration.  "
            "You can edit the feature columns and sample ID column below."
        )
        desc.setWordWrap(True)
        vbox.addWidget(desc)

        # Input features section.
        input_group = QGroupBox("Input Features")
        input_layout = QGridLayout(input_group)

        input_layout.addWidget(QLabel("Feature Columns:"), 0, 0, Qt.AlignmentFlag.AlignTop)
        self._feature_list = QListWidget()
        self._feature_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._feature_list.setMaximumHeight(140)
        input_layout.addWidget(self._feature_list, 0, 1)

        input_layout.addWidget(QLabel("Sample ID Column:"), 1, 0)
        self._id_col_combo = QComboBox()
        input_layout.addWidget(self._id_col_combo, 1, 1)

        vbox.addWidget(input_group)

        # Ground truth section.
        gt_group = QGroupBox("Ground Truth")
        gt_layout = QGridLayout(gt_group)

        self._gt_format_label = QLabel("")
        gt_layout.addWidget(QLabel("Format:"), 0, 0)
        gt_layout.addWidget(self._gt_format_label, 0, 1)

        self._gt_params_label = QLabel("")
        self._gt_params_label.setWordWrap(True)
        gt_layout.addWidget(QLabel("Output Parameters:"), 1, 0)
        gt_layout.addWidget(self._gt_params_label, 1, 1)

        self._gt_freq_label = QLabel("")
        gt_layout.addWidget(QLabel("Frequencies:"), 2, 0)
        gt_layout.addWidget(self._gt_freq_label, 2, 1)

        self._gt_templates_label = QLabel("")
        gt_layout.addWidget(QLabel("Templates:"), 3, 0)
        gt_layout.addWidget(self._gt_templates_label, 3, 1)

        vbox.addWidget(gt_group)

        # JSON preview.
        json_group = QGroupBox("Configuration JSON (read-only)")
        json_layout = QVBoxLayout(json_group)
        self._json_preview = QTextEdit()
        self._json_preview.setReadOnly(True)
        self._json_preview.setMaximumHeight(200)
        json_layout.addWidget(self._json_preview)
        vbox.addWidget(json_group)

        return page

    # ---------------------------------------------------------------
    # Navigation
    # ---------------------------------------------------------------

    def _update_nav(self) -> None:
        idx = self._pages.currentIndex()
        self._back_btn.setVisible(idx > 0)
        if idx == 0:
            self._next_btn.setText("Scan && Classify")
            self._next_btn.setEnabled(True)
        elif idx == 1:
            self._next_btn.setText("Build Config")
            self._next_btn.setEnabled(True)
        elif idx == 2:
            self._next_btn.setText("Accept")
            self._next_btn.setEnabled(True)

    def _go_back(self) -> None:
        idx = self._pages.currentIndex()
        if idx > 0:
            self._pages.setCurrentIndex(idx - 1)
            self._update_nav()

    def _go_next(self) -> None:
        idx = self._pages.currentIndex()
        if idx == 0:
            self._start_scan()
        elif idx == 1:
            self._build_and_show_config()
            self._pages.setCurrentIndex(2)
            self._update_nav()
        elif idx == 2:
            self._accept_config()

    # ---------------------------------------------------------------
    # Page 0 logic: scan
    # ---------------------------------------------------------------

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if path:
            self._dir_edit.setText(path)

    def _start_scan(self) -> None:
        directory = self._dir_edit.text().strip()
        if not directory or not Path(directory).is_dir():
            self._scan_status.setText("Please select a valid directory.")
            return

        self._scan_status.setText("Scanning and classifying...")
        self._next_btn.setEnabled(False)

        self._worker = _ScanWorker(directory, self._api_key, self)
        self._worker.finished.connect(self._on_scan_done)
        self._worker.error.connect(self._on_scan_error)
        self._worker.start()

    def _on_scan_done(self, scan: ScanResult, classification: ClassificationResult) -> None:
        self._scan = scan
        self._classification = classification
        self._scan_status.setText(f"Found {len(scan.files)} files/groups.")
        self._populate_review_table()
        self._pages.setCurrentIndex(1)
        self._update_nav()

    def _on_scan_error(self, error: str) -> None:
        self._scan_status.setText(f"Error: {error}")
        self._next_btn.setEnabled(True)

    # ---------------------------------------------------------------
    # Page 1 logic: review assignments
    # ---------------------------------------------------------------

    def _populate_review_table(self) -> None:
        if not self._classification:
            return

        self._summary_label.setText(
            self._classification.dataset_summary or "Dataset scanned successfully."
        )

        assignments = self._classification.assignments
        self._review_table.setRowCount(len(assignments))
        self._role_combos: list[QComboBox] = []

        for row, a in enumerate(assignments):
            # Path.
            path_item = QTableWidgetItem(a.file_info.rel_path)
            path_item.setFlags(path_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._review_table.setItem(row, 0, path_item)

            # Summary.
            summary = a.file_info.summary
            if a.file_info.is_group:
                summary = f"[{a.file_info.file_count} files] {summary}"
            summary_item = QTableWidgetItem(summary)
            summary_item.setFlags(summary_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._review_table.setItem(row, 1, summary_item)

            # Role dropdown.
            combo = QComboBox()
            for role in ROLES:
                combo.addItem(f"{role}", role)
            combo.setCurrentIndex(ROLES.index(a.role) if a.role in ROLES else len(ROLES) - 1)
            self._review_table.setCellWidget(row, 2, combo)
            self._role_combos.append(combo)

            # Confidence.
            conf_item = QTableWidgetItem(a.confidence)
            conf_item.setFlags(conf_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._review_table.setItem(row, 3, conf_item)

        self._review_table.resizeColumnsToContents()
        # Re-apply stretch on description column.
        self._review_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

    def _on_review_row_changed(self, row: int, _col: int, _prev_row: int, _prev_col: int) -> None:
        if not self._classification or row < 0 or row >= len(self._classification.assignments):
            return
        a = self._classification.assignments[row]
        fi = a.file_info
        lines = [
            f"Path: {fi.rel_path}",
            f"Extension: {fi.extension}",
            f"Reason: {a.reason}",
        ]
        if fi.columns:
            lines.append(f"Columns: {fi.columns}")
        if fi.shape:
            lines.append(f"Shape: {fi.shape}")
        if fi.row_count is not None:
            lines.append(f"Rows: {fi.row_count}")
        if fi.value_ranges:
            for col, vr in list(fi.value_ranges.items())[:5]:
                lines.append(f"  {col}: {vr['min']:.4g} .. {vr['max']:.4g}")
        if fi.constant_columns:
            lines.append(f"Constant columns: {fi.constant_columns}")
        if a.feature_columns:
            lines.append(f"Feature columns: {a.feature_columns}")
        if a.id_column:
            lines.append(f"ID column: {a.id_column}")
        self._detail_text.setText("\n".join(lines))

    # ---------------------------------------------------------------
    # Page 2 logic: build and confirm config
    # ---------------------------------------------------------------

    def _build_and_show_config(self) -> None:
        if not self._scan or not self._classification:
            return

        # Update classification from the user's role edits.
        for i, combo in enumerate(self._role_combos):
            if i < len(self._classification.assignments):
                self._classification.assignments[i].role = combo.currentData()

        self._config = build_config(self._scan, self._classification)
        self._populate_confirm_page()

    def _populate_confirm_page(self) -> None:
        if not self._config:
            return

        ic = self._config.input_config
        gt = self._config.ground_truth_config

        # Feature columns list — user can toggle on/off.
        self._feature_list.clear()
        all_cols = ic.feature_columns[:]
        # Also add any columns from input files that aren't in feature_columns yet.
        for a in (self._classification.assignments if self._classification else []):
            if a.role == "input_parameters" and a.file_info.columns:
                for c in a.file_info.columns:
                    if c not in all_cols:
                        all_cols.append(c)
        for col in all_cols:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if col in ic.feature_columns
                else Qt.CheckState.Unchecked
            )
            self._feature_list.addItem(item)

        # ID column combo.
        self._id_col_combo.clear()
        self._id_col_combo.addItem("(none)", "")
        for col in all_cols:
            self._id_col_combo.addItem(col, col)
        if ic.sample_id_column:
            idx = self._id_col_combo.findData(ic.sample_id_column)
            if idx >= 0:
                self._id_col_combo.setCurrentIndex(idx)

        # Ground truth summary.
        self._gt_format_label.setText(gt.format or "Unknown")
        self._gt_params_label.setText(", ".join(gt.output_parameters) if gt.output_parameters else "N/A")
        freq_text = f"{gt.num_frequencies} points" if gt.num_frequencies else "Unknown"
        if gt.drop_first_frequency:
            freq_text += " (first freq dropped)"
        self._gt_freq_label.setText(freq_text)
        self._gt_templates_label.setText(", ".join(self._config.templates) if self._config.templates else "Single dataset")

        # JSON preview.
        import json
        self._json_preview.setText(json.dumps(config_to_dict(self._config), indent=2))

    def _accept_config(self) -> None:
        if not self._config:
            return

        # Apply user's final column selections.
        ic = self._config.input_config
        ic.feature_columns = []
        for i in range(self._feature_list.count()):
            item = self._feature_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                ic.feature_columns.append(item.text())

        ic.sample_id_column = self._id_col_combo.currentData() or ""

        self.config_ready.emit(self._config)
        self.accept()

    # ---------------------------------------------------------------
    # Public access
    # ---------------------------------------------------------------

    def get_config(self) -> DatasetConfig | None:
        return self._config
