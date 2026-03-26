"""Deterministic schema builder from confirmed role assignments.

After the scanner finds files and the classifier (LLM or heuristic) suggests
roles, the user confirms the assignments in the wizard UI.  This module takes
those confirmed assignments and produces a ``DatasetConfig`` — a structured
description of how to load the dataset for training.

This is purely code — no LLM calls, no guessing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dataset_classifier import ClassificationResult, RoleAssignment
from .dataset_scanner import ScanResult


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class InputConfig:
    """Describes how to load input features (geometric parameters)."""

    # Path to the tabular file(s) with input parameters.
    # For a single file: ["tabular/data.csv"]
    # For per-template files: ["tabular/oct_l.csv", "tabular/rec_r.csv", ...]
    file_paths: list[str] = field(default_factory=list)
    # Column names that are design parameters (model inputs).
    feature_columns: list[str] = field(default_factory=list)
    # Column used to match rows to ground truth files.
    sample_id_column: str = ""
    # Whether parameters are in a separate file or embedded in ground truth.
    source: str = "file"  # "file" or "embedded"


@dataclass
class GroundTruthConfig:
    """Describes how to load ground truth (simulation output)."""

    # Loading strategy identifier.
    format: str = ""  # "inline", "per_sample", "array"
    # Path pattern for per-sample files (e.g. "csv/{template}/").
    file_paths: list[str] = field(default_factory=list)
    # File extension for per-sample files.
    file_extension: str = ""
    # Column names in per-sample CSVs.
    columns: list[str] = field(default_factory=list)
    # S-parameters or channel names.
    output_parameters: list[str] = field(default_factory=list)
    # Parts: ["re", "im"], ["mag", "db", "angle_deg"], etc.
    output_parts: list[str] = field(default_factory=list)
    # Number of frequency points per sample.
    num_frequencies: int | None = None
    # Whether to skip the first frequency (DC point).
    drop_first_frequency: bool = False
    # For numpy arrays: path to the pkl/npy/npz file.
    array_path: str = ""
    # Array shape description.
    array_shape: list[int] | None = None


@dataclass
class LayoutConfig:
    """Describes layout / image data (optional, for future use)."""

    file_paths: list[str] = field(default_factory=list)
    format: str = ""  # "png", "numpy_array", "gds", "pixel_csv"
    array_path: str = ""
    array_shape: list[int] | None = None


@dataclass
class DatasetConfig:
    """Complete dataset configuration built from confirmed role assignments."""

    dataset_name: str = ""
    root_dir: str = ""
    input_config: InputConfig = field(default_factory=InputConfig)
    ground_truth_config: GroundTruthConfig = field(default_factory=GroundTruthConfig)
    layout_configs: list[LayoutConfig] = field(default_factory=list)
    metadata_paths: list[str] = field(default_factory=list)
    # Templates / variants detected.
    templates: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_config(
    scan: ScanResult,
    classification: ClassificationResult,
) -> DatasetConfig:
    """Build a DatasetConfig from confirmed scan + classification results.

    This function is deterministic — it reads the assignments and builds the
    config from the facts in the FileInfo objects.
    """
    root = Path(scan.root)
    config = DatasetConfig(
        dataset_name=root.name,
        root_dir=scan.root,
    )

    input_assignments: list[RoleAssignment] = []
    gt_assignments: list[RoleAssignment] = []
    layout_assignments: list[RoleAssignment] = []
    metadata_assignments: list[RoleAssignment] = []

    for a in classification.assignments:
        if a.role == "input_parameters":
            input_assignments.append(a)
        elif a.role == "ground_truth":
            gt_assignments.append(a)
        elif a.role == "layout":
            layout_assignments.append(a)
        elif a.role == "metadata":
            metadata_assignments.append(a)

    _build_input_config(config, input_assignments)
    _build_ground_truth_config(config, gt_assignments)
    _build_layout_configs(config, layout_assignments)
    config.metadata_paths = [a.file_info.rel_path for a in metadata_assignments]

    # Detect templates from repeated directory patterns.
    _detect_templates(config, gt_assignments)

    return config


def config_to_dict(config: DatasetConfig) -> dict[str, Any]:
    """Serialize a DatasetConfig to a JSON-compatible dictionary."""
    return {
        "dataset_name": config.dataset_name,
        "root_dir": config.root_dir,
        "templates": config.templates,
        "input_feature": {
            "file_paths": config.input_config.file_paths,
            "feature_columns": config.input_config.feature_columns,
            "sample_id_column": config.input_config.sample_id_column,
            "source": config.input_config.source,
        },
        "ground_truth": {
            "source": config.ground_truth_config.format,
            "file_paths": config.ground_truth_config.file_paths,
            "file_extension": config.ground_truth_config.file_extension,
            "columns": config.ground_truth_config.columns,
            "output_parameters": config.ground_truth_config.output_parameters,
            "output_parts": config.ground_truth_config.output_parts,
            "num_frequencies": config.ground_truth_config.num_frequencies,
            "drop_first_frequency": config.ground_truth_config.drop_first_frequency,
            "array_path": config.ground_truth_config.array_path,
            "array_shape": config.ground_truth_config.array_shape,
        },
        "layout": [
            {
                "file_paths": lc.file_paths,
                "format": lc.format,
                "array_path": lc.array_path,
                "array_shape": lc.array_shape,
            }
            for lc in config.layout_configs
        ],
        "metadata_paths": config.metadata_paths,
    }


def save_config(config: DatasetConfig, path: str | Path) -> Path:
    """Save a DatasetConfig as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(config), indent=2), encoding="utf-8")
    return path


def config_to_readme(config: DatasetConfig, output_dir: str | Path) -> Path:
    """Generate a README.md with a JSON schema block from a DatasetConfig.

    The generated README is compatible with the ``dataset_schema.py`` parser
    so the existing ``build_cache`` pipeline can load the data.
    """
    output_dir = Path(output_dir)
    ic = config.input_config
    gt = config.ground_truth_config

    # Build the schema JSON that dataset_schema.py can parse.
    all_columns = ic.feature_columns[:]
    if ic.sample_id_column and ic.sample_id_column not in all_columns:
        all_columns.append(ic.sample_id_column)

    schema: dict = {
        "dataset_name": config.dataset_name,
        "input_feature": {
            "columns": all_columns,
            "feature_columns": ic.feature_columns,
        },
        "ground_truth": {
            "source": gt.format or "per_sample",
            "ground_truth_parameters": gt.output_parameters,
            "ground_truth_parts": gt.output_parts or ["re", "im"],
            "drop_first_frequency": gt.drop_first_frequency,
        },
    }

    if ic.sample_id_column:
        schema["input_feature"]["sample_id_column"] = ic.sample_id_column

    if gt.file_extension:
        schema["ground_truth"]["file_extension"] = gt.file_extension

    schema_json = json.dumps(schema, indent=2)

    # Build the README text.
    lines = [
        f"# {config.dataset_name}",
        "",
        f"Auto-generated by Dataset Wizard.",
        "",
    ]

    if config.templates:
        lines.append(f"**Templates**: {', '.join(config.templates)}")
        lines.append("")

    lines.append("## Input Features")
    lines.append("")
    lines.append(f"**Columns**: {', '.join(ic.feature_columns)}")
    if ic.sample_id_column:
        lines.append(f"**Sample ID**: {ic.sample_id_column}")
    if ic.file_paths:
        lines.append(f"**Source files**: {', '.join(ic.file_paths)}")
    lines.append("")

    lines.append("## Ground Truth")
    lines.append("")
    lines.append(f"**Format**: {gt.format}")
    if gt.output_parameters:
        lines.append(f"**Parameters**: {', '.join(gt.output_parameters)}")
    if gt.num_frequencies:
        lines.append(f"**Frequencies**: {gt.num_frequencies} points")
    if gt.file_paths:
        lines.append(f"**Data directories**: {', '.join(gt.file_paths)}")
    lines.append("")

    lines.append("## Schema")
    lines.append("")
    lines.append("```json")
    lines.append(schema_json)
    lines.append("```")
    lines.append("")

    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return readme_path


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _build_input_config(config: DatasetConfig, assignments: list[RoleAssignment]) -> None:
    """Build InputConfig from input_parameters assignments."""
    if not assignments:
        return

    ic = config.input_config
    ic.source = "file"

    # Collect file paths and merge feature columns.
    all_feature_cols: list[str] = []
    id_col = ""
    for a in assignments:
        fi = a.file_info
        ic.file_paths.append(fi.rel_path)

        # Use the assignment's feature_columns if provided.
        if a.feature_columns:
            for c in a.feature_columns:
                if c not in all_feature_cols:
                    all_feature_cols.append(c)

        if a.id_column:
            id_col = a.id_column

    ic.feature_columns = all_feature_cols
    ic.sample_id_column = id_col


def _build_ground_truth_config(config: DatasetConfig, assignments: list[RoleAssignment]) -> None:
    """Build GroundTruthConfig from ground_truth assignments."""
    if not assignments:
        return

    gt = config.ground_truth_config

    for a in assignments:
        fi = a.file_info

        # Per-sample CSV groups.
        if fi.is_group and fi.extension == ".csv":
            gt.format = gt.format or "per_sample"
            gt.file_paths.append(fi.rel_path)
            gt.file_extension = fi.extension
            if fi.columns and not gt.columns:
                gt.columns = fi.columns
                # Extract S-parameter names from column names.
                _extract_sparams(gt, fi.columns)
            if fi.row_count and not gt.num_frequencies:
                gt.num_frequencies = fi.row_count

        # Touchstone file groups.
        elif fi.is_group and fi.extra.get("sparams"):
            gt.format = gt.format or "per_sample"
            gt.file_paths.append(fi.rel_path)
            gt.file_extension = fi.extension
            gt.output_parameters = fi.extra["sparams"]
            gt.output_parts = ["re", "im"]
            if fi.extra.get("num_frequencies"):
                gt.num_frequencies = fi.extra["num_frequencies"]
            if fi.extra.get("first_frequency") == 0.0:
                gt.drop_first_frequency = True

        # Pre-packed numpy/pickle arrays.
        elif fi.extension in (".pkl", ".npy", ".npz") and fi.shape and len(fi.shape) == 3:
            gt.format = gt.format or "array"
            gt.array_path = fi.rel_path
            gt.array_shape = fi.shape
            gt.num_frequencies = fi.shape[1] if len(fi.shape) >= 2 else None


def _extract_sparams(gt: GroundTruthConfig, columns: list[str]) -> None:
    """Extract S-parameter names and parts from CSV column names."""
    import re
    sparam_pattern = re.compile(r"^s(\d)(\d)_(real|imag|mag|db|angle_deg|re|im)$", re.IGNORECASE)
    seen_params: list[str] = []
    seen_parts: set[str] = set()

    for col in columns:
        m = sparam_pattern.match(col)
        if m:
            i, j = int(m.group(1)), int(m.group(2))
            # Normalize to upper-triangular: Sij where i <= j.
            if i > j:
                i, j = j, i
            param = f"S{i}{j}"
            if param not in seen_params:
                seen_params.append(param)
            part = m.group(3).lower()
            # Normalize part names.
            if part == "real":
                part = "re"
            elif part == "imag":
                part = "im"
            seen_parts.add(part)

    if seen_params:
        gt.output_parameters = seen_params
    if seen_parts:
        gt.output_parts = sorted(seen_parts)


def _build_layout_configs(config: DatasetConfig, assignments: list[RoleAssignment]) -> None:
    """Build LayoutConfig entries from layout assignments."""
    for a in assignments:
        fi = a.file_info
        lc = LayoutConfig()

        if fi.extension in (".pkl", ".npy") and fi.shape:
            lc.format = "numpy_array"
            lc.array_path = fi.rel_path
            lc.array_shape = fi.shape
        elif fi.is_group:
            lc.format = fi.extension.lstrip(".")
            lc.file_paths = [fi.rel_path]
        else:
            continue

        config.layout_configs.append(lc)


def _detect_templates(config: DatasetConfig, gt_assignments: list[RoleAssignment]) -> None:
    """Detect template names from ground truth directory paths."""
    templates: list[str] = []
    for a in gt_assignments:
        fi = a.file_info
        if fi.is_group:
            # The rel_path for a group is the directory, e.g. "csv/oct_l".
            parts = Path(fi.rel_path).parts
            if len(parts) >= 2:
                template = parts[-1]
                if template not in templates:
                    templates.append(template)
    config.templates = templates
