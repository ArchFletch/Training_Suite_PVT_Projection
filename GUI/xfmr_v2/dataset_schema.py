"""Parse and validate dataset schema metadata stored in README files.

This module is the bridge between human-written dataset documentation and the
rest of the codebase. The overall flow is:

1. Read a dataset README.
2. Extract a fenced JSON block.
3. Validate that the block contains the fields the loaders rely on.
4. Convert the raw dictionary into small immutable dataclasses.

By concentrating that logic here, the data loader can assume it is working with
clean, already-validated schema objects.

Supported ground-truth formats:
- ``touchstone``: per-sample Touchstone files with S-parameter data.
- ``cadence_csv``: Cadence Ocean CSV exports where input parameters and
  swept output values are embedded together in block-structured files.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# README schemas live inside a fenced JSON block.
README_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

SUPPORTED_FORMATS = ("touchstone", "cadence_csv")


# Input-feature metadata.
# This block describes the tabular file that provides per-sample input features.
@dataclass(frozen=True)
class InputFeatureSchema:
    # `columns` preserves the exact order found in the input-feature file.
    columns: tuple[str, ...]
    # `feature_columns` is the subset of columns that the model should consume as input features.
    feature_columns: tuple[str, ...]
    # `sample_id_column` points to the column used to match each input-feature row to its
    # ground-truth file.  Empty string for embedded-parameter formats (cadence_csv).
    sample_id_column: str = ""
    # "file" means a separate tabular file provides the features (touchstone workflow).
    # "embedded" means features are extracted from the ground-truth files (cadence_csv).
    source: str = "file"
    # Maps CSV parameter key → (column_name, scale_factor).  Only used when source="embedded".
    # Stored as a sorted tuple of (csv_key, column, scale) for hashability.
    parameter_keys: tuple[tuple[str, str, float], ...] = ()

    @property
    def expected_width(self) -> int:
        return len(self.columns)

    @property
    def feature_indices(self) -> tuple[tuple[str, int], ...]:
        return tuple((name, self.columns.index(name)) for name in self.feature_columns)

    @property
    def sample_id_index(self) -> int:
        return self.columns.index(self.sample_id_column)

    @property
    def parameter_key_map(self) -> dict[str, tuple[str, float]]:
        """Return {csv_key: (column_name, scale)} for convenient lookup."""
        return {csv_key: (col, scale) for csv_key, col, scale in self.parameter_keys}


# Ground-truth metadata.
# This block describes how to read each per-sample ground-truth artifact.
@dataclass(frozen=True)
class Ground_TruthSchema:
    # `format` selects the parser implementation: "touchstone" or "cadence_csv".
    format: str
    # `file_extension` lets the loader build the ground-truth filename from a sample id.
    # Empty string for cadence_csv where filenames are given explicitly.
    file_extension: str = ""
    # `ground_truth_parameters` chooses which physical quantities to extract from the file.
    ground_truth_parameters: tuple[str, ...] = ()
    # `ground_truth_parts` expands each quantity into concrete channels like real/imaginary parts.
    ground_truth_parts: tuple[str, ...] = ("re", "im")
    # Some datasets include a duplicated or invalid first frequency point that should be skipped.
    drop_first_frequency: bool = False
    # For cadence_csv: mapping of channel name → CSV filename, stored as sorted tuple of pairs.
    channel_files: tuple[tuple[str, str], ...] = ()
    # Optional per-channel units for MAE reporting (e.g. "dB", "deg", "V/sqrt(Hz)").
    # Stored as sorted tuple of (channel_name, unit_string) pairs.
    channel_units: tuple[tuple[str, str], ...] = ()
    # Optional per-channel transforms applied before normalization (e.g. "log10").
    # Stored as sorted tuple of (channel_name, transform_name) pairs.
    channel_transforms: tuple[tuple[str, str], ...] = ()

    @property
    def channel_names(self) -> list[str]:
        if self.format == "cadence_csv":
            # For cadence_csv each channel file maps to one output channel directly.
            return list(self.ground_truth_parameters)
        return [f"{name}_{part}" for name in self.ground_truth_parameters for part in self.ground_truth_parts]

    @property
    def channel_file_map(self) -> dict[str, str]:
        """Return {channel_name: filename} for convenient lookup."""
        return dict(self.channel_files)

    @property
    def channel_unit_map(self) -> dict[str, str]:
        """Return {channel_name: unit_string} for convenient lookup."""
        return dict(self.channel_units)

    @property
    def channel_transform_map(self) -> dict[str, str]:
        """Return {channel_name: transform_name} for convenient lookup."""
        return dict(self.channel_transforms)


# Full dataset metadata.
# This object groups the parsed input-feature/ground-truth schema and adds a few convenience helpers.
@dataclass(frozen=True)
class DatasetSchema:
    # Human-readable dataset label used in cache summaries and artifacts.
    dataset_name: str
    # Description of the tabular input-feature file.
    input_feature: InputFeatureSchema
    # Description of the per-sample ground-truth artifacts.
    ground_truth: Ground_TruthSchema
    # Remember where the schema came from so cache metadata can point back to it.
    readme_path: Path | None = None

    @property
    def schema_hash(self) -> str:
        payload = json.dumps(self.to_metadata(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_metadata(self) -> dict[str, Any]:
        input_meta: dict[str, Any] = {
            "columns": list(self.input_feature.columns),
            "feature_columns": list(self.input_feature.feature_columns),
            "source": self.input_feature.source,
        }
        if self.input_feature.source == "file":
            input_meta["sample_id_column"] = self.input_feature.sample_id_column
        if self.input_feature.parameter_keys:
            input_meta["parameter_keys"] = {
                csv_key: [col, scale] for csv_key, col, scale in self.input_feature.parameter_keys
            }

        gt_meta: dict[str, Any] = {"format": self.ground_truth.format}
        if self.ground_truth.format == "touchstone":
            gt_meta["file_extension"] = self.ground_truth.file_extension
            gt_meta["ground_truth_parameters"] = list(self.ground_truth.ground_truth_parameters)
            gt_meta["ground_truth_parts"] = list(self.ground_truth.ground_truth_parts)
            gt_meta["drop_first_frequency"] = self.ground_truth.drop_first_frequency
        elif self.ground_truth.format == "cadence_csv":
            gt_meta["channel_files"] = dict(self.ground_truth.channel_files)
            gt_meta["ground_truth_parameters"] = list(self.ground_truth.ground_truth_parameters)
        if self.ground_truth.channel_units:
            gt_meta["channel_units"] = dict(self.ground_truth.channel_units)
        if self.ground_truth.channel_transforms:
            gt_meta["channel_transforms"] = dict(self.ground_truth.channel_transforms)

        return {
            "dataset_name": self.dataset_name,
            "input_feature": input_meta,
            "ground_truth": gt_meta,
        }


# Public README parser.
def parse_dataset_readme(readme_path: str | Path) -> DatasetSchema:
    """Parse one dataset README that contains a JSON schema block."""
    readme_path = Path(readme_path)
    payload = _extract_schema_payload(readme_path.read_text(encoding="utf-8"))
    input_feature_payload = payload["input_feature"]
    ground_truth_payload = payload["ground_truth"]
    fmt = str(ground_truth_payload["format"]).lower()

    if fmt == "cadence_csv":
        return _build_cadence_csv_schema(payload, input_feature_payload, ground_truth_payload, readme_path)
    return _build_touchstone_schema(payload, input_feature_payload, ground_truth_payload, readme_path)


def _build_touchstone_schema(
    payload: dict[str, Any],
    input_feature_payload: dict[str, Any],
    ground_truth_payload: dict[str, Any],
    readme_path: Path,
) -> DatasetSchema:
    return DatasetSchema(
        dataset_name=str(payload.get("dataset_name", readme_path.parent.name)),
        input_feature=InputFeatureSchema(
            columns=tuple(input_feature_payload["columns"]),
            feature_columns=tuple(input_feature_payload["feature_columns"]),
            sample_id_column=str(input_feature_payload["sample_id_column"]),
            source="file",
        ),
        ground_truth=Ground_TruthSchema(
            format="touchstone",
            file_extension=str(ground_truth_payload["file_extension"]).lower(),
            ground_truth_parameters=tuple(ground_truth_payload["ground_truth_parameters"]),
            ground_truth_parts=tuple(ground_truth_payload.get("ground_truth_parts", ["re", "im"])),
            drop_first_frequency=bool(ground_truth_payload.get("drop_first_frequency", False)),
        ),
        readme_path=readme_path.resolve(),
    )


def _build_cadence_csv_schema(
    payload: dict[str, Any],
    input_feature_payload: dict[str, Any],
    ground_truth_payload: dict[str, Any],
    readme_path: Path,
) -> DatasetSchema:
    # Build parameter_keys from the JSON mapping.
    raw_keys = input_feature_payload.get("parameter_keys", {})
    param_keys: list[tuple[str, str, float]] = []
    for csv_key, spec in sorted(raw_keys.items()):
        if isinstance(spec, list):
            col, scale = str(spec[0]), float(spec[1]) if len(spec) > 1 else 1.0
        elif isinstance(spec, dict):
            col, scale = str(spec["column"]), float(spec.get("scale", 1.0))
        else:
            col, scale = str(spec), 1.0
        param_keys.append((csv_key, col, scale))

    # Channel files define both the output channels and their CSV filenames.
    raw_channels = ground_truth_payload["channel_files"]
    channel_files = tuple(sorted(raw_channels.items()))
    channel_names = tuple(name for name, _ in channel_files)

    # Optional per-channel units and transforms.
    raw_units = ground_truth_payload.get("channel_units", {})
    channel_units = tuple(sorted(raw_units.items()))
    raw_transforms = ground_truth_payload.get("channel_transforms", {})
    channel_transforms = tuple(sorted(raw_transforms.items()))

    return DatasetSchema(
        dataset_name=str(payload.get("dataset_name", readme_path.parent.name)),
        input_feature=InputFeatureSchema(
            columns=tuple(input_feature_payload["columns"]),
            feature_columns=tuple(input_feature_payload["feature_columns"]),
            source="embedded",
            parameter_keys=tuple(param_keys),
        ),
        ground_truth=Ground_TruthSchema(
            format="cadence_csv",
            ground_truth_parameters=channel_names,
            ground_truth_parts=("raw",),
            channel_files=channel_files,
            channel_units=channel_units,
            channel_transforms=channel_transforms,
            drop_first_frequency=bool(ground_truth_payload.get("drop_first_frequency", False)),
        ),
        readme_path=readme_path.resolve(),
    )


# Private parsing helpers.
def _extract_schema_payload(text: str) -> dict[str, Any]:
    match = README_JSON_BLOCK.search(text)
    if match is None:
        raise ValueError("Dataset README must contain one fenced ```json ... ``` schema block.")
    payload = json.loads(match.group(1))
    _validate_schema_payload(payload)
    return payload


def _validate_schema_payload(payload: dict[str, Any]) -> None:
    if "input_feature" not in payload or "ground_truth" not in payload:
        raise ValueError("Dataset schema must define both 'input_feature' and 'ground_truth' sections.")
    input_feature_schema = payload["input_feature"]
    ground_truth_schema = payload["ground_truth"]
    fmt = str(ground_truth_schema.get("format", "")).lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported ground-truth format {fmt!r}. Choose from {SUPPORTED_FORMATS}.")

    # Column validation is common to all formats.
    if "columns" not in input_feature_schema or "feature_columns" not in input_feature_schema:
        raise ValueError("Dataset input-feature schema must define 'columns' and 'feature_columns'.")
    columns = list(input_feature_schema["columns"])
    feature_columns = list(input_feature_schema["feature_columns"])
    if len(columns) != len(set(columns)):
        raise ValueError("Input-feature columns must be unique.")
    unknown_features = [name for name in feature_columns if name not in columns]
    if unknown_features:
        raise ValueError(f"Input features are missing from input-feature columns: {unknown_features}")

    if fmt == "cadence_csv":
        _validate_cadence_csv_payload(input_feature_schema, ground_truth_schema)
    else:
        _validate_touchstone_payload(input_feature_schema, ground_truth_schema, columns)


def _validate_touchstone_payload(
    input_feature_schema: dict[str, Any],
    ground_truth_schema: dict[str, Any],
    columns: list[str],
) -> None:
    required_input = {"sample_id_column"}
    required_gt = {"file_extension", "ground_truth_parameters"}
    missing_input = required_input.difference(input_feature_schema)
    missing_gt = required_gt.difference(ground_truth_schema)
    if missing_input:
        raise ValueError(f"Dataset input-feature schema is missing fields: {sorted(missing_input)}")
    if missing_gt:
        raise ValueError(f"Dataset ground-truth schema is missing fields: {sorted(missing_gt)}")
    sample_id_column = input_feature_schema["sample_id_column"]
    if sample_id_column not in columns:
        raise ValueError(f"Sample id column '{sample_id_column}' is not present in input-feature columns.")
    if not ground_truth_schema["ground_truth_parameters"]:
        raise ValueError("Ground-truth schema must define at least one ground-truth parameter.")
    ground_truth_parts = list(ground_truth_schema.get("ground_truth_parts", ["re", "im"]))
    unsupported_parts = [part for part in ground_truth_parts if part not in {"re", "im", "mag", "db", "angle_deg"}]
    if unsupported_parts:
        raise ValueError(f"Unsupported ground-truth parts: {unsupported_parts}")


def _validate_cadence_csv_payload(
    input_feature_schema: dict[str, Any],
    ground_truth_schema: dict[str, Any],
) -> None:
    if "channel_files" not in ground_truth_schema:
        raise ValueError("Cadence CSV ground-truth schema must define 'channel_files'.")
    channel_files = ground_truth_schema["channel_files"]
    if not isinstance(channel_files, dict) or not channel_files:
        raise ValueError("'channel_files' must be a non-empty mapping of channel name to CSV filename.")
    source = input_feature_schema.get("source", "embedded")
    if source == "embedded" and "parameter_keys" not in input_feature_schema:
        raise ValueError(
            "Cadence CSV input-feature schema with source='embedded' must define 'parameter_keys' "
            "to map CSV parameter names to column names and unit scales."
        )
