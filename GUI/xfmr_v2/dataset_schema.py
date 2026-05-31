"""Parse and validate dataset schema metadata stored in README files.

The overall flow is:

1. Read a dataset README.
2. Extract a fenced JSON block.
3. Validate that the block contains the fields the loaders rely on.
4. Convert the raw dictionary into small immutable dataclasses.

Supported loading strategies (``source``):
- ``inline``: all data in a single CSV; rows grouped by feature columns.
- ``per_sample``: one GT file per sample (CSV or Touchstone); features in
  a separate tabular file.
- ``array``: pre-packed numpy/pickle arrays.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# README schemas live inside a fenced JSON block.
README_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

# Common aliases for ground-truth part names -> canonical names used internally.
_GT_PART_ALIASES: dict[str, str] = {
    "real": "re",
    "imag": "im",
    "imaginary": "im",
}

SUPPORTED_GT_PARTS = {"re", "im", "mag", "db", "angle_deg"}
SUPPORTED_SOURCES = ("inline", "per_sample", "array")


def _normalize_gt_parts(parts: list[str]) -> list[str]:
    """Normalize ground-truth part names to their canonical forms."""
    return [_GT_PART_ALIASES.get(p, p) for p in parts]


# ---------------------------------------------------------------------------
# Input-feature metadata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InputFeatureSchema:
    columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    sample_id_column: str = ""
    # "file" = separate tabular file; "embedded" = extracted from GT files.
    source: str = "file"
    # For source="embedded" (cadence-style): CSV key -> (column, scale).
    parameter_keys: tuple[tuple[str, str, float], ...] = ()
    # Relative path from dataset root to the input-feature file.
    file_path: str = ""

    @property
    def expected_width(self) -> int:
        return len(self.columns)

    @property
    def feature_indices(self) -> tuple[tuple[str, int], ...]:
        return tuple((name, self.columns.index(name)) for name in self.feature_columns)

    @property
    def sample_id_index(self) -> int:
        return self.columns.index(self.sample_id_column)


# ---------------------------------------------------------------------------
# Ground-truth metadata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Ground_TruthSchema:
    # Loading strategy: "inline", "per_sample", or "array".
    source: str
    ground_truth_parameters: tuple[str, ...] = ()
    ground_truth_parts: tuple[str, ...] = ("re", "im")
    drop_first_frequency: bool = False
    # per_sample: file extension to construct GT filenames (.csv, .s2p, .s4p …).
    file_extension: str = ""
    # per_sample / array: path to GT directory or pickle file.
    data_dir: str = ""
    # Explicit name of the frequency column in CSV data.
    frequency_column: str = ""
    # Display label for the swept axis. Defaults to "Frequency (GHz)".
    # Set to e.g. "VDIFF (mV)" for non-frequency sweeps.
    sweep_label: str = ""
    # array: frequency range in Hz.
    frequency_start_hz: float = 0.0
    frequency_stop_hz: float = 0.0
    # inline/cadence-style: channel -> CSV filename.
    channel_files: tuple[tuple[str, str], ...] = ()
    # Optional per-channel units / transforms.
    channel_units: tuple[tuple[str, str], ...] = ()
    channel_transforms: tuple[tuple[str, str], ...] = ()

    @property
    def channel_names(self) -> list[str]:
        if self.channel_files:
            return list(self.ground_truth_parameters)
        return [f"{name}_{part}" for name in self.ground_truth_parameters for part in self.ground_truth_parts]


# ---------------------------------------------------------------------------
# Full dataset metadata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSchema:
    dataset_name: str
    input_feature: InputFeatureSchema
    ground_truth: Ground_TruthSchema
    readme_path: Path | None = None


# ---------------------------------------------------------------------------
# Public README parser
# ---------------------------------------------------------------------------
def parse_dataset_readme(readme_path: str | Path) -> DatasetSchema:
    """Parse one dataset README that contains a JSON schema block."""
    readme_path = Path(readme_path)
    payload = _extract_schema_payload(readme_path.read_text(encoding="utf-8"))
    input_feature_payload = payload["input_feature"]
    ground_truth_payload = payload["ground_truth"]
    src = str(ground_truth_payload["source"]).lower()

    if src == "inline":
        return _build_inline_schema(payload, input_feature_payload, ground_truth_payload, readme_path)
    if src == "per_sample":
        return _build_per_sample_schema(payload, input_feature_payload, ground_truth_payload, readme_path)
    if src == "array":
        return _build_array_schema(payload, input_feature_payload, ground_truth_payload, readme_path)
    raise ValueError(f"Unsupported source {src!r}. Choose from {SUPPORTED_SOURCES}.")


# ---------------------------------------------------------------------------
# Extraction and validation
# ---------------------------------------------------------------------------
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

    src = str(ground_truth_schema.get("source", "")).lower()
    if not src:
        # Cadence-style READMEs declare `format: cadence_csv` + channel_files instead of an
        # explicit source; treat those as the inline (one-CSV-per-channel) source.
        fmt = str(ground_truth_schema.get("format", "")).lower()
        if ground_truth_schema.get("channel_files") or fmt == "cadence_csv":
            ground_truth_schema["source"] = "inline"
            src = "inline"
    if src not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported ground-truth source {src!r}. Choose from {SUPPORTED_SOURCES}.")

    # Column validation.  For embedded-source inputs (cadence-style), columns
    # can be auto-derived from feature_columns when not explicitly provided.
    if "feature_columns" not in input_feature_schema:
        raise ValueError("Dataset input-feature schema must define 'feature_columns'.")
    if "columns" not in input_feature_schema:
        # Auto-populate columns from feature_columns for embedded sources.
        input_feature_schema["columns"] = list(input_feature_schema["feature_columns"])
    columns = list(input_feature_schema["columns"])
    feature_columns = list(input_feature_schema["feature_columns"])
    if len(columns) != len(set(columns)):
        raise ValueError("Input-feature columns must be unique.")
    unknown_features = [name for name in feature_columns if name not in columns]
    if unknown_features:
        raise ValueError(f"Input features are missing from input-feature columns: {unknown_features}")

    # Normalize GT parts.
    if "ground_truth_parts" in ground_truth_schema:
        ground_truth_schema["ground_truth_parts"] = _normalize_gt_parts(
            list(ground_truth_schema["ground_truth_parts"])
        )
        unsupported = [p for p in ground_truth_schema["ground_truth_parts"] if p not in SUPPORTED_GT_PARTS]
        if unsupported:
            raise ValueError(f"Unsupported ground-truth parts: {unsupported}")

    # Source-specific validation.
    if src == "inline":
        _validate_inline(input_feature_schema, ground_truth_schema)
    elif src == "per_sample":
        _validate_per_sample(input_feature_schema, ground_truth_schema, columns)
    elif src == "array":
        _validate_array(ground_truth_schema)


def _validate_inline(
    input_feature_schema: dict[str, Any],
    ground_truth_schema: dict[str, Any],
) -> None:
    if not ground_truth_schema.get("ground_truth_parameters"):
        if not ground_truth_schema.get("channel_files"):
            raise ValueError("Inline source must define 'ground_truth_parameters' or 'channel_files'.")

    # Auto-convert ground_truth.file_path to channel_files for single-file cadence.
    gt_file_path = ground_truth_schema.get("file_path", "")
    if gt_file_path and not ground_truth_schema.get("channel_files"):
        params = ground_truth_schema.get("ground_truth_parameters", [])
        if params and input_feature_schema.get("source") == "embedded":
            ground_truth_schema["channel_files"] = {p: gt_file_path for p in params}

    if not input_feature_schema.get("feature_columns"):
        raise ValueError("Inline source must define 'feature_columns' to identify/group samples.")


def _validate_per_sample(
    input_feature_schema: dict[str, Any],
    ground_truth_schema: dict[str, Any],
    columns: list[str],
) -> None:
    if not ground_truth_schema.get("ground_truth_parameters"):
        raise ValueError("per_sample source must define 'ground_truth_parameters'.")
    sample_id_col = input_feature_schema.get("sample_id_column", "")
    if sample_id_col and sample_id_col not in columns:
        raise ValueError(f"Sample id column '{sample_id_col}' is not in input-feature columns.")


def _validate_array(ground_truth_schema: dict[str, Any]) -> None:
    if not ground_truth_schema.get("ground_truth_parameters"):
        raise ValueError("array source must define 'ground_truth_parameters'.")


# ---------------------------------------------------------------------------
# Schema builders
# ---------------------------------------------------------------------------
def _build_inline_schema(
    payload: dict[str, Any],
    input_feature_payload: dict[str, Any],
    ground_truth_payload: dict[str, Any],
    readme_path: Path,
) -> DatasetSchema:
    # Cadence-style inline: channel_files present, features embedded.
    channel_files_raw = ground_truth_payload.get("channel_files", {})
    # Normalize list-of-objects format to dict: [{file_path: ..., ...}] -> {param: file_path}
    if isinstance(channel_files_raw, list) and channel_files_raw:
        gt_params = list(ground_truth_payload.get("ground_truth_parameters", []))
        normalized: dict[str, str] = {}
        for i, entry in enumerate(channel_files_raw):
            if isinstance(entry, dict) and "file_path" in entry:
                name = gt_params[i] if i < len(gt_params) else f"channel_{i}"
                normalized[name] = entry["file_path"]
            elif isinstance(entry, str):
                name = gt_params[i] if i < len(gt_params) else f"channel_{i}"
                normalized[name] = entry
        channel_files_raw = normalized
    if isinstance(channel_files_raw, dict) and channel_files_raw:
        channel_files = tuple(sorted(channel_files_raw.items()))
        channel_names = tuple(name for name, _ in channel_files)
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
        raw_units = ground_truth_payload.get("channel_units", {})
        raw_transforms = ground_truth_payload.get("channel_transforms", {})
        gt_parts = tuple(ground_truth_payload.get("ground_truth_parts", ["raw"]))
        return DatasetSchema(
            dataset_name=str(payload.get("dataset_name", readme_path.parent.name)),
            input_feature=InputFeatureSchema(
                columns=tuple(input_feature_payload["columns"]),
                feature_columns=tuple(input_feature_payload["feature_columns"]),
                source="embedded",
                parameter_keys=tuple(param_keys),
                file_path=str(input_feature_payload.get("file_path", "")),
            ),
            ground_truth=Ground_TruthSchema(
                source="inline",
                ground_truth_parameters=channel_names,
                ground_truth_parts=gt_parts,
                channel_files=channel_files,
                channel_units=tuple(sorted(raw_units.items())) if isinstance(raw_units, dict) else (),
                channel_transforms=tuple(sorted(raw_transforms.items())) if isinstance(raw_transforms, dict) else (),
                drop_first_frequency=bool(ground_truth_payload.get("drop_first_frequency", False)),
                data_dir=str(ground_truth_payload.get("data_dir", "")),
                frequency_column=str(ground_truth_payload.get("frequency_column", "")),
                sweep_label=str(ground_truth_payload.get("sweep_label", "")),
            ),
            readme_path=readme_path.resolve(),
        )

    # Standard inline: single flat CSV.
    return DatasetSchema(
        dataset_name=str(payload.get("dataset_name", readme_path.parent.name)),
        input_feature=InputFeatureSchema(
            columns=tuple(input_feature_payload["columns"]),
            feature_columns=tuple(input_feature_payload["feature_columns"]),
            sample_id_column=str(input_feature_payload.get("sample_id_column", "")),
            source="file",
            file_path=str(input_feature_payload.get("file_path", "")),
        ),
        ground_truth=Ground_TruthSchema(
            source="inline",
            ground_truth_parameters=tuple(ground_truth_payload["ground_truth_parameters"]),
            ground_truth_parts=tuple(ground_truth_payload.get("ground_truth_parts", ["re", "im"])),
            drop_first_frequency=bool(ground_truth_payload.get("drop_first_frequency", False)),
            frequency_column=str(ground_truth_payload.get("frequency_column", "")),
        ),
        readme_path=readme_path.resolve(),
    )


def _build_per_sample_schema(
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
            sample_id_column=str(input_feature_payload.get("sample_id_column", "")),
            source="file",
            file_path=str(input_feature_payload.get("file_path", "")),
        ),
        ground_truth=Ground_TruthSchema(
            source="per_sample",
            file_extension=str(ground_truth_payload.get("file_extension", ".csv")).lower(),
            ground_truth_parameters=tuple(ground_truth_payload["ground_truth_parameters"]),
            ground_truth_parts=tuple(ground_truth_payload.get("ground_truth_parts", ["re", "im"])),
            drop_first_frequency=bool(ground_truth_payload.get("drop_first_frequency", False)),
            data_dir=str(ground_truth_payload.get("data_dir", "")),
            frequency_column=str(ground_truth_payload.get("frequency_column", "")),
        ),
        readme_path=readme_path.resolve(),
    )


def _build_array_schema(
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
            sample_id_column=str(input_feature_payload.get("sample_id_column", "")),
            source="file",
            file_path=str(input_feature_payload.get("file_path", "")),
        ),
        ground_truth=Ground_TruthSchema(
            source="array",
            ground_truth_parameters=tuple(ground_truth_payload["ground_truth_parameters"]),
            ground_truth_parts=tuple(ground_truth_payload.get("ground_truth_parts", ["re", "im"])),
            drop_first_frequency=bool(ground_truth_payload.get("drop_first_frequency", False)),
            data_dir=str(ground_truth_payload.get("data_dir", "")),
            frequency_start_hz=float(ground_truth_payload.get("frequency_start_hz") or 0),
            frequency_stop_hz=float(ground_truth_payload.get("frequency_stop_hz") or 0),
        ),
        readme_path=readme_path.resolve(),
    )
