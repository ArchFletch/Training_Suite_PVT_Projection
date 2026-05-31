"""Build caches and training splits from README-described raw data.

This module handles the entire data journey:

1. Resolve where the input-feature table and output files live on disk.
2. Read the README schema that explains how those files should be interpreted.
3. Convert the raw files into a compact `.npz` cache for repeated training runs.
4. Load the cache back, create deterministic train/validation/test splits,
   normalize the features, and build PyTorch `DataLoader` objects.

The module is intentionally split into small helpers so each stage is easy to
trace in isolation.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .dataset_schema import DatasetSchema, Ground_TruthSchema, parse_dataset_readme

# Default paths for the current XFMR dataset layout on disk.
DATA_ROOT = Path(r"C:\Users\tc57\Box\Rice_AIDRFIC\XFMR\XFMR_1to1\XFMR_2503_1x1_SameXY")
CACHE_PATH = Path("artifacts/cache/xfmr_1to1_v2.npz")

# Touchstone channel names use the Sij pattern, for example S11 or S34.
TOUCHSTONE_NAME = re.compile(r"^[Ss](\d+)(\d+)$")

# SI-suffix multipliers used by Cadence CSV exports.
_SI_SUFFIX: dict[str, float] = {
    "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
    "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12,
}


# Resolved filesystem locations for one dataset.
@dataclass
class DataSources:
    """Resolved input-feature table and ground-truth-data directory paths."""

    # Folder that conceptually owns the dataset and usually contains the README schema.
    dataset_root: Path
    # Path to the row-oriented text file that stores one sample of input features per line.
    # None for embedded-parameter formats (cadence_csv).
    input_feature_path: Path | None
    # Directory containing one output file per sample id.
    ground_truth_data_dir: Path


# Normalized loaders plus the metadata reused by training and evaluation.
@dataclass
class SplitBundle:
    """Prepared train/validation/test loaders and shared normalization metadata."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    # Exact shuffled indices used for each split so experiments can reproduce them.
    split_indices: dict[str, np.ndarray]
    # Names of all input-feature columns saved in the cache.
    input_feature_names: list[str]
    # Only the non-constant features survive into the model.
    active_names: list[str]
    # Constant features are tracked so reports can explain why they were dropped.
    dropped_names: list[str]
    # Statistics computed from the training split and reused for all splits.
    input_feature_mean: np.ndarray
    input_feature_std: np.ndarray
    # Frequency axis in three different forms: raw Hz, GHz for display, and normalized
    # coordinates fed to the model.
    frequency_hz: np.ndarray
    frequency_ghz: np.ndarray
    frequency_norm: np.ndarray
    # Display label for the swept axis (e.g. "Frequency (GHz)", "VDIFF (mV)").
    sweep_label: str
    # Human-readable names for the predicted outputs.
    target_names: list[str]
    channel_names: list[str]
    cache_path: Path
    # Per-channel, per-frequency target normalization statistics computed from
    # the training split.  Shape: (num_channels, num_frequencies).
    target_mean: np.ndarray
    target_std: np.ndarray
    # Optional per-channel unit strings for MAE reporting (e.g. ["dB", "deg"]).
    channel_units: list[str]
    # Per-channel transforms applied before normalization (e.g. ["", "log10"]).
    channel_transforms: list[str]


# ---------------------------------------------------------------------------
# Loader-code execution (in-memory, no files saved to disk)
# ---------------------------------------------------------------------------
_LOADER_ALLOWED_MODULES = {
    "csv", "re", "json", "math", "struct", "io", "os.path",
    "numpy", "pathlib",
}


def run_loader_code(
    code: str,
    dataset_root: str | Path,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Execute loader code in memory and return standardized arrays.

    The code must define ``load_dataset(dataset_root, max_samples=None)``
    returning a dict with at least ``features``, ``targets``, ``sweep_axis``.

    Raises ValueError on any failure (compilation, execution, validation).
    """
    import importlib
    import builtins

    # Use standard builtins — the loader code is user-provided and trusted.
    safe_globals: dict[str, Any] = {"__builtins__": builtins}

    for mod_name in _LOADER_ALLOWED_MODULES:
        try:
            safe_globals[mod_name.split(".")[0]] = importlib.import_module(mod_name)
        except ImportError:
            pass
    safe_globals["np"] = safe_globals.get("numpy")
    safe_globals["Path"] = Path

    try:
        exec(compile(code, "<loader>", "exec"), safe_globals)
    except Exception as exc:
        raise ValueError(f"Loader code compilation failed: {exc}") from exc

    load_fn = safe_globals.get("load_dataset")
    if load_fn is None:
        raise ValueError("Loader code must define a 'load_dataset' function.")

    try:
        result = load_fn(str(dataset_root), max_samples=max_samples)
    except Exception as exc:
        raise ValueError(f"Loader execution failed: {exc}") from exc

    required = {"features", "targets", "sweep_axis"}
    missing = required - set(result.keys())
    if missing:
        raise ValueError(f"Loader result missing keys: {missing}")

    features = np.asarray(result["features"], dtype=np.float32)
    targets = np.asarray(result["targets"], dtype=np.float32)
    sweep = np.asarray(result["sweep_axis"], dtype=np.float32)

    if features.ndim != 2:
        raise ValueError(f"features must be 2D (samples, features), got shape {features.shape}")
    if targets.ndim == 2:
        targets = targets[:, np.newaxis, :]
    if targets.ndim != 3:
        raise ValueError(f"targets must be 2D or 3D, got shape {targets.shape}")

    result["features"] = features
    result["targets"] = targets
    result["sweep_axis"] = sweep

    # --- Validate the loaded data ---
    warnings = _validate_loader_output(features, targets, sweep, result)
    if warnings:
        result["validation_warnings"] = warnings

    return result


def _validate_loader_output(
    features: np.ndarray,
    targets: np.ndarray,
    sweep: np.ndarray,
    result: dict[str, Any],
) -> list[str]:
    """Check for common data loading issues. Returns a list of warning strings."""
    warnings: list[str] = []
    num_samples = features.shape[0]
    num_channels = targets.shape[1]
    num_sweep = targets.shape[2]

    # 1. Sample count sanity
    if num_samples < 2:
        warnings.append(
            f"Only {num_samples} sample(s) loaded. This is likely a parsing error — "
            f"check that sample boundaries are detected correctly."
        )

    # 2. NaN / Inf check
    nan_features = np.isnan(features).sum()
    nan_targets = np.isnan(targets).sum()
    inf_targets = np.isinf(targets).sum()
    if nan_features > 0:
        warnings.append(f"Features contain {nan_features} NaN values.")
    if nan_targets > 0:
        warnings.append(f"Targets contain {nan_targets} NaN values.")
    if inf_targets > 0:
        warnings.append(f"Targets contain {inf_targets} Inf values.")

    # 3. Extreme target values (likely parser bug or simulator error)
    target_absmax = float(np.abs(targets).max())
    target_p99 = float(np.percentile(np.abs(targets), 99))
    if target_absmax > 10 * target_p99 and target_p99 > 0:
        warnings.append(
            f"Target max |{target_absmax:.2f}| is >10x the 99th percentile |{target_p99:.2f}|. "
            f"This suggests extreme outliers or parser errors."
        )

    # 4. Constant targets (all same value for a channel)
    for ch in range(num_channels):
        ch_std = float(targets[:, ch, :].std())
        ch_name = result.get("channel_names", [f"ch{ch}"])[ch] if ch < len(result.get("channel_names", [])) else f"ch{ch}"
        if ch_std < 1e-10:
            warnings.append(f"Channel '{ch_name}' has zero variance — all values are identical.")

    # 5. Sweep axis issues
    if len(sweep) < 2:
        warnings.append(f"Sweep axis has only {len(sweep)} point(s).")
    elif not np.all(np.diff(sweep) > 0) and not np.all(np.diff(sweep) < 0):
        warnings.append("Sweep axis is not monotonically increasing or decreasing.")

    # 6. Suspiciously many sweep points per sample (likely merged samples)
    if num_sweep > 10000:
        warnings.append(
            f"Each sample has {num_sweep} sweep points — this is unusually high. "
            f"Check that sample boundaries are being detected correctly."
        )

    # 7. Feature variance check
    for i in range(features.shape[1]):
        col = features[:, i]
        if col.max() == col.min() and num_samples > 1:
            fname = result.get("feature_names", [f"x{i}"])[i] if i < len(result.get("feature_names", [])) else f"x{i}"
            warnings.append(f"Feature '{fname}' is constant (value={col[0]:.6g}) — will be dropped during training.")

    return warnings


# Public cache and split API.
def load_existing_cache(
    cache_path: str | Path,
    progress_callback=None,
) -> dict[str, Any]:
    """Load an existing .npz cache and return its summary metadata.

    This does NOT build a cache from raw data.  Use ``build_cache_from_loader``
    to create a new cache from a ``loader.py``.
    """
    def emit(event: str, message: str, **payload: Any) -> None:
        if progress_callback is not None:
            progress_callback({"phase": "scan", "event": event, "message": message, **payload})

    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(".json")

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {cache_path}\n"
            "Build the cache first before it can be loaded."
        )

    with np.load(cache_path, allow_pickle=False) as data:
        summary = {
            "status": "existing",
            "cache_path": str(cache_path.resolve()),
            "num_samples": int(data["features"].shape[0]),
            "num_features": int(data["features"].shape[1]),
            "num_channels": int(data["targets"].shape[1]),
            "num_frequencies": int(data["frequency_hz"].shape[0]),
        }
        # Read metadata from the .json sidecar if available.
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            summary["dataset_name"] = meta.get("dataset_name", cache_path.stem)
            summary["dataset_root"] = meta.get("dataset_root", "")
            summary["input_feature_path"] = meta.get("input_feature_path", "")
            summary["ground_truth_data_dir"] = meta.get("ground_truth_data_dir", "")
            summary["readme_path"] = meta.get("readme_path", "")
            summary["dataset_schema"] = meta.get("dataset_schema", {})
        else:
            summary["dataset_name"] = cache_path.stem
            summary["dataset_root"] = ""
            summary["input_feature_path"] = ""
            summary["ground_truth_data_dir"] = ""
            summary["readme_path"] = ""
            summary["dataset_schema"] = {}

    emit("cache_existing", f"Loaded existing cache: {summary['num_samples']} samples.",
         cache_path=summary["cache_path"],
         total_samples=summary["num_samples"],
         frequency_count=summary["num_frequencies"])
    return summary


def build_cache_from_loader(
    loader_code: str,
    dataset_root: str | Path,
    cache_path: str | Path,
    max_samples: int | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    """Execute loader code in memory and save the result as a cache.

    This is the primary entry point for the loader-based loading flow. The code
    runs once in memory, produces standardized arrays, and saves them to the
    cache. No files are written to the dataset directory.
    """
    def emit(event: str, message: str, **payload: Any) -> None:
        if progress_callback is not None:
            progress_callback({"phase": "scan", "event": event, "message": message, **payload})

    emit("loader_running", "Executing AI-generated loader code...")
    result = run_loader_code(loader_code, dataset_root, max_samples)

    features = result["features"]
    targets = result["targets"]
    frequency_hz = result["sweep_axis"]
    feature_names = result.get("feature_names", [f"x{i}" for i in range(features.shape[1])])
    channel_names_list = result.get("channel_names", [f"ch{i}" for i in range(targets.shape[1])])
    target_names = result.get("target_names", channel_names_list)
    channel_unit_list = result.get("channel_units", [""] * len(channel_names_list))
    channel_transform_list = result.get("channel_transforms", [""] * len(channel_names_list))
    sweep_label = result.get("sweep_label", "Frequency (GHz)")
    dataset_name = result.get("dataset_name", Path(dataset_root).name)

    # Check for data issues before caching.
    validation_warnings = result.get("validation_warnings", [])
    if validation_warnings:
        warning_text = "\n".join(f"  - {w}" for w in validation_warnings)
        emit("loader_warning",
             f"Data validation warnings:\n{warning_text}")
        # Block caching if there are critical issues (e.g. only 1 sample).
        if features.shape[0] < 2:
            raise ValueError(
                f"Loader produced only {features.shape[0]} sample(s). "
                f"This is almost certainly a parsing bug.\n"
                f"Validation warnings:\n{warning_text}"
            )

    emit("loader_done",
         f"Loaded {features.shape[0]} samples, {targets.shape[1]} channels, "
         f"{targets.shape[2]} sweep points.")

    summary = _save_cache(
        cache_path,
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        features=features,
        targets=targets,
        frequency_hz=frequency_hz,
        feature_names=feature_names,
        channel_names=channel_names_list,
        target_names=target_names,
        channel_units=channel_unit_list,
        channel_transforms=channel_transform_list,
        sweep_label=sweep_label,
    )
    emit("cache_saved", f"Cache saved: {features.shape[0]} samples.", cache_path=summary["cache_path"])
    return summary


def _save_cache(
    cache_path: str | Path,
    *,
    dataset_root: str | Path,
    dataset_name: str,
    features: np.ndarray,
    targets: np.ndarray,
    frequency_hz: np.ndarray,
    feature_names: list[str],
    channel_names: list[str],
    target_names: list[str],
    channel_units: list[str],
    channel_transforms: list[str],
    sweep_label: str,
    ground_truth_data_dir: str | Path | None = None,
    input_feature_path: str = "",
    readme_path: str = "",
    dataset_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write standardized arrays to a ``.npz`` cache plus a ``.json`` metadata sidecar.

    Shared by both cache builders (loader-code and schema-driven) so every cache has
    the exact same on-disk layout that ``load_existing_cache`` / ``load_split_bundle``
    expect.
    """
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(".json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache_path,
        features=features,
        targets=targets,
        frequency_hz=frequency_hz,
        input_feature_names=np.asarray(feature_names),
        channel_names=np.asarray(channel_names),
        target_names=np.asarray(target_names),
        channel_units=np.asarray(channel_units),
        channel_transforms=np.asarray(channel_transforms),
        sweep_label=np.asarray(sweep_label),
    )

    summary = {
        "status": "created",
        "cache_path": str(cache_path.resolve()),
        "dataset_root": str(Path(dataset_root).resolve()),
        "dataset_name": dataset_name,
        "input_feature_path": input_feature_path,
        "ground_truth_data_dir": (
            str(Path(ground_truth_data_dir).resolve()) if ground_truth_data_dir else str(Path(dataset_root).resolve())
        ),
        "readme_path": readme_path,
        "dataset_schema": dataset_schema
        or {
            "input_feature": {"columns": list(feature_names), "feature_columns": list(feature_names)},
            "ground_truth": {
                "source": "loader",
                "ground_truth_parameters": list(target_names),
                "ground_truth_parts": ["raw"],
            },
        },
        "num_samples": int(features.shape[0]),
        "num_features": int(features.shape[1]),
        "num_channels": int(targets.shape[1]),
        "num_frequencies": int(targets.shape[2]),
    }
    meta_path.write_text(json.dumps(summary, indent=2))
    return summary


def build_cache_from_dataset(
    dataset_root: str | Path,
    cache_path: str | Path,
    max_samples: int | None = None,
    progress_callback=None,
    should_stop=None,
) -> dict[str, Any]:
    """Build a cache directly from a README-described dataset folder.

    Reads the dataset README's JSON schema block and uses the suite's built-in
    parsers (no loader code required). Supported ground-truth sources:

    - ``per_sample``: one Touchstone (.sNp) or CSV file per sample.
    - ``inline`` with ``channel_files`` (Cadence CSV exports).

    Other declared sources raise ``NotImplementedError`` with guidance to use
    ``build_cache_from_loader`` instead.
    """
    def emit(event: str, message: str, **payload: Any) -> None:
        if progress_callback is not None:
            progress_callback({"phase": "scan", "event": event, "message": message, **payload})

    root = Path(dataset_root)
    readme = next((root / name for name in ("README.md", "README.txt") if (root / name).is_file()), None)
    if readme is None:
        raise FileNotFoundError(
            f"No README.md/README.txt schema found in {root}. "
            "Add a README with a fenced ```json``` schema block, or build the cache with loader code."
        )
    emit("schema_parsing", f"Reading dataset schema from {readme.name}.")
    schema = parse_dataset_readme(readme)
    sources = resolve_data_sources_from_schema(root, schema)
    src = schema.ground_truth.source
    emit("cache_build_started", f"Building cache for {schema.dataset_name} (source={src}).")

    if src == "per_sample":
        features, targets, frequency_hz = _build_per_sample_arrays(sources, schema, max_samples, emit)
        channel_names = list(schema.ground_truth.channel_names)
        channel_units = [""] * len(channel_names)
        channel_transforms = [""] * len(channel_names)
    elif src == "inline" and schema.ground_truth.channel_files:
        features, targets, frequency_hz, channel_names, channel_units, channel_transforms = _build_cadence_arrays(
            sources, schema, max_samples, emit
        )
    else:
        raise NotImplementedError(
            f"Building a cache from ground-truth source '{src}' is not supported in the folder/GUI flow yet. "
            "Supported: per_sample (Touchstone/CSV) and Cadence CSV (inline + channel_files). "
            "For other formats, build the cache with loader code via build_cache_from_loader()."
        )

    summary = _save_cache(
        cache_path,
        dataset_root=root,
        dataset_name=schema.dataset_name,
        features=features,
        targets=targets,
        frequency_hz=frequency_hz,
        feature_names=list(schema.input_feature.feature_columns),
        channel_names=channel_names,
        target_names=list(channel_names),
        channel_units=channel_units,
        channel_transforms=channel_transforms,
        sweep_label=schema.ground_truth.sweep_label or "Frequency (GHz)",
        ground_truth_data_dir=sources.ground_truth_data_dir,
        input_feature_path=str(sources.input_feature_path.resolve()) if sources.input_feature_path else "",
        readme_path=str(readme.resolve()),
        dataset_schema={
            "input_feature": {
                "columns": list(schema.input_feature.columns),
                "feature_columns": list(schema.input_feature.feature_columns),
                "source": schema.input_feature.source,
            },
            "ground_truth": {
                "source": src,
                "ground_truth_parameters": list(schema.ground_truth.ground_truth_parameters),
                "ground_truth_parts": list(schema.ground_truth.ground_truth_parts),
            },
        },
    )
    emit("cache_saved", f"Cache saved: {features.shape[0]} samples.", cache_path=summary["cache_path"])
    return summary


def _build_cadence_arrays(
    sources: DataSources,
    schema: DatasetSchema,
    max_samples: int | None,
    emit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Assemble features/targets from Cadence CSV exports (one CSV per channel).

    Each channel's CSV holds all samples as sequential parameter blocks; input
    features come from the block headers via ``parameter_keys``. Reuses the shared
    ``_parse_cadence_csv`` parser.
    """
    channel_files = dict(schema.ground_truth.channel_files)
    channel_order = list(schema.ground_truth.ground_truth_parameters)
    feature_columns = list(schema.input_feature.feature_columns)
    param_key_map = {csv_key: (col, float(scale)) for csv_key, col, scale in schema.input_feature.parameter_keys}
    if not param_key_map and feature_columns:
        param_key_map = {col: (col, 1.0) for col in feature_columns}
    data_dir = sources.ground_truth_data_dir

    first = channel_order[0]
    params_list, freq_ghz, first_values = _parse_cadence_csv(data_dir / channel_files[first], param_key_map, max_samples)
    frequency_hz = (freq_ghz * 1e9).astype(np.float32)
    num_samples, num_freq = len(params_list), len(frequency_hz)
    emit("input_rows_loaded", f"Loaded {num_samples} samples from {Path(channel_files[first]).name}.",
         total_samples=num_samples)

    features = np.zeros((num_samples, len(feature_columns)), dtype=np.float32)
    for idx, params in enumerate(params_list):
        features[idx] = [params[col] for col in feature_columns]

    targets = np.zeros((num_samples, len(channel_order), num_freq), dtype=np.float32)
    targets[:, 0, :] = first_values
    for ch_idx, channel in enumerate(channel_order):
        if ch_idx == 0:
            continue
        _, ch_freq_ghz, ch_values = _parse_cadence_csv(data_dir / channel_files[channel], param_key_map, max_samples)
        if not np.allclose(freq_ghz, ch_freq_ghz, rtol=1e-4):
            raise ValueError(f"Frequency grid in {channel} does not match {first}.")
        if len(ch_values) != num_samples:
            raise ValueError(f"Sample count mismatch: {first} has {num_samples} but {channel} has {len(ch_values)}.")
        targets[:, ch_idx, :] = ch_values
        emit("cache_progress", f"Loaded channel '{channel}'.", completed_samples=num_samples, total_samples=num_samples)

    if schema.ground_truth.drop_first_frequency and num_freq > 1:
        frequency_hz = frequency_hz[1:]
        targets = targets[:, :, 1:]

    units = dict(schema.ground_truth.channel_units)
    transforms = dict(schema.ground_truth.channel_transforms)
    channel_units = [units.get(name, "") for name in channel_order]
    channel_transforms = [transforms.get(name, "") for name in channel_order]
    return features, targets, frequency_hz, channel_order, channel_units, channel_transforms


def ensure_cache(
    data_root: str | Path | None = DATA_ROOT,
    cache_path: str | Path = CACHE_PATH,
    max_samples: int | None = None,
    input_feature_path: str | Path | None = None,
    ground_truth_data_dir: str | Path | None = None,
) -> Path:
    """Return a ready-to-use cache path.

    The cache must already exist (created via ``build_cache_from_loader``).
    """
    p = Path(cache_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Cache not found: {p}\n"
            "Build the cache first."
        )
    return p




def resolve_data_sources_from_schema(
    dataset_root: Path,
    schema: "DatasetSchema",
) -> DataSources:
    """Resolve paths using the schema's ``file_path`` and ``data_dir`` fields.

    When the schema specifies where the input-feature file and ground-truth
    directory are located (relative to the dataset root), this function builds
    ``DataSources`` directly — no heuristics needed.  Fields left empty in the
    schema fall back to the legacy defaults (``log.txt``, ``SPData/``).
    """
    # Input-feature file.
    input_feature_file: Path | None = None
    if schema.input_feature.file_path:
        input_feature_file = dataset_root / schema.input_feature.file_path
    elif schema.input_feature.source == "file":
        candidate = dataset_root / "log.txt"
        if candidate.exists():
            input_feature_file = candidate

    # Ground-truth data directory.
    if schema.ground_truth.data_dir:
        ground_truth_dir = dataset_root / schema.ground_truth.data_dir
    else:
        sp_candidate = dataset_root / "SPData"
        ground_truth_dir = sp_candidate if sp_candidate.is_dir() else dataset_root

    # Validate.
    if input_feature_file is not None and not input_feature_file.exists():
        raise FileNotFoundError(
            f"Input-feature file specified in schema not found: {input_feature_file}"
        )
    if not ground_truth_dir.exists():
        raise FileNotFoundError(
            f"Ground-truth directory specified in schema not found: {ground_truth_dir}"
        )

    return DataSources(
        dataset_root=dataset_root,
        input_feature_path=input_feature_file,
        ground_truth_data_dir=ground_truth_dir,
    )








# Cache loading and split preparation.
def load_split_bundle(
    cache_path: str | Path,
    batch_size: int,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    max_samples: int | None = None,
    pin_memory: bool = False,
) -> SplitBundle:
    """Load cached arrays, drop constant inputs, normalize, and build loaders."""
    with np.load(cache_path, allow_pickle=False) as data:
        # `build_cache` already enforced the schema, so this stage only reshapes and normalizes.
        features = data["features"].astype(np.float32)
        targets = data["targets"].astype(np.float32)
        frequency_hz = data["frequency_hz"].astype(np.float32)
        input_feature_names = data["input_feature_names"].astype(str).tolist()
        channel_names = data["channel_names"].astype(str).tolist()
        target_names = data["target_names"].astype(str).tolist()
        channel_units = data["channel_units"].astype(str).tolist() if "channel_units" in data else [""] * len(channel_names)
        channel_transforms = data["channel_transforms"].astype(str).tolist() if "channel_transforms" in data else [""] * len(channel_names)
        sweep_label = str(data["sweep_label"]) if "sweep_label" in data else "Frequency (GHz)"

    if max_samples is not None:
        features = features[:max_samples]
        targets = targets[:max_samples]

    # Compute the split first, then derive the active features and normalization
    # statistics from the training split only.
    split = split_indices(len(features), train_frac, val_frac, seed)
    train_x = features[split["train"]]
    # A feature is constant only if every value in the training set is identical.
    # Using absolute std threshold (e.g. 1e-8) would incorrectly drop features
    # with tiny but meaningful SI-unit values (e.g. capacitance in farads ~1e-13).
    active_mask = train_x.max(axis=0) != train_x.min(axis=0)
    active_names = [name for name, keep in zip(input_feature_names, active_mask, strict=True) if keep]
    dropped_names = [name for name, keep in zip(input_feature_names, active_mask, strict=True) if not keep]
    mean = train_x[:, active_mask].mean(axis=0).astype(np.float64)
    std = train_x[:, active_mask].std(axis=0).astype(np.float64)
    # Only clamp truly zero std (constant features are already filtered out).
    std[std == 0] = 1.0
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    # This mirrors how the model will be used in practice: validation and test data
    # are normalized with statistics computed from the training data only.
    # Apply the train-derived normalization to every split.
    x = (features[:, active_mask] - mean) / std

    # Normalize targets per-channel, per-frequency using training-split statistics.
    # This is critical for multi-channel outputs with different scales (e.g. gain
    # in dB and phase in degrees).
    train_y = targets[split["train"]]
    target_mean = train_y.mean(axis=0).astype(np.float64)
    target_std = train_y.std(axis=0).astype(np.float64)
    # Clamp near-zero std per frequency point to prevent extreme normalized values.
    # Use per-channel global std as the floor so no frequency point gets blown up.
    for ch in range(target_std.shape[0]):
        ch_global_std = float(train_y[:, ch, :].std())
        floor = max(ch_global_std * 0.01, 1e-30)  # 1% of global std
        target_std[ch][target_std[ch] < floor] = floor
    target_mean = target_mean.astype(np.float32)
    target_std = target_std.astype(np.float32)
    y = (targets - target_mean) / target_std

    train_loader = _make_loader(x, y, split["train"], batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = _make_loader(x, y, split["val"], batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = _make_loader(x, y, split["test"], batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    # For frequency-swept data, convert Hz→GHz. For other sweeps, use raw values.
    is_frequency = "freq" in sweep_label.lower() or sweep_label == "Frequency (GHz)"
    frequency_ghz = (frequency_hz / 1.0e9) if is_frequency else frequency_hz
    return SplitBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        split_indices=split,
        input_feature_names=input_feature_names,
        active_names=active_names,
        dropped_names=dropped_names,
        input_feature_mean=mean,
        input_feature_std=std,
        frequency_hz=frequency_hz,
        frequency_ghz=frequency_ghz,
        frequency_norm=normalize_frequency(frequency_ghz),
        target_names=target_names,
        channel_names=channel_names,
        cache_path=Path(cache_path),
        target_mean=target_mean,
        target_std=target_std,
        channel_units=channel_units,
        channel_transforms=channel_transforms,
        sweep_label=sweep_label,
    )




def _inverse_channel_transforms(values: np.ndarray, transforms: list[str]) -> np.ndarray:
    """Undo per-channel transforms for MAE reporting in original units."""
    out = values.copy()
    for ch_idx, transform in enumerate(transforms):
        if transform == "log10":
            out[:, ch_idx, :] = np.power(10.0, out[:, ch_idx, :])
    return out


def split_indices(num_samples: int, train_frac: float, val_frac: float, seed: int) -> dict[str, np.ndarray]:
    """Create one deterministic shuffled split."""
    if num_samples < 1:
        raise ValueError("Need at least one sample to create splits.")

    order = np.random.default_rng(seed).permutation(num_samples)
    if num_samples == 1:
        # A one-sample dataset cannot support validation/test splits, so keep the
        # single example in train and leave the others empty.
        empty = order[:0]
        return {"train": order, "val": empty, "test": empty}

    # Keep at least one training sample and, when possible, one held-out sample.
    train_end = max(int(num_samples * train_frac), 1)
    val_end = min(max(train_end + int(num_samples * val_frac), train_end + 1), num_samples - 1)
    return {"train": order[:train_end], "val": order[train_end:val_end], "test": order[val_end:]}


def normalize_frequency(frequency_ghz: np.ndarray) -> np.ndarray:
    """Map frequency to [-1, 1] for the model's frequency input."""
    lo, hi = float(frequency_ghz.min()), float(frequency_ghz.max())
    if np.isclose(lo, hi):
        return np.zeros_like(frequency_ghz, dtype=np.float32)
    return (2.0 * (frequency_ghz - lo) / (hi - lo) - 1.0).astype(np.float32)




def _make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    # Build one split loader without repeating the same TensorDataset/DataLoader
    # boilerplate three times in `load_split_bundle`.
    return DataLoader(
        TensorDataset(torch.from_numpy(features[indices]), torch.from_numpy(targets[indices])),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )


# Format-specific cache builders.
def _build_per_sample_arrays(
    sources: DataSources,
    schema: DatasetSchema,
    max_samples: int | None,
    emit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unified per-sample loader: CSV or Touchstone, auto-detected from extension."""
    assert sources.input_feature_path is not None
    input_feature_schema = schema.input_feature
    ground_truth_schema = schema.ground_truth
    ext = ground_truth_schema.file_extension.lower()
    is_touchstone = bool(re.fullmatch(r"\.s\d+p", ext))

    # Load input features — auto-detect log.txt (list-per-line) vs CSV.
    if sources.input_feature_path.suffix.lower() in (".csv", ".tsv"):
        import csv as csv_module
        with sources.input_feature_path.open(encoding="utf-8") as fh:
            reader = csv_module.DictReader(fh)
            csv_rows = list(reader)
        if max_samples is not None:
            csv_rows = csv_rows[:max_samples]
        total_samples = len(csv_rows)
        feature_col_names = list(input_feature_schema.feature_columns)
        sample_id_col = input_feature_schema.sample_id_column
        use_csv = True
    else:
        rows = _load_input_feature_rows(
            sources.input_feature_path, max_samples,
            expected_width=input_feature_schema.expected_width,
        )
        total_samples = len(rows)
        feature_columns_idx = tuple(idx for _, idx in input_feature_schema.feature_indices)
        sample_id_index = input_feature_schema.sample_id_index
        use_csv = False

    emit("input_rows_loaded",
         f"Loaded {total_samples} input rows from {sources.input_feature_path.name}.",
         total_samples=total_samples,
         input_feature_path=str(sources.input_feature_path.resolve()))

    # Build GT column names for per-sample CSV files.
    target_col_names: list[str] | None = None
    if not is_touchstone:
        target_col_names = _resolve_gt_column_names(ground_truth_schema)

    num_features = len(input_feature_schema.feature_columns)
    features = np.zeros((total_samples, num_features), dtype=np.float32)
    targets: np.ndarray | None = None
    frequency_hz: np.ndarray | None = None
    emit("cache_build_started", f"Building cache from {total_samples} samples.",
         total_samples=total_samples)
    progress_interval = max(1, total_samples // 25) if total_samples else 1

    skipped = 0
    valid_idx = 0
    for idx in range(total_samples):
        # Resolve sample ID and feature values.
        if use_csv:
            row_dict = csv_rows[idx]
            sample_id = row_dict.get(sample_id_col, str(idx)) if sample_id_col else str(idx)
            feat_values = [float(row_dict[col]) for col in feature_col_names]
        else:
            row_list = rows[idx]
            sample_id = str(int(row_list[sample_id_index]))
            feat_values = [float(row_list[c]) for c in feature_columns_idx]

        # Find the GT file.
        sample_path = _find_per_sample_file(
            sources.ground_truth_data_dir, sample_id, ext,
        )
        if sample_path is None:
            skipped += 1
            emit("cache_progress", f"Skipping sample '{sample_id}': ground-truth file not found.")
            continue

        # Load GT data.
        try:
            if is_touchstone:
                sample_freq, sample_target = _load_target_sample(sample_path, ground_truth_schema)
            else:
                assert target_col_names is not None
                sample_freq, sample_target = _load_per_sample_csv(
                    sample_path, target_col_names,
                    frequency_column=ground_truth_schema.frequency_column,
                )
        except Exception as exc:
            skipped += 1
            emit("cache_progress", f"Skipping sample '{sample_id}': {exc}")
            continue

        if targets is None:
            frequency_hz = sample_freq
            targets = np.zeros((total_samples, sample_target.shape[0], sample_target.shape[1]), dtype=np.float32)
        elif not np.allclose(frequency_hz, sample_freq):
            skipped += 1
            emit("cache_progress", f"Skipping sample '{sample_id}': frequency mismatch.")
            continue

        features[valid_idx] = feat_values
        targets[valid_idx] = sample_target
        valid_idx += 1
        if valid_idx == 1 or (idx + 1) == total_samples or (idx + 1) % progress_interval == 0:
            emit("cache_progress",
                 f"Loaded ground-truth sample {valid_idx}/{total_samples} (skipped {skipped}).",
                 completed_samples=valid_idx, total_samples=total_samples)

    if targets is None or frequency_hz is None or valid_idx == 0:
        raise ValueError("No samples were loaded.")
    if skipped:
        emit("cache_progress", f"Skipped {skipped} samples with missing ground-truth data.")
    features = features[:valid_idx]
    targets = targets[:valid_idx]

    if ground_truth_schema.drop_first_frequency and len(frequency_hz) > 1:
        frequency_hz = frequency_hz[1:]
        targets = targets[:, :, 1:]
    return features, targets, frequency_hz






def _part_to_csv_suffix(part: str) -> str:
    """Map schema part names to common CSV column suffixes."""
    return {"re": "real", "im": "imag"}.get(part, part)


def _resolve_gt_column_names(
    gt_schema: Ground_TruthSchema,
    csv_columns: set[str] | None = None,
) -> list[str]:
    """Build the ordered list of GT column names to read from CSV data.

    Auto-detects column names from csv_columns using common naming conventions.
    """
    csv_columns = csv_columns or set()
    result: list[str] = []
    for param in gt_schema.ground_truth_parameters:
        for part in gt_schema.ground_truth_parts:
            candidates = [
                f"{param}_{part}",
                f"{param.lower()}_{_part_to_csv_suffix(part)}",
                f"{param.lower()}_{part}",
                # Common convention: "S11" for dB/mag, "S11_phase" for angle.
                param if part in ("db", "mag") else f"{param}_phase" if part == "angle_deg" else f"{param}_{part}",
            ]
            matched = False
            for c in candidates:
                if c in csv_columns:
                    result.append(c)
                    matched = True
                    break
            if not matched:
                result.append(f"{param.lower()}_{_part_to_csv_suffix(part)}")
    return result


def _resolve_frequency_column(explicit: str, csv_columns: set[str]) -> str:
    """Return the frequency column name, using explicit if set, otherwise auto-detect."""
    if explicit:
        return explicit
    for candidate in ("freq", "frequency", "Freq", "Frequency", "FREQ", "Freq_GHz", "freq_ghz"):
        if candidate in csv_columns:
            return candidate
    raise ValueError(
        "Could not auto-detect a frequency column. "
        "Set 'frequency_column' in the ground_truth schema."
    )


def _find_per_sample_file(gt_dir: Path, sample_id: str, extension: str) -> Path | None:
    """Find a per-sample GT file by trying common naming patterns."""
    candidate = gt_dir / f"{sample_id}{extension}"
    if candidate.exists():
        return candidate
    for f in gt_dir.iterdir():
        if f.is_file() and sample_id in f.stem and f.suffix.lower() == extension:
            return f
    # Search subdirectories one level deep.
    for sub in gt_dir.iterdir():
        if not sub.is_dir():
            continue
        candidate = sub / f"{sample_id}{extension}"
        if candidate.exists():
            return candidate
        for f in sub.iterdir():
            if f.is_file() and sample_id in f.stem and f.suffix.lower() == extension:
                return f
    return None


def _load_per_sample_csv(
    path: Path,
    target_col_names: list[str],
    frequency_column: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Load frequency and target channels from a per-sample CSV.

    Returns (frequency_hz, target_channels) where target_channels has
    shape (num_channels, num_frequencies).
    """
    import csv as csv_module

    with path.open(encoding="utf-8") as fh:
        reader = csv_module.DictReader(fh)
        csv_columns = reader.fieldnames or []
        rows = list(reader)

    num_freqs = len(rows)
    freq_col = _resolve_frequency_column(frequency_column, set(csv_columns))

    frequency_hz = np.zeros(num_freqs, dtype=np.float64)
    if freq_col:
        for i, row in enumerate(rows):
            frequency_hz[i] = float(row[freq_col])

    # Map target column names to actual CSV columns (case-insensitive match).
    csv_col_lower_map = {c.lower(): c for c in csv_columns}
    resolved_cols: list[str] = []
    for tc in target_col_names:
        if tc in csv_columns:
            resolved_cols.append(tc)
        elif tc.lower() in csv_col_lower_map:
            resolved_cols.append(csv_col_lower_map[tc.lower()])
        else:
            # Try alternate suffix: real↔re, imag↔im.
            alt = tc.replace("_real", "_re").replace("_imag", "_im")
            if alt.lower() in csv_col_lower_map:
                resolved_cols.append(csv_col_lower_map[alt.lower()])
            else:
                alt2 = tc.replace("_re", "_real").replace("_im", "_imag")
                if alt2.lower() in csv_col_lower_map:
                    resolved_cols.append(csv_col_lower_map[alt2.lower()])
                else:
                    raise KeyError(
                        f"Target column '{tc}' not found in {path.name}. "
                        f"Available columns: {csv_columns}"
                    )

    target_channels = np.zeros((len(resolved_cols), num_freqs), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, col in enumerate(resolved_cols):
            target_channels[j, i] = float(row[col])

    return frequency_hz, target_channels


# Cadence CSV parsing helpers.
def _parse_si(s: str) -> float:
    """Parse a numeric string with an optional SI suffix (e.g. '191.6p' → 1.916e-10)."""
    s = s.strip()
    if not s:
        raise ValueError("Empty SI string")
    if s[-1] in _SI_SUFFIX:
        return float(s[:-1]) * _SI_SUFFIX[s[-1]]
    return float(s)


def _parse_cadence_csv(
    path: Path,
    param_key_map: dict[str, tuple[str, float]],
    max_samples: int | None = None,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    """Parse one Cadence Ocean CSV file and return (params_list, freq_ghz, values).

    Returns
    -------
    params_list : list of dict
        One dict per sample mapping column names to scaled float values.
    freq_ghz : ndarray, shape (num_frequencies,)
        Shared frequency grid in GHz.
    values : ndarray, shape (num_samples, num_frequencies)
        Output values for each sample at each frequency point.
    """
    with path.open(encoding="utf-8") as handle:
        content = handle.read()

    # Split on lines that begin with the first expected parameter key.
    # The Cadence CSV format uses "CS = ..." blocks separated by whitespace.
    first_key = next(iter(param_key_map))
    blocks = re.split(rf"\n(?={re.escape(first_key)}\s*=)", content)

    params_list: list[dict[str, float]] = []
    data_list: list[np.ndarray] = []
    freq_ref: np.ndarray | None = None

    for block in blocks:
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if not lines or f"{first_key}" not in lines[0]:
            continue

        raw_params: dict[str, str] = {}
        data_rows: list[tuple[float, float]] = []

        for line in lines:
            # Skip Cadence column-header lines which tend to be very long.
            if len(line) > 60 and "=" not in line:
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                if key in param_key_map:
                    raw_params[key] = val.strip()
            else:
                parts = line.split()
                # Check for space-separated parameter lines (e.g. "VCM  599m").
                if len(parts) >= 2 and parts[0] in param_key_map:
                    raw_params[parts[0]] = parts[1]
                elif len(parts) == 2:
                    try:
                        data_rows.append((_parse_si(parts[0]), _parse_si(parts[1])))
                    except ValueError:
                        pass

        if len(raw_params) < len(param_key_map) or not data_rows:
            continue

        # Convert raw parameter strings to scaled floats.
        sample_params: dict[str, float] = {}
        for csv_key, (col_name, scale) in param_key_map.items():
            raw_val = raw_params[csv_key]
            # Some Cadence exports append extra tokens after the value (e.g. "599m").
            # Take only the first whitespace-delimited token.
            raw_val = raw_val.split()[0] if raw_val else raw_val
            sample_params[col_name] = _parse_si(raw_val) * scale

        freqs_hz = np.array([r[0] for r in data_rows])
        values = np.array([r[1] for r in data_rows], dtype=np.float32)

        if freq_ref is None:
            freq_ref = freqs_hz
        params_list.append(sample_params)
        data_list.append(values)

        if max_samples is not None and len(params_list) >= max_samples:
            break

    if freq_ref is None or not data_list:
        raise ValueError(f"No valid sample blocks found in {path}")

    return params_list, (freq_ref * 1e-9).astype(np.float64), np.array(data_list, dtype=np.float32)


def _load_input_feature_rows(
    input_feature_path: Path,
    max_samples: int | None,
    expected_width: int,
) -> list[list[Any]]:
    # Each line is stored as a Python literal sequence, so `literal_eval` is a
    # compact and safe way to recover the row structure.
    rows: list[list[Any]] = []
    with input_feature_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = ast.literal_eval(line)
            if len(row) != expected_width:
                # The schema and the raw file must agree exactly. Failing early here
                # is much easier to debug than silently misaligned columns later.
                raise ValueError(f"Expected {expected_width} columns, got {len(row)}")
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows




# Ground-truth parsing entry point.
def _load_target_sample(path: Path, schema: Ground_TruthSchema) -> tuple[np.ndarray, np.ndarray]:
    # This dispatch point is where support for additional output file formats would
    # be added in the future.
    ext = path.suffix.lower()
    if re.fullmatch(r"\.s\d+p", ext):
        return _parse_touchstone(path, schema)
    raise NotImplementedError(f"Unsupported target file type: {ext}")


# Touchstone parsing helpers.
def _parse_touchstone(path: Path, schema: Ground_TruthSchema) -> tuple[np.ndarray, np.ndarray]:
    """Parse one Touchstone file and return configured ground-truth channels."""
    # These variables are populated from the optional metadata headers inside the file.
    numeric_tokens: list[float] = []
    number_ports: int | None = None
    matrix_format = "full"
    data_format = "ri"
    frequency_unit = "hz"

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()

            # Skip blank lines and comment lines before looking for structured metadata.
            if not line or line.startswith("!"):
                continue
            lower = line.lower()

            if line.startswith("#"):
                # The option line tells us which frequency unit the file uses and how
                # each complex number is encoded (RI, MA, or DB).
                parts = line[1:].split()
                if len(parts) >= 3:
                    frequency_unit = parts[0].lower()
                    data_format = parts[2].lower()
                continue

            if lower.startswith("[number of ports]"):
                number_ports = int(line.split("]", 1)[1].strip())
                continue

            if lower.startswith("[matrix format]"):
                matrix_format = line.split("]", 1)[1].strip().lower()
                continue

            if lower == "[end]":
                break

            if line.startswith("["):
                # Other section markers do not carry numeric payload that we need here.
                continue

            numeric_tokens.extend(float(value) for value in line.split())

    # Convert the flat numeric payload into one complex matrix per frequency point.
    ports = number_ports or _infer_touchstone_ports(path, schema.file_extension)
    pair_count = _touchstone_pair_count(ports, matrix_format)
    values_per_frequency = 1 + 2 * pair_count
    if not numeric_tokens or len(numeric_tokens) % values_per_frequency != 0:
        raise ValueError(f"Unexpected Touchstone token count in {path}")

    # After this reshape, each row corresponds to one frequency point and the remaining
    # values describe the complex response matrix at that frequency.
    array = np.asarray(numeric_tokens, dtype=np.float64).reshape(-1, values_per_frequency)
    frequency_hz = _convert_frequency_to_hz(array[:, 0], frequency_unit)
    complex_pairs = _touchstone_pairs_to_complex(
        array[:, 1:].reshape(array.shape[0], pair_count, 2),
        data_format,
    )
    matrix = _build_touchstone_matrix(complex_pairs, ports, matrix_format)
    # NOTE: drop_first_frequency is handled by the caller (_build_per_sample_arrays),
    # not here, to avoid double-dropping.

    # Finally, flatten the requested parameters and parts into the channel order
    # expected by the model and the cache file.
    channels: list[np.ndarray] = []
    for parameter in schema.ground_truth_parameters:
        values = _touchstone_parameter(matrix, parameter)
        for part in schema.ground_truth_parts:
            channels.append(_complex_to_part(values, part))
    return frequency_hz.astype(np.float32), np.stack(channels, axis=0).astype(np.float32)


def _infer_touchstone_ports(path: Path, default_extension: str) -> int:
    # Most Touchstone files encode the port count in the extension, such as `.s4p`.
    for candidate in (path.suffix.lower(), default_extension.lower()):
        match = re.fullmatch(r"\.s(\d+)p", candidate)
        if match is not None:
            return int(match.group(1))
    raise ValueError(f"Could not infer the Touchstone port count from {path.name}")


def _touchstone_pair_count(number_ports: int, matrix_format: str) -> int:
    # Lower/upper triangular storage keeps only the independent matrix entries.
    if matrix_format == "lower":
        return number_ports * (number_ports + 1) // 2
    if matrix_format == "upper":
        return number_ports * (number_ports + 1) // 2
    if matrix_format == "full":
        return number_ports * number_ports
    raise ValueError(f"Unsupported Touchstone matrix format: {matrix_format}")


def _convert_frequency_to_hz(values: np.ndarray, unit: str) -> np.ndarray:
    # Normalize all supported Touchstone frequency units to Hz so the rest of the
    # pipeline can work in a single unit system.
    scale = {
        "hz": 1.0,
        "khz": 1.0e3,
        "mhz": 1.0e6,
        "ghz": 1.0e9,
    }.get(unit.lower())
    if scale is None:
        raise ValueError(f"Unsupported Touchstone frequency unit: {unit}")
    return values * scale


def _touchstone_pairs_to_complex(pairs: np.ndarray, data_format: str) -> np.ndarray:
    # Touchstone stores each complex value either as RI, MA, or DB plus angle.
    # Convert everything into one shared complex representation first.
    data_format = data_format.lower()
    if data_format == "ri":
        return pairs[:, :, 0] + 1j * pairs[:, :, 1]
    if data_format == "ma":
        return pairs[:, :, 0] * np.exp(1j * np.deg2rad(pairs[:, :, 1]))
    if data_format == "db":
        return np.power(10.0, pairs[:, :, 0] / 20.0) * np.exp(1j * np.deg2rad(pairs[:, :, 1]))
    raise ValueError(f"Unsupported Touchstone data format: {data_format}")


def _build_touchstone_matrix(values: np.ndarray, number_ports: int, matrix_format: str) -> np.ndarray:
    # Reconstruct a full square response matrix for each frequency point.
    if matrix_format == "full":
        return values.reshape(values.shape[0], number_ports, number_ports).astype(np.complex64)
    if matrix_format == "lower":
        rows, cols = np.tril_indices(number_ports)
    elif matrix_format == "upper":
        rows, cols = np.triu_indices(number_ports)
    else:
        raise ValueError(f"Unsupported Touchstone matrix format: {matrix_format}")

    # Mirror the stored triangular entries across the diagonal to recover the full matrix.
    matrix = np.zeros((values.shape[0], number_ports, number_ports), dtype=np.complex64)
    matrix[:, rows, cols] = values
    matrix[:, cols, rows] = values
    return matrix


def _touchstone_parameter(matrix: np.ndarray, parameter_name: str) -> np.ndarray:
    # Translate names like S34 into zero-based matrix indices.
    match = TOUCHSTONE_NAME.fullmatch(parameter_name)
    if match is None:
        raise ValueError(f"Unsupported ground-truth parameter name: {parameter_name}")
    row = int(match.group(1)) - 1
    col = int(match.group(2)) - 1
    if row < 0 or col < 0 or row >= matrix.shape[1] or col >= matrix.shape[2]:
        raise ValueError(f"Ground-truth parameter {parameter_name} is out of range for a {matrix.shape[1]}-port file.")
    return matrix[:, row, col]


def _complex_to_part(values: np.ndarray, part: str) -> np.ndarray:
    # The model may train on several different views of the same complex ground truth.
    if part == "re":
        return values.real.astype(np.float32)
    if part == "im":
        return values.imag.astype(np.float32)
    if part == "mag":
        return np.abs(values).astype(np.float32)
    if part == "db":
        return (20.0 * np.log10(np.maximum(np.abs(values), 1e-12))).astype(np.float32)
    if part == "angle_deg":
        return np.rad2deg(np.angle(values)).astype(np.float32)
    raise ValueError(f"Unsupported ground-truth part: {part}")


