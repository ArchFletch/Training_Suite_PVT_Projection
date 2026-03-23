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

from .dataset_schema import DatasetSchema, Ground_TruthSchema, InputFeatureSchema, parse_dataset_readme

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
    # coordinates for the frequency trunk network.
    frequency_hz: np.ndarray
    frequency_ghz: np.ndarray
    frequency_norm: np.ndarray
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


# Public cache and split API.
def build_cache(
    data_root: str | Path | None = DATA_ROOT,
    cache_path: str | Path = CACHE_PATH,
    overwrite: bool = False,
    max_samples: int | None = None,
    input_feature_path: str | Path | None = None,
    ground_truth_data_dir: str | Path | None = None,
    progress_callback=None,
    should_stop=None,
) -> dict[str, Any]:
    """Read one dataset once and save a compact local cache."""

    def emit(event: str, message: str, **payload: Any) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "phase": "scan",
                "event": event,
                "message": message,
                **payload,
            }
        )

    # Resolve all filesystem inputs up front so the rest of the function can work
    # with concrete `Path` objects instead of juggling optional arguments.
    sources = resolve_data_sources(
        data_root=data_root,
        input_feature_path=input_feature_path,
        ground_truth_data_dir=ground_truth_data_dir,
    )

    # The README schema is the single source of truth for interpreting the raw files.
    schema = _load_dataset_schema_from_sources(sources)
    if schema is None:
        raise FileNotFoundError(
            "Dataset README schema not found. Add README.md or README.txt with one fenced ```json``` schema block "
            f"under {sources.dataset_root} so the loader knows the input-feature columns and ground-truth channels."
        )
    emit(
        "schema_ready",
        f"Loaded dataset schema for {schema.dataset_name}.",
        dataset_name=schema.dataset_name,
        readme_path=str(schema.readme_path) if schema.readme_path is not None else None,
    )

    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(".json")

    # Reuse an existing cache when both the data paths and the schema match.
    if cache_path.exists() and not overwrite:
        if meta_path.exists():
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not _cache_matches_sources(existing_meta, sources, schema):
                raise ValueError(
                    "Cache path already exists for different data sources or schema metadata. "
                    "Choose another cache path or pass overwrite=True."
                )
        with np.load(cache_path, allow_pickle=False) as data:
            summary = {
                "status": "existing",
                "cache_path": str(cache_path.resolve()),
                **_cache_source_metadata(sources, schema),
                "num_samples": int(data["features"].shape[0]),
                "num_frequencies": int(data["frequency_hz"].shape[0]),
            }
        emit(
            "cache_existing",
            f"Using existing cache {cache_path.name}.",
            cache_path=summary["cache_path"],
            total_samples=summary["num_samples"],
            frequency_count=summary["num_frequencies"],
        )
        return summary

    # Branch based on ground-truth format.
    if schema.ground_truth.format == "cadence_csv":
        features, targets, frequency_hz = _build_cadence_csv_arrays(
            sources, schema, max_samples, emit,
        )
    else:
        features, targets, frequency_hz = _build_touchstone_arrays(
            sources, schema, max_samples, emit,
        )

    total_samples = int(features.shape[0])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    emit(
        "cache_write_started",
        f"Writing cache to {cache_path.name}.",
        cache_path=str(cache_path.resolve()),
        total_samples=total_samples,
    )
    # Build per-channel unit and transform lists.
    unit_map = schema.ground_truth.channel_unit_map
    transform_map = schema.ground_truth.channel_transform_map
    channel_name_list = schema.ground_truth.channel_names
    channel_unit_list = [unit_map.get(ch, "") for ch in channel_name_list]
    channel_transform_list = [transform_map.get(ch, "") for ch in channel_name_list]

    # Apply per-channel transforms before saving to cache.
    targets = _apply_channel_transforms(targets, channel_transform_list)

    np.savez_compressed(
        cache_path,
        features=features,
        targets=targets,
        frequency_hz=frequency_hz,
        input_feature_names=np.asarray(schema.input_feature.feature_columns),
        channel_names=np.asarray(channel_name_list),
        target_names=np.asarray(list(schema.ground_truth.ground_truth_parameters)),
        channel_units=np.asarray(channel_unit_list),
        channel_transforms=np.asarray(channel_transform_list),
    )

    summary = {
        "status": "created",
        "cache_path": str(cache_path.resolve()),
        **_cache_source_metadata(sources, schema),
        "num_samples": total_samples,
        "num_features": int(features.shape[1]),
        "num_channels": int(targets.shape[1]),
        "num_frequencies": int(targets.shape[2]),
    }
    meta_path.write_text(json.dumps(summary, indent=2))
    emit(
        "cache_saved",
        f"Saved cache {cache_path.name}.",
        cache_path=summary["cache_path"],
        total_samples=summary["num_samples"],
        frequency_count=summary["num_frequencies"],
    )
    return summary


def ensure_cache(
    data_root: str | Path | None = DATA_ROOT,
    cache_path: str | Path = CACHE_PATH,
    max_samples: int | None = None,
    input_feature_path: str | Path | None = None,
    ground_truth_data_dir: str | Path | None = None,
) -> Path:
    """Return a ready-to-use cache path."""
    # This helper exists so callers that only need a cache path do not have to care
    # whether the cache already existed or had to be created just now.
    build_cache(
        data_root=data_root,
        cache_path=cache_path,
        max_samples=max_samples,
        input_feature_path=input_feature_path,
        ground_truth_data_dir=ground_truth_data_dir,
    )
    return Path(cache_path)


# Source and schema resolution helpers.
def resolve_data_sources(
    data_root: str | Path | None = DATA_ROOT,
    input_feature_path: str | Path | None = None,
    ground_truth_data_dir: str | Path | None = None,
) -> DataSources:
    """Resolve the effective input-feature table and ground-truth-data directory."""
    input_feature_file = Path(input_feature_path) if input_feature_path not in (None, "") else None
    ground_truth_dir = Path(ground_truth_data_dir) if ground_truth_data_dir not in (None, "") else None

    # The loader supports two ways of specifying the dataset:
    # - one `data_root`, which implies the conventional file layout
    # - explicit input/output paths
    explicit_input_feature_and_ground_truth = input_feature_file is not None and ground_truth_dir is not None
    root = (
        None if explicit_input_feature_and_ground_truth else Path(data_root) if data_root not in (None, "") else None
    )

    # These default names still reflect the current dataset layout on disk.
    if input_feature_file is None and root is not None:
        candidate = root / "log.txt"
        # For cadence_csv datasets there is no log.txt; input_feature_path stays None
        # and the build_cache path will extract parameters from the CSV files instead.
        if candidate.exists():
            input_feature_file = candidate
    elif input_feature_file is not None and input_feature_file.is_dir():
        input_feature_file = input_feature_file / "log.txt"

    if ground_truth_dir is None and root is not None:
        sp_candidate = root / "SPData"
        # For cadence_csv datasets the CSV files live directly in the data root.
        ground_truth_dir = sp_candidate if sp_candidate.is_dir() else root

    if ground_truth_dir is None:
        raise ValueError("Provide data_root or ground_truth_data_dir.")
    # Validate input_feature_file only when it was explicitly requested or when
    # the default log.txt actually existed.
    if input_feature_file is not None:
        if not input_feature_file.exists():
            raise FileNotFoundError(f"Input-feature description file not found: {input_feature_file}")
        if not input_feature_file.is_file():
            raise ValueError(f"Input-feature description path must be a file: {input_feature_file}")
    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"Ground-truth data directory not found: {ground_truth_dir}")
    if not ground_truth_dir.is_dir():
        raise ValueError(f"Ground-truth data path must be a directory: {ground_truth_dir}")

    # When explicit paths are used, recover the shared dataset root from the common
    # parent so schema discovery and cache metadata still have a stable anchor point.
    if root is not None:
        dataset_root = root
    elif input_feature_file is not None:
        dataset_root = Path(
            os.path.commonpath([str(input_feature_file.parent), str(ground_truth_dir.parent)])
        )
    else:
        dataset_root = ground_truth_dir
    return DataSources(
        dataset_root=dataset_root,
        input_feature_path=input_feature_file,
        ground_truth_data_dir=ground_truth_dir,
    )


def _load_dataset_schema_from_sources(sources: DataSources) -> DatasetSchema | None:
    # The README can live at the dataset root or beside the input-feature/ground-truth data.
    readme_path = _find_dataset_readme_path(sources)
    if readme_path is None:
        return None
    return parse_dataset_readme(readme_path)


def _find_dataset_readme_path(sources: DataSources) -> Path | None:
    # Prefer a README at the dataset root, but also allow one beside the data folders
    # when the root was inferred from explicit paths.
    readme_dirs = [sources.dataset_root]
    if sources.input_feature_path is not None:
        common_parent = sources.input_feature_path.parent
        if common_parent == sources.ground_truth_data_dir.parent and common_parent != sources.dataset_root:
            readme_dirs.append(common_parent)

    for directory in readme_dirs:
        for filename in ("README.md", "README.txt"):
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    return None


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

    if max_samples is not None:
        features = features[:max_samples]
        targets = targets[:max_samples]

    # Compute the split first, then derive the active features and normalization
    # statistics from the training split only.
    split = split_indices(len(features), train_frac, val_frac, seed)
    train_x = features[split["train"]]
    active_mask = train_x.std(axis=0) > 1e-8
    active_names = [name for name, keep in zip(input_feature_names, active_mask, strict=True) if keep]
    dropped_names = [name for name, keep in zip(input_feature_names, active_mask, strict=True) if not keep]
    mean = train_x[:, active_mask].mean(axis=0).astype(np.float32)
    std = train_x[:, active_mask].std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0

    # This mirrors how the model will be used in practice: validation and test data
    # are normalized with statistics computed from the training data only.
    # Apply the train-derived normalization to every split.
    x = (features[:, active_mask] - mean) / std

    # Normalize targets per-channel, per-frequency using training-split statistics.
    # This is critical for multi-channel outputs with different scales (e.g. gain
    # in dB and phase in degrees).
    train_y = targets[split["train"]]
    target_mean = train_y.mean(axis=0).astype(np.float32)   # (channels, freq)
    target_std = train_y.std(axis=0).astype(np.float32)     # (channels, freq)
    target_std[target_std < 1e-8] = 1.0
    y = (targets - target_mean) / target_std

    train_loader = _make_loader(x, y, split["train"], batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = _make_loader(x, y, split["val"], batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = _make_loader(x, y, split["test"], batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    frequency_ghz = frequency_hz / 1.0e9
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
    )


def _apply_channel_transforms(targets: np.ndarray, transforms: list[str]) -> np.ndarray:
    """Apply per-channel transforms (e.g. log10) to the target array in-place."""
    for ch_idx, transform in enumerate(transforms):
        if transform == "log10":
            # Clamp to a small positive floor to avoid log(0).
            targets[:, ch_idx, :] = np.log10(np.maximum(targets[:, ch_idx, :], 1e-30))
        elif transform and transform != "":
            raise ValueError(f"Unsupported channel transform: {transform!r}")
    return targets


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
    """Map frequency to [-1, 1] for the trunk network."""
    lo, hi = float(frequency_ghz.min()), float(frequency_ghz.max())
    if np.isclose(lo, hi):
        return np.zeros_like(frequency_ghz, dtype=np.float32)
    return (2.0 * (frequency_ghz - lo) / (hi - lo) - 1.0).astype(np.float32)


# Cache metadata and raw input-feature parsing.
def _cache_source_metadata(sources: DataSources, schema: DatasetSchema) -> dict[str, Any]:
    return {
        "dataset_root": str(sources.dataset_root.resolve()),
        "input_feature_path": str(sources.input_feature_path.resolve()) if sources.input_feature_path else None,
        "ground_truth_data_dir": str(sources.ground_truth_data_dir.resolve()),
        "readme_path": str(schema.readme_path) if schema.readme_path is not None else None,
        "dataset_name": schema.dataset_name,
        "schema_hash": schema.schema_hash,
        "dataset_schema": schema.to_metadata(),
    }


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
def _build_touchstone_arrays(
    sources: DataSources,
    schema: DatasetSchema,
    max_samples: int | None,
    emit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load features from log.txt and targets from per-sample Touchstone files."""
    assert sources.input_feature_path is not None
    rows = _load_input_feature_rows(
        sources.input_feature_path,
        max_samples,
        expected_width=schema.input_feature.expected_width,
    )
    emit(
        "input_rows_loaded",
        f"Loaded {len(rows)} input rows from {sources.input_feature_path.name}.",
        total_samples=len(rows),
        input_feature_path=str(sources.input_feature_path.resolve()),
    )
    input_feature_schema = schema.input_feature
    ground_truth_schema = schema.ground_truth
    feature_columns = tuple(index for _, index in input_feature_schema.feature_indices)
    sample_id_index = input_feature_schema.sample_id_index

    features = np.zeros((len(rows), len(feature_columns)), dtype=np.float32)
    targets: np.ndarray | None = None
    frequency_hz: np.ndarray | None = None
    total_samples = len(rows)
    emit(
        "cache_build_started",
        f"Building cache from {total_samples} samples.",
        total_samples=total_samples,
    )
    progress_interval = max(1, total_samples // 25) if total_samples else 1

    for idx, row in enumerate(rows):
        file_id = int(row[sample_id_index])
        features[idx] = [float(row[col]) for col in feature_columns]
        sample_path = sources.ground_truth_data_dir / f"{file_id}{ground_truth_schema.file_extension}"
        sample_freq, sample_target = _load_target_sample(sample_path, ground_truth_schema)
        if targets is None:
            frequency_hz = sample_freq
            targets = np.zeros((len(rows), sample_target.shape[0], sample_target.shape[1]), dtype=np.float32)
        elif not np.allclose(frequency_hz, sample_freq):
            raise ValueError(f"Frequency mismatch in {sample_path.name}")
        targets[idx] = sample_target
        completed_samples = idx + 1
        if completed_samples == 1 or completed_samples == total_samples or completed_samples % progress_interval == 0:
            emit(
                "cache_progress",
                f"Loaded ground-truth sample {completed_samples}/{total_samples}.",
                completed_samples=completed_samples,
                total_samples=total_samples,
            )

    if targets is None or frequency_hz is None:
        raise ValueError("No samples were loaded.")
    return features, targets, frequency_hz


def _build_cadence_csv_arrays(
    sources: DataSources,
    schema: DatasetSchema,
    max_samples: int | None,
    emit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse Cadence Ocean CSV exports and build features/targets arrays.

    Each CSV file contains all samples as sequential parameter blocks.  Input
    features are extracted from the parameter headers inside the blocks and
    converted using the ``parameter_keys`` mapping defined in the schema.
    """
    channel_file_map = schema.ground_truth.channel_file_map
    channel_order = list(schema.ground_truth.ground_truth_parameters)
    param_key_map = schema.input_feature.parameter_key_map
    feature_columns = list(schema.input_feature.feature_columns)
    data_dir = sources.ground_truth_data_dir

    # Parse the first channel file to discover the shared frequency grid and input features.
    first_channel = channel_order[0]
    first_path = data_dir / channel_file_map[first_channel]
    emit(
        "cache_build_started",
        f"Parsing Cadence CSV files for {schema.dataset_name}.",
    )
    params_list, freq_ghz, first_values = _parse_cadence_csv(
        first_path, param_key_map, max_samples,
    )
    emit(
        "input_rows_loaded",
        f"Loaded {len(params_list)} samples from {first_path.name}.",
        total_samples=len(params_list),
    )
    frequency_hz = (freq_ghz * 1e9).astype(np.float32)
    num_samples = len(params_list)
    num_freq = len(frequency_hz)

    # Build the features array from extracted parameters.
    features = np.zeros((num_samples, len(feature_columns)), dtype=np.float32)
    for idx, params in enumerate(params_list):
        features[idx] = [params[col] for col in feature_columns]

    # Build the targets array, one channel per CSV file.
    targets = np.zeros((num_samples, len(channel_order), num_freq), dtype=np.float32)
    targets[0 : len(first_values)] = first_values[:, np.newaxis, :]
    # Treat the first channel as already loaded at index 0.
    # (targets[:, 0, :] is set; we used np.newaxis above for the single-channel shape.)
    targets[:, 0, :] = first_values

    for ch_idx, channel in enumerate(channel_order):
        if ch_idx == 0:
            continue  # Already loaded above.
        ch_path = data_dir / channel_file_map[channel]
        _, ch_freq_ghz, ch_values = _parse_cadence_csv(ch_path, param_key_map, max_samples)
        if not np.allclose(freq_ghz, ch_freq_ghz, rtol=1e-4):
            raise ValueError(f"Frequency grid in {ch_path.name} does not match {first_path.name}.")
        if len(ch_values) != num_samples:
            raise ValueError(
                f"Sample count mismatch: {first_path.name} has {num_samples} samples "
                f"but {ch_path.name} has {len(ch_values)}."
            )
        targets[:, ch_idx, :] = ch_values
        emit(
            "cache_progress",
            f"Loaded channel '{channel}' from {ch_path.name}.",
            completed_samples=num_samples,
            total_samples=num_samples,
        )

    if schema.ground_truth.drop_first_frequency and num_freq > 1:
        frequency_hz = frequency_hz[1:]
        targets = targets[:, :, 1:]

    return features, targets, frequency_hz


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
                if len(parts) == 2:
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


def _cache_matches_sources(meta: dict[str, Any], sources: DataSources, schema: DatasetSchema) -> bool:
    # Reuse a cache only when both the raw data location and the schema hash match.
    cached_input = meta.get("input_feature_path")
    ground_truth_data_dir = meta.get("ground_truth_data_dir")
    schema_hash = meta.get("schema_hash")
    if ground_truth_data_dir is None or schema_hash is None:
        return False
    if Path(ground_truth_data_dir).resolve() != sources.ground_truth_data_dir.resolve():
        return False
    # For file-based sources, also check the input-feature path matches.
    if sources.input_feature_path is not None:
        if cached_input is None:
            return False
        if Path(cached_input).resolve() != sources.input_feature_path.resolve():
            return False
    return schema_hash == schema.schema_hash


# Ground-truth parsing entry point.
def _load_target_sample(path: Path, schema: Ground_TruthSchema) -> tuple[np.ndarray, np.ndarray]:
    # This dispatch point is where support for additional output file formats would
    # be added in the future.
    if schema.format == "touchstone":
        return _parse_touchstone(path, schema)
    raise NotImplementedError(f"Unsupported target format: {schema.format}")


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
    if schema.drop_first_frequency and len(frequency_hz) > 1:
        frequency_hz = frequency_hz[1:]
        matrix = matrix[1:]

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
