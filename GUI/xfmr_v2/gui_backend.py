"""GUI-facing backend helpers for the Surrogate Model Training Suite.

This module keeps the PySide window code focused on presentation and user
interaction. It wraps the existing training/search/suggestion pipeline with a
GUI-friendly surface that can:

- scan datasets and summarize README-driven schema details
- enumerate the CUDA / CPU devices available for training
- validate transfer-learning compatibility
- orchestrate baseline-only or baseline-plus-transfer runs
- export trained checkpoints to ONNX (for MATLAB or other runtimes)
- save and load GUI config files
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .app_paths import current_runtime_paths
from .atomic_json import write_json_atomically
from .data import build_cache_from_dataset, load_existing_cache, load_split_bundle
from .export_onnx import export_checkpoint_to_onnx
from .runner import TrainConfig, TransferConfig, run_self_transfer, train_baseline
from .search import SearchConfig, quick_hyperparameter_search
from .suggest import SuggestConfig, suggest_initial_settings

def list_available_devices() -> list[dict[str, str]]:
    """Enumerate the compute devices the user can train on.

    Each entry has an ``id`` (a ``torch.device`` string such as ``"cuda:0"`` or
    ``"cpu"``) and a human-readable ``label``. CUDA devices are listed first when
    PyTorch reports them as available, followed by CPU as an always-present
    fallback. This is cheap enough to call synchronously when the GUI starts.
    """

    devices: list[dict[str, str]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            memory_gb = round(torch.cuda.get_device_properties(index).total_memory / (1024**3), 1)
            devices.append({
                "id": f"cuda:{index}",
                "label": f"cuda:{index} — {name} ({memory_gb} GB)",
            })
    devices.append({"id": "cpu", "label": "cpu — CPU"})
    return devices


def scan_dataset(
    *,
    cache_path: str,
    train_frac: float,
    val_frac: float,
    seed: int,
    max_samples: int | None = None,
    split_corner_columns: list[str] | None = None,
    split_design_columns: list[str] | None = None,
    progress_callback=None,
    should_stop=None,
    # Legacy params — kept for signature compat but ignored.
    input_feature_path: str = "",
    ground_truth_data_dir: str = "",
    dataset_root: str = "",
    overwrite_mismatched_cache: bool = False,
) -> dict[str, Any]:
    """Load an existing cache and summarize the dataset for the GUI.

    New caches are created via ``build_cache_from_dataset``.  This function
    only loads caches that already exist.
    """
    if progress_callback is not None:
        progress_callback(
            {"phase": "scan", "event": "started",
             "message": "Loading cached dataset.",
             "cache_path": cache_path}
        )

    cache_summary = load_existing_cache(cache_path, progress_callback)

    if progress_callback is not None:
        progress_callback(
            {"phase": "scan", "event": "split_loading_started",
             "message": "Preparing train, validation, and test splits.",
             "cache_path": cache_summary["cache_path"],
             "cache_status": cache_summary["status"]}
        )

    # Preview the SAME split the run would use. A scan may run before the corner
    # picker is validated against this dataset (scanning is what populates it),
    # so a bad selection must not break the scan itself — fall back to the
    # row-level preview and say so, instead of failing or silently previewing a
    # split the run will not use.
    split_note = ""
    split_warnings: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bundle = load_split_bundle(
                cache_path=cache_path,
                batch_size=16,
                seed=seed,
                train_frac=train_frac,
                val_frac=val_frac,
                max_samples=max_samples,
                pin_memory=False,
                split_corner_columns=split_corner_columns,
                split_design_columns=split_design_columns,
            )
        split_warnings = [str(entry.message) for entry in caught]
    except ValueError as error:
        if not (split_corner_columns or split_design_columns):
            raise
        split_note = str(error)
        bundle = load_split_bundle(
            cache_path=cache_path,
            batch_size=16,
            seed=seed,
            train_frac=train_frac,
            val_frac=val_frac,
            max_samples=max_samples,
            pin_memory=False,
        )

    if bundle.design_counts is not None:
        counts = bundle.design_counts
        split_mode = (
            f"Design-level: {counts['train']} / {counts['val']} / {counts['test']} "
            "train/val/test designs (all corner rows of a design stay in one fold)"
        )
    elif split_note:
        split_mode = f"Row-level (requested design-level split unavailable: {split_note})"
    else:
        split_mode = "Row-level"

    schema = cache_summary.get("dataset_schema", {})
    preview_rows = [
        ("Samples", str(cache_summary["num_samples"])),
        ("Training Samples", str(len(bundle.split_indices["train"]))),
        ("Validation Samples", str(len(bundle.split_indices["val"]))),
        ("Test Samples", str(len(bundle.split_indices["test"]))),
        ("Split Mode", split_mode),
        ("Split Warnings", _join_values(split_warnings) or "None"),
        ("Sweep Points", str(cache_summary["num_frequencies"])),
        ("Dataset Name", cache_summary.get("dataset_name", "")),
        ("Active Input-Feature Columns", _join_values(bundle.active_names)),
        ("Constant Fields Removed", _join_values(bundle.dropped_names) or "None"),
        ("Ground-Truth Channels", _join_values(bundle.channel_names)),
        ("Cache Status", cache_summary["status"]),
    ]

    result = {
        "status": "ok",
        "cache_summary": cache_summary,
        "schema_status": "Valid",
        "dataset_name": cache_summary.get("dataset_name", ""),
        "readme_path": "",
        "dataset_root": cache_summary.get("dataset_root", ""),
        "input_feature_path": cache_summary.get("input_feature_path", ""),
        "ground_truth_data_dir": cache_summary.get("ground_truth_data_dir", ""),
        "cache_path": cache_summary["cache_path"],
        "schema": schema,
        "preview_rows": preview_rows,
        "split_warnings": split_warnings,
        "active_input_feature_names": bundle.active_names,
        "dropped_input_feature_names": bundle.dropped_names,
        "frequency_count": int(cache_summary["num_frequencies"]),
        "frequency_range_ghz": [
            float(bundle.frequency_ghz.min()),
            float(bundle.frequency_ghz.max()),
        ],
        "sweep_label": getattr(bundle, "sweep_label", "Frequency (GHz)"),
    }
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "scan",
                "event": "completed",
                "message": f"Dataset scan completed for {result['dataset_name']}.",
                "dataset_name": result["dataset_name"],
                "readme_path": result["readme_path"],
                "total_samples": int(cache_summary["num_samples"]),
                "frequency_count": result["frequency_count"],
            }
        )
    return result


def build_and_scan_dataset(
    *,
    cache_path: str,
    train_frac: float,
    val_frac: float,
    seed: int,
    max_samples: int | None = None,
    split_corner_columns: list[str] | None = None,
    split_design_columns: list[str] | None = None,
    dataset_root: str = "",
    input_feature_path: str = "",
    ground_truth_data_dir: str = "",
    overwrite: bool = False,
    progress_callback=None,
    should_stop=None,
) -> dict[str, Any]:
    """Build a cache from the chosen dataset folder (if needed), then scan it.

    Builds the cache via ``build_cache_from_dataset`` (format auto-detected from the
    folder contents) when no cache exists yet (or when ``overwrite`` is set), then
    loads and summarizes it for the GUI. This is the folder -> cache -> preview
    entry point.
    """
    build_root = dataset_root or ground_truth_data_dir or (
        str(Path(input_feature_path).parent) if input_feature_path else ""
    )
    if overwrite or not Path(cache_path).is_file():
        if not build_root:
            raise ValueError("Select a dataset folder so the cache can be built.")
        build_cache_from_dataset(
            build_root,
            cache_path,
            max_samples=max_samples,
            progress_callback=progress_callback,
            should_stop=should_stop,
        )

    return scan_dataset(
        cache_path=cache_path,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
        max_samples=max_samples,
        split_corner_columns=split_corner_columns,
        split_design_columns=split_design_columns,
        progress_callback=progress_callback,
        should_stop=should_stop,
    )


def run_training_workflow(
    *,
    baseline_config: TrainConfig | None,
    transfer_config: TransferConfig | None,
    progress_callback=None,
    should_stop=None,
) -> dict[str, Any]:
    """Run baseline-only, transfer-only, or baseline-plus-transfer from the GUI.

    Self-transfer is standalone: it trains its own per-band models from scratch and
    computes normalization from the dataset cache, so it does not depend on a baseline
    run.
    """

    baseline_summary: dict[str, Any] | None = None
    transfer_summary: dict[str, Any] | None = None

    if baseline_config is not None:
        baseline_summary = train_baseline(
            baseline_config,
            show_progress=False,
            progress_callback=progress_callback,
            should_stop=should_stop,
        )
        if baseline_summary.get("status") == "stopped":
            return {"status": "stopped", "baseline": baseline_summary, "transfer": None}

    if transfer_config is not None:
        transfer_summary = run_self_transfer(
            transfer_config,
            show_progress=False,
            progress_callback=progress_callback,
            should_stop=should_stop,
        )
        if transfer_summary.get("status") == "stopped":
            return {"status": "stopped", "baseline": baseline_summary, "transfer": transfer_summary}

    return {
        "status": "ok",
        "baseline": baseline_summary,
        "transfer": transfer_summary,
    }


def export_model_to_onnx(
    *,
    checkpoint_path: str,
    out_path: str | None = None,
    bake_normalization: bool = True,
    progress_callback=None,
    should_stop=None,
) -> dict[str, Any]:
    """Export a trained checkpoint to ONNX (for MATLAB or any ONNX runtime).

    Writes ``<out>.onnx`` plus a ``<out>.meta.json`` sidecar and returns their paths.
    Normalization is baked into the graph, so the exported model maps raw design
    parameters straight to physical-unit curves.
    """
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "export",
                "event": "started",
                "message": f"Exporting {Path(checkpoint_path).name} to ONNX...",
                "checkpoint_path": checkpoint_path,
            }
        )

    # torch.onnx.export serializes through the `onnx` package; surface a clear,
    # actionable error instead of a deep torch traceback when it is missing.
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The 'onnx' package is required to export ONNX models. Install it with: pip install onnx"
        ) from exc

    onnx_path = export_checkpoint_to_onnx(
        checkpoint_path,
        out_path,
        bake_normalization=bake_normalization,
    )
    meta_path = onnx_path.with_suffix(".meta.json")

    if progress_callback is not None:
        progress_callback(
            {
                "phase": "export",
                "event": "completed",
                "message": f"ONNX export complete: {onnx_path}",
                "onnx_path": str(onnx_path),
            }
        )

    return {"status": "ok", "onnx_path": str(onnx_path), "meta_path": str(meta_path)}


def save_gui_config(path: str | Path, payload: dict[str, Any]) -> None:
    """Save one GUI config payload as JSON, atomically (see atomic_json)."""

    write_json_atomically(path, payload)


def load_gui_config(path: str | Path) -> dict[str, Any]:
    """Load one GUI config payload."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_last_session(payload: dict[str, Any]) -> None:
    """Persist the most recent GUI state into the active runtime state folder."""

    save_gui_config(current_runtime_paths().last_session_path, payload)


def load_last_session() -> dict[str, Any] | None:
    """Load the most recent saved GUI state when available."""

    path = current_runtime_paths().last_session_path
    if not path.is_file():
        return None
    return load_gui_config(path)


def build_suggest_result(
    *,
    input_feature_path: str = "",
    ground_truth_data_dir: str = "",
    dataset_root: str = "",
    cache_path: str,
    seed: int,
    train_frac: float,
    val_frac: float,
    max_samples: int | None,
    split_corner_columns: list[str] | None = None,
    split_design_columns: list[str] | None = None,
    progress_callback=None,
    should_stop=None,
) -> dict[str, Any]:
    """Thin GUI wrapper around the scan-only suggestion backend."""

    if dataset_root:
        effective_data_root = dataset_root
        effective_input = None
        effective_gt = None
    elif not input_feature_path and ground_truth_data_dir:
        effective_data_root = ground_truth_data_dir
        effective_input = None
        effective_gt = None
    else:
        effective_data_root = None
        effective_input = input_feature_path or None
        effective_gt = ground_truth_data_dir or None

    return suggest_initial_settings(
        SuggestConfig(
            data_root=effective_data_root,
            input_feature_path=effective_input,
            ground_truth_data_dir=effective_gt,
            cache_path=cache_path,
            seed=seed,
            train_frac=train_frac,
            val_frac=val_frac,
            split_corner_columns=split_corner_columns,
            split_design_columns=split_design_columns,
            max_samples=max_samples,
        ),
        show_progress=False,
        progress_callback=progress_callback,
        should_stop=should_stop,
    )


def run_search(
    config: SearchConfig,
    *,
    progress_callback=None,
    should_stop=None,
) -> dict[str, Any]:
    """Run quick search with GUI-friendly defaults."""

    filtered_progress = _gui_search_progress_filter(progress_callback)
    return quick_hyperparameter_search(
        config,
        show_progress=False,
        progress_callback=filtered_progress,
        should_stop=should_stop,
    )


def default_run_name(input_feature_path: str | Path) -> str:
    """Derive a readable run name from the selected dataset."""

    path = Path(input_feature_path)
    stem = path.parent.name or path.stem
    return stem.replace(" ", "_").lower()


def make_run_roots(model_output_dir: str, run_name: str) -> dict[str, str]:
    """Build artifact roots for search, baseline, and transfer outputs."""

    root = Path(model_output_dir) / run_name
    return {
        "root": str(root),
        "search": str(root / "search"),
        "baseline": str(root / "baseline"),
        "transfer": str(root / "transfer"),
    }


def _gui_search_progress_filter(progress_callback):
    if progress_callback is None:
        return None

    def emit(payload: dict[str, Any]) -> None:
        if not _should_forward_search_progress(payload):
            return
        progress_callback(payload)

    return emit


def _should_forward_search_progress(payload: dict[str, Any]) -> bool:
    # emit_progress (runner/search) nests per-run fields under "data" (see
    # ProgressEvent.to_dict), while scan callbacks emit them flat — accept both.
    data = payload.get("data") or {}

    def field(key: str) -> Any:
        return data.get(key, payload.get(key))

    if payload.get("phase") != "baseline" or field("trial_index") is None:
        return True
    event = payload.get("event")
    if event == "checkpoint_updated":
        return False
    if event != "epoch_end":
        return True
    epoch = int(field("epoch") or 0)
    total_epochs = int(field("total_epochs") or 0)
    return epoch <= 1 or epoch == total_epochs or epoch % 5 == 0


def _join_values(values: list[str] | tuple[str, ...]) -> str:
    return ", ".join(values)
