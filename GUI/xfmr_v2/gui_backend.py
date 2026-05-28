"""GUI-facing backend helpers for the Surrogate Model Traning Suite.

This module keeps the PySide window code focused on presentation and user
interaction. It wraps the existing training/search/suggestion pipeline with a
GUI-friendly surface that can:

- scan datasets and summarize README-driven schema details
- enumerate the CUDA / CPU devices available for training
- validate transfer-learning compatibility
- orchestrate baseline-only or baseline-plus-transfer runs
- save and load GUI config files
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .app_paths import current_runtime_paths
from .data import load_existing_cache, load_split_bundle
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
    progress_callback=None,
    should_stop=None,
    # Legacy params — kept for signature compat but ignored.
    input_feature_path: str = "",
    ground_truth_data_dir: str = "",
    dataset_root: str = "",
    overwrite_mismatched_cache: bool = False,
) -> dict[str, Any]:
    """Load an existing cache and summarize the dataset for the GUI.

    New caches are created via ``build_cache_from_loader`` (the 'Run & Cache'
    button).  This function only loads caches that already exist.
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

    bundle = load_split_bundle(
        cache_path=cache_path,
        batch_size=16,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        max_samples=max_samples,
        pin_memory=False,
    )

    schema = cache_summary.get("dataset_schema", {})
    preview_rows = [
        ("Samples", str(cache_summary["num_samples"])),
        ("Training Samples", str(len(bundle.split_indices["train"]))),
        ("Validation Samples", str(len(bundle.split_indices["val"]))),
        ("Test Samples", str(len(bundle.split_indices["test"]))),
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


def check_transfer_compatibility(
    *,
    base_run_dir: str,
    cache_path: str,
    input_feature_path: str | None = None,
    ground_truth_data_dir: str | None = None,
    progress_callback=None,
    should_stop=None,
) -> dict[str, Any]:
    """Check whether one baseline run can be used with the current dataset cache."""

    run_dir = Path(base_run_dir)
    checkpoint_path = run_dir / "best_model.pt"
    summary_path = run_dir / "summary.json"
    if not checkpoint_path.is_file():
        return {
            "status": "Mismatch",
            "message": "The selected baseline run does not contain best_model.pt.",
        }
    if not Path(cache_path).is_file():
        return {
            "status": "Mismatch",
            "message": "The selected cache file does not exist yet.",
        }

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    with np.load(cache_path, allow_pickle=False) as data:
        cache_input_feature_names = _array_names(data, "input_feature_names", "feature_names")
        cache_channel_names = _array_names(data, "channel_names")
        frequency_count = int(data["frequency_hz"].shape[0])

    checkpoint_active_names = list(
        checkpoint["active_input_feature_names"]
        if "active_input_feature_names" in checkpoint
        else checkpoint["active_feature_names"]
    )
    checkpoint_channel_names = list(checkpoint["target_channel_names"])
    if any(name not in cache_input_feature_names for name in checkpoint_active_names):
        return {
            "status": "Mismatch",
            "message": "The selected baseline run expects input features that are not present in the current cache.",
        }
    if checkpoint_channel_names != cache_channel_names:
        return {
            "status": "Mismatch",
            "message": "The selected baseline run predicts different ground-truth channels than the current cache.",
        }

    cfg = checkpoint.get("config", {})
    mismatch_notes: list[str] = []
    if input_feature_path:
        checkpoint_input_feature_path = cfg.get("input_feature_path") or cfg.get("input_path")
        if checkpoint_input_feature_path not in (None, "", input_feature_path):
            mismatch_notes.append("Baseline run was created from a different input-feature path.")
    if ground_truth_data_dir:
        checkpoint_ground_truth_path = cfg.get("ground_truth_data_dir") or cfg.get("output_data_dir")
        if checkpoint_ground_truth_path not in (None, "", ground_truth_data_dir):
            mismatch_notes.append("Baseline run was created from a different ground-truth data folder.")

    message = "The selected baseline run is compatible with the current cache."
    if mismatch_notes:
        message += " " + " ".join(mismatch_notes)

    summary: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return {
        "status": "Compatible",
        "message": message,
        "frequency_count": frequency_count,
        "active_input_feature_names": checkpoint_active_names,
        "ground_truth_channels": checkpoint_channel_names,
        "base_run_dir": str(run_dir.resolve()),
        "summary": summary,
    }


def run_training_workflow(
    *,
    baseline_config: TrainConfig | None,
    transfer_config: TransferConfig | None,
    transfer_base_run_dir: str | None,
    progress_callback=None,
    should_stop=None,
) -> dict[str, Any]:
    """Run baseline-only, transfer-only, or baseline-plus-transfer from the GUI."""

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
        if transfer_base_run_dir:
            transfer_config.base_run_dir = transfer_base_run_dir
        elif baseline_summary is not None:
            transfer_config.base_run_dir = baseline_summary["run_dir"]
        else:
            raise ValueError("Transfer learning requires either a baseline run from this session or an existing base run.")
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


def save_gui_config(path: str | Path, payload: dict[str, Any]) -> None:
    """Save one GUI config payload as JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    if payload.get("phase") != "baseline" or payload.get("trial_index") is None:
        return True
    event = payload.get("event")
    if event == "checkpoint_updated":
        return False
    if event != "epoch_end":
        return True
    epoch = int(payload.get("epoch", 0) or 0)
    total_epochs = int(payload.get("total_epochs", 0) or 0)
    return epoch <= 1 or epoch == total_epochs or epoch % 5 == 0


def _join_values(values: list[str] | tuple[str, ...]) -> str:
    return ", ".join(values)


def _array_names(data: np.lib.npyio.NpzFile, *keys: str) -> list[str]:
    for key in keys:
        if key in data:
            return data[key].astype(str).tolist()
    raise KeyError(f"None of the expected array keys were present: {keys}")
