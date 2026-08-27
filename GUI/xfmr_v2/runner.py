"""Training and self-transfer runners for the v2 XFMR project.

This module sits at the center of the repository:
- it turns cached data into model training loops
- it evaluates checkpoints and writes artifacts
- it implements the optional self-transfer experiment

The helper functions near the bottom are intentionally kept in this same file so
the full training flow can be read top-to-bottom in one place.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    CACHE_PATH,
    DATA_ROOT,
    design_split_indices,
    ensure_cache,
    load_split_bundle,
    resolve_split_fractions,
    split_indices,
)
from .model import SpectraHydra, SpectraHydraProj, SpectraNet
from .progress import ProgressCallback, RunCancelled, StopChecker, emit_progress, request_stop


#
# Configuration objects
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Baseline training configuration."""

    # Data and artifact locations.
    data_root: str | None = str(DATA_ROOT)
    input_feature_path: str | None = None
    ground_truth_data_dir: str | None = None
    cache_path: str = str(CACHE_PATH)
    output_dir: str = "artifacts/runs/baseline_v2"

    # Optimization settings.
    seed: int = 42
    batch_size: int = 16
    epochs: int = 300
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Dataset split settings.
    train_frac: float = 0.8
    val_frac: float = 0.1
    # Optional design-level split: names of the input columns that vary across the
    # PVT corners of one design (e.g. Temp_C, VDD, proc_* one-hots). When set, rows
    # identical in every OTHER column are one design and all of its corner rows land
    # in the same train/val/test fold, so no design leaks between splits. ``None``
    # keeps the default row-level split. Works with every model type.
    #
    # This list must cover EVERY column that moves with the corner, derived ones
    # included (physics anchors, a frozen corner embedding); miss one and each
    # design splinters into per-corner designs, which is a row-level split wearing
    # a design-level label. Prefer split_design_columns below, where the same
    # mistake merges designs instead — coarser, never leakier. Set at most one.
    split_corner_columns: list[str] | None = None
    # Names of the columns that IDENTIFY a design (the geometry parameters);
    # everything else is treated as corner-varying.
    split_design_columns: list[str] | None = None

    # Model architecture settings.
    model_type: str = "SpectraNet"  # "SpectraNet", "SpectraHydra", or "SpectraHydraProj"
    width: int = 512
    depth: int = 5
    # SpectraHydraProj only: names of the PVT corner/condition feature columns
    # (e.g. Temp_C, VDD, process one-hots) fed to the learned corner projection,
    # and the width of the projected embedding. Stored as names — not indices —
    # because constant columns are dropped per training split, which would shift
    # raw indices. Ignored by the other model types.
    projection_columns: list[str] | None = None
    projection_dim: int = 16

    # Loss and scheduler settings.
    loss_function: str = "rmse"  # "rmse" or "mse"
    scheduler: str = "plateau"   # "plateau" or "cosine"

    # Runtime controls.
    use_amp: bool = True
    max_samples: int | None = None
    # Compute device to train on, e.g. "cuda:0", "cuda:1", or "cpu".
    # ``None`` means auto-select (first CUDA device when available, else CPU).
    device: str | None = None

    # Optional external evaluation set: an .npz file holding ``features``,
    # ``targets``, and ``feature_names`` arrays (a dataset bundle.npz qualifies).
    # After training, the best checkpoint is also scored on this file and the
    # per-channel MAE is reported next to the internal test metrics. Point it at
    # data the training dataset never contains (e.g. extra held-out designs) to
    # get an honest external benchmark — the internal test fold changes with the
    # seed, this number does not. Feature columns are matched by name; targets
    # must share the training dataset's channel order, frequency grid, and value
    # space. The file is validated before training starts.
    eval_dataset_path: str | None = None


@dataclass
class TransferConfig:
    """Standalone self-transfer configuration (no baseline run required).

    Band 0 is trained from scratch on its frequency slice; later bands are seeded
    from their neighbours during the forward/backward sweep. Normalization and the
    train/val/test split are computed from the cache's training split, exactly as
    baseline training does, so reported MAE is comparable.
    """

    cache_path: str = str(CACHE_PATH)
    output_dir: str = "artifacts/runs/self_transfer_v2"
    model_type: str = "SpectraNet"
    width: int = 512
    depth: int = 5
    # SpectraHydraProj only — same semantics as the TrainConfig fields.
    projection_columns: list[str] | None = None
    projection_dim: int = 16
    seed: int = 42
    train_frac: float = 0.8
    val_frac: float = 0.1
    # Same semantics as the ``TrainConfig`` fields: either one switches the split to
    # design-level so no design's corner rows leak between train and test.
    split_corner_columns: list[str] | None = None
    split_design_columns: list[str] | None = None
    num_bands: int = 10
    iterations: int = 10
    transfer_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 5e-5
    weight_decay: float | None = None
    use_amp: bool = True
    # Compute device to train on, e.g. "cuda:0", "cuda:1", or "cpu".
    # ``None`` means auto-select (first CUDA device when available, else CPU).
    device: str | None = None


MODEL_TYPES = ("SpectraNet", "SpectraHydra", "SpectraHydraProj")

# Back-compat: runs saved under the old model-type names still load. Map the legacy
# string to its current equivalent so old checkpoints, configs, and saved GUI forms
# keep working after the rename.
_MODEL_TYPE_ALIASES = {
    "FlatMLP": "SpectraNet",
    "CTLE_MLP": "SpectraHydra",
}


def canonical_model_type(model_type: str) -> str:
    """Normalize a (possibly legacy) model-type string to its current name."""
    return _MODEL_TYPE_ALIASES.get(model_type, model_type)

# LR-reduction patience for the "plateau" scheduler. Training no longer early-stops,
# so this only controls when ReduceLROnPlateau lowers the learning rate.
_PLATEAU_LR_PATIENCE = 10


def resolve_device(spec: str | None) -> torch.device:
    """Turn a user-supplied device string into a usable ``torch.device``.

    ``None`` (or an empty string) auto-selects the first CUDA device when CUDA is
    available, otherwise CPU. A requested CUDA device falls back to CPU when CUDA
    is unavailable so a stale selection can never crash a run.

    When a specific CUDA device is selected, it is also made the current CUDA
    device so AMP / ``GradScaler`` and any current-device-implicit ops land on the
    same GPU the tensors are moved to.
    """
    if not spec:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif spec.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(spec)
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            # A persisted selection like "cuda:1" after moving to a one-GPU
            # machine would otherwise raise "invalid device ordinal".
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(device)
    return device


# Torch's CPU thread pool spin-waits, so on a many-core host it burns enormous CPU on
# the trivial host-side work GPU training does (batch collation, per-step loss readback).
_GPU_TRAINING_CPU_THREADS = 4


def limit_cpu_threads_for_gpu(device: torch.device) -> None:
    """Shrink torch's CPU thread pool when the real work happens on a GPU.

    Measured on a 128-thread host (L40, 9.2k samples, batch 64): one epoch used
    ~102 s of CPU time across ~127 spinning threads to produce 0.8 s of wall-clock
    progress. Capping the pool cuts that to ~0.7 s of CPU time and is marginally
    *faster*, because these tensors are far too small to profit from a wide pool.

    CPU training is left untouched, where a wide pool genuinely does help.
    """
    if device.type != "cuda":
        return
    if torch.get_num_threads() > _GPU_TRAINING_CPU_THREADS:
        torch.set_num_threads(_GPU_TRAINING_CPU_THREADS)


def build_model(model_type: str, *, num_frequencies: int, **kwargs: Any) -> nn.Module:
    """Instantiate a model by name, forwarding architecture kwargs.

    Legacy model-type names (e.g. saved before the rename) are accepted and
    normalized via :func:`canonical_model_type`.
    """
    model_type = canonical_model_type(model_type)
    if model_type == "SpectraNet":
        return SpectraNet(num_frequencies=num_frequencies, **kwargs)
    if model_type == "SpectraHydra":
        return SpectraHydra(num_frequencies=num_frequencies, **kwargs)
    if model_type == "SpectraHydraProj":
        return SpectraHydraProj(num_frequencies=num_frequencies, **kwargs)
    raise ValueError(f"Unknown model_type {model_type!r}. Choose from {MODEL_TYPES}.")


def resolve_projection_kwargs(
    model_type: str,
    projection_columns: list[str] | None,
    projection_dim: int,
    active_names: list[str],
    dropped_names: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build the extra :func:`build_model` kwargs for ``SpectraHydraProj``.

    Returns an empty dict for every other model type, so callers can splat the
    result unconditionally. Corner columns are configured by NAME and resolved
    here — at model-build time — to indices in the model's input order
    (``active_names``), because constant columns are dropped per training split
    and raw cache indices would silently shift.

    A configured column that was dropped as constant is skipped: a constant
    column carries no corner information in this dataset. Unknown names, an
    empty configuration, and a configuration where every column was dropped all
    raise so the run fails loudly instead of training a degenerate projection.
    """
    if canonical_model_type(model_type) != "SpectraHydraProj":
        return {}
    # Dedupe while preserving order (e.g. a repeated name in a CLI flag) so the
    # projection never embeds the same column twice.
    requested = list(dict.fromkeys(str(name) for name in (projection_columns or []) if str(name).strip()))
    if not requested:
        raise ValueError(
            "SpectraHydraProj needs at least one PVT corner column. Set "
            "projection_columns to names from the dataset's input features "
            "(e.g. temperature, supply, or process-corner columns)."
        )
    known = set(active_names) | set(dropped_names)
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ValueError(
            f"Projection corner column(s) {unknown} are not input features of this "
            f"dataset. Available columns: {sorted(known)}."
        )
    indices = [active_names.index(name) for name in requested if name in active_names]
    if not indices:
        raise ValueError(
            f"All configured projection corner columns {requested} are constant in "
            "the training split, so there is no corner information to project. "
            "Pick columns that vary across samples (e.g. Temp_C, VDD, proc_*)."
        )
    skipped = [name for name in requested if name not in active_names]
    if skipped:
        # Constancy is a property of the current training split (seed / fractions /
        # max_samples), not the dataset, so a silently smaller embedding could
        # otherwise diverge between runs that share a config. Warn loudly.
        warnings.warn(
            f"Projection corner column(s) {skipped} are constant in this training "
            f"split and were skipped; the corner projection embeds only "
            f"{[active_names[i] for i in indices]}.",
            stacklevel=2,
        )
    return {"corner_indices": indices, "projection_dim": int(projection_dim)}


def frequency_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RMSE over frequency, averaged across channels."""
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=-1) + 1e-12).mean()


def frequency_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum of per-channel MSE — matches the notebook pattern ``loss_g + loss_p``.

    For single-channel models this is identical to ``nn.MSELoss()``.  For
    multi-channel models the per-channel MSE values are summed (not averaged),
    so each head contributes equal gradient magnitude regardless of how many
    channels there are.
    """
    # pred/target shape: (batch, channels, frequency)
    # Per-channel mean over (batch, frequency), then summed over channels. Written as
    # one vectorized reduction rather than a Python loop over channels: identical value,
    # 3 kernels instead of ~4 per channel plus one backward node instead of many.
    # For a single channel this reduces to plain MSE, as before.
    return ((pred - target) ** 2).mean(dim=(0, 2)).sum()


LOSS_FUNCTIONS = ("rmse", "mse")
SCHEDULER_TYPES = ("plateau", "cosine")


def _build_loss_fn(name: str):
    if name == "mse":
        return frequency_mse
    if name == "rmse":
        return frequency_rmse
    raise ValueError(f"Unknown loss_function {name!r}. Choose from {LOSS_FUNCTIONS}.")


def _progress_data(event_context: dict[str, Any] | None = None, **data: Any) -> dict[str, Any]:
    # Merge caller-provided context with event-specific fields so nested workflows
    # can add metadata like `trial_index` or `transfer_iteration` without repeating it.
    payload = dict(event_context or {})
    payload.update(data)
    return payload


#
# Baseline training flow
# -----------------------------------------------------------------------------
def run_baseline_trial(
    config: TrainConfig,
    show_progress: bool = True,
    save_artifacts: bool = False,
    evaluation_split: str = "val",
    run_dir: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
    should_stop: StopChecker | None = None,
    event_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one baseline configuration and evaluate the best checkpoint on one split."""
    if evaluation_split not in {"train", "val", "test"}:
        raise ValueError("evaluation_split must be one of: train, val, test.")
    emit_progress(
        progress_callback,
        event="started",
        phase="baseline",
        message="Baseline training started.",
        **_progress_data(event_context, config=asdict(config), evaluation_split=evaluation_split),
    )
    seed_all(config.seed)
    enable_fast_cuda()
    device = resolve_device(config.device)
    limit_cpu_threads_for_gpu(device)
    bundle: Any | None = None
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    start_time = perf_counter()
    try:
        request_stop(should_stop)
        # Stage 1: make sure the cache exists and points at the requested dataset.
        emit_progress(
            progress_callback,
            event="cache_check_started",
            phase="baseline",
            message="Checking that the training cache matches the selected dataset.",
            **_progress_data(event_context, cache_path=config.cache_path, device=str(device)),
        )
        cache_path = ensure_cache(
            data_root=config.data_root,
            cache_path=config.cache_path,
            max_samples=config.max_samples,
            input_feature_path=config.input_feature_path,
            ground_truth_data_dir=config.ground_truth_data_dir,
            should_stop=should_stop,
        )
        emit_progress(
            progress_callback,
            event="cache_ready",
            phase="baseline",
            message="Training cache is ready.",
            **_progress_data(event_context, cache_path=str(cache_path), device=str(device)),
        )
        request_stop(should_stop)
        # Stage 2: load cached arrays, create the split, and compute normalization
        # statistics from the training split.
        emit_progress(
            progress_callback,
            event="split_loading_started",
            phase="baseline",
            message="Loading cached arrays and building the train / validation / test split.",
            **_progress_data(event_context, cache_path=str(cache_path)),
        )
        # An External Eval Set supplies the accuracy number, so the internal test
        # fold is redundant and its rows go to training instead. Keyed on the field
        # being set, not on the file loading, because the split is built here and
        # the file is only read further down.
        merge_test_into_train = bool(config.eval_dataset_path)
        effective_train_frac, _ = resolve_split_fractions(
            config.train_frac, config.val_frac, merge_test_into_train
        )
        bundle = load_split_bundle(
            cache_path=cache_path,
            batch_size=config.batch_size,
            seed=config.seed,
            train_frac=config.train_frac,
            val_frac=config.val_frac,
            max_samples=config.max_samples,
            pin_memory=device.type == "cuda",
            split_corner_columns=config.split_corner_columns,
            split_design_columns=config.split_design_columns,
            merge_test_into_train=merge_test_into_train,
        )
        if bundle.design_counts is not None:
            data_ready_message = (
                "Training data split and normalization are ready. Design-level split: "
                f"{bundle.design_counts['train']} train / {bundle.design_counts['val']} val / "
                f"{bundle.design_counts['test']} test designs (all corner rows of a design "
                "stay in one fold)."
            )
        else:
            data_ready_message = "Training data split and normalization are ready."
        if merge_test_into_train:
            # The recorded train_frac is the requested one, so the log has to name
            # the fraction actually trained on or the run looks like a plain 0.8 run.
            data_ready_message += (
                f" External Eval Set configured, so the internal test fold was merged "
                f"into training: train fraction {config.train_frac:g} -> "
                f"{effective_train_frac:g}, no test fold."
            )
        emit_progress(
            progress_callback,
            event="data_ready",
            phase="baseline",
            message=data_ready_message,
            **_progress_data(
                event_context,
                train_samples=int(len(bundle.split_indices["train"])),
                val_samples=int(len(bundle.split_indices["val"])),
                test_samples=int(len(bundle.split_indices["test"])),
                effective_train_frac=float(effective_train_frac),
                test_fold_merged_into_train=bool(merge_test_into_train),
                design_counts=bundle.design_counts,
                active_input_feature_names=bundle.active_names,
                dropped_input_feature_names=bundle.dropped_names,
                frequency_count=int(len(bundle.frequency_ghz)),
                frequency_min_ghz=float(bundle.frequency_ghz.min()),
                frequency_max_ghz=float(bundle.frequency_ghz.max()),
            ),
        )

        # Load (and thereby validate) the external evaluation set before any
        # training time is spent: a bad path or a mismatched file must fail now,
        # not after the full epoch budget.
        external_eval: tuple[np.ndarray, np.ndarray, list[str]] | None = None
        if config.eval_dataset_path:
            external_eval = _load_external_eval(config.eval_dataset_path, bundle)
            ready_message = (
                f"External evaluation set is ready: {len(external_eval[0])} rows "
                f"from {Path(config.eval_dataset_path).name}."
            )
            for note in external_eval[2]:
                ready_message += f" Note: {note}"
            emit_progress(
                progress_callback,
                event="external_eval_ready",
                phase="baseline",
                message=ready_message,
                **_progress_data(
                    event_context,
                    eval_dataset_path=str(config.eval_dataset_path),
                    external_eval_rows=int(len(external_eval[0])),
                    external_eval_notes=list(external_eval[2]),
                ),
            )

        # Stage 3: create the model and optimization objects.
        model = build_model(
            config.model_type,
            num_frequencies=len(bundle.frequency_ghz),
            input_feature_dim=len(bundle.active_names),
            ground_truth_channels=len(bundle.channel_names),
            width=config.width,
            depth=config.depth,
            **resolve_projection_kwargs(
                config.model_type,
                config.projection_columns,
                config.projection_dim,
                bundle.active_names,
                bundle.dropped_names,
            ),
        ).to(device)
        loss_fn = _build_loss_fn(config.loss_function)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        if config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        else:
            # Fixed LR-reduction patience: this is the plateau scheduler's own knob for
            # lowering the learning rate, not training early stopping (which was removed).
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=_PLATEAU_LR_PATIENCE
            )
        amp = config.use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        emit_progress(
            progress_callback,
            event="model_ready",
            phase="baseline",
            message="Model and optimizer are ready. Training epochs are starting.",
            **_progress_data(
                event_context,
                parameter_count=int(sum(parameter.numel() for parameter in model.parameters())),
                amp_enabled=bool(amp),
                device=str(device),
            ),
        )

        # Stage 4: run the full train/validation loop, tracking the best checkpoint by
        # validation loss. Training always runs the full epoch budget (no early stopping).
        for epoch in range(1, config.epochs + 1):
            request_stop(should_stop)
            train_loss = _run_epoch(
                model,
                bundle.train_loader,
                optimizer,
                scaler,
                device,
                amp,
                should_stop=should_stop,
                loss_fn=loss_fn,
            )
            val_loss = _eval_loss(model, bundle.val_loader, device, amp, should_stop=should_stop, loss_fn=loss_fn)
            if config.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if show_progress:
                print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            # A NaN best_val (e.g. an AMP loss spike at epoch 1) must not pin the
            # checkpoint forever: every later `x < nan` comparison is False.
            improved = best_state is None or math.isnan(best_val) or val_loss < best_val
            if improved:
                # Clone the state dict so later optimizer steps cannot mutate the stored
                # "best so far" checkpoint in place.
                best_val, best_epoch = val_loss, epoch
                best_state = clone_state(model.state_dict())
            elapsed_seconds = perf_counter() - start_time
            eta_seconds = 0.0 if epoch <= 0 else max((config.epochs - epoch) * elapsed_seconds / epoch, 0.0)
            emit_progress(
                progress_callback,
                event="epoch_end",
                phase="baseline",
                message=f"Epoch {epoch}/{config.epochs} complete.",
                **_progress_data(
                    event_context,
                    epoch=epoch,
                    total_epochs=config.epochs,
                    train_loss=float(train_loss),
                    val_loss=float(val_loss),
                    best_val_loss=float(best_val),
                    best_epoch=int(best_epoch),
                    epochs_completed=int(len(history)),
                    elapsed_seconds=float(elapsed_seconds),
                    eta_seconds=float(eta_seconds),
                ),
            )
            if improved:
                emit_progress(
                    progress_callback,
                    event="checkpoint_updated",
                    phase="baseline",
                    message=f"Validation loss improved at epoch {epoch}.",
                    **_progress_data(
                        event_context,
                        epoch=epoch,
                        best_epoch=int(best_epoch),
                        best_val_loss=float(best_val),
                    ),
                )

        runtime_seconds = perf_counter() - start_time
        if best_state is None:
            # This fallback is mostly defensive: it handles degenerate runs that
            # finished without ever recording an improvement.
            best_state = clone_state(model.state_dict())
            if history:
                best_epoch = len(history)
                best_val = float(history[-1]["val_loss"])
        request_stop(should_stop)
        model.load_state_dict(best_state)
        # Stage 5: evaluate the best checkpoint on the requested split.
        evaluation_loader = _select_loader(bundle, evaluation_split)
        # train_frac + val_frac >= 1 leaves this fold empty on purpose: the run
        # scores an External Eval Set instead, so there is nothing to evaluate here
        # and every internal metric, plot and card is simply omitted.
        has_evaluation_split = len(evaluation_loader.dataset) > 0
        evaluation_loss = evaluation_mae = None
        freq_mae: list[float] = []
        per_channel_mae: list[float] = []
        channel_mae_with_units: list[str] = []
        if not has_evaluation_split:
            emit_progress(
                progress_callback,
                event="evaluation_skipped",
                phase="baseline",
                message=(
                    f"No {evaluation_split} rows ("
                    + (
                        "the External Eval Set merged the test fold into training"
                        if merge_test_into_train
                        else "train + validation fractions total 1.0"
                    )
                    + "); skipping the internal evaluation."
                ),
                **_progress_data(event_context, evaluation_split=evaluation_split),
            )
        else:
            evaluation_loss, evaluation_mae, freq_mae, per_channel_mae = _eval_metrics(
                model,
                evaluation_loader,
                device,
                amp,
                should_stop=should_stop,
                target_mean=bundle.target_mean,
                target_std=bundle.target_std,
                channel_transforms=bundle.channel_transforms,
                loss_fn=loss_fn,
            )
            # Build per-channel MAE strings with units for display.
            # Only show per-channel breakdown when channel_units are defined (e.g. CTLE
            # with "dB", "deg").  For touchstone data without units, show one combined
            # average MAE to keep the display clean.
            channel_mae_with_units = _channel_mae_labels(
                bundle.channel_names, bundle.channel_units, per_channel_mae
            )
            emit_progress(
                progress_callback,
                event="evaluation_completed",
                phase="baseline",
                message=f"{evaluation_split.capitalize()} evaluation completed.",
                **_progress_data(
                    event_context,
                    evaluation_split=evaluation_split,
                    evaluation_loss=float(evaluation_loss),
                    average_evaluation_mae=float(evaluation_mae),
                    frequency_ghz=bundle.frequency_ghz.tolist(),
                    frequency_mae=freq_mae,
                    per_channel_mae=per_channel_mae,
                    channel_mae_with_units=channel_mae_with_units,
                ),
            )

        # Score the same best checkpoint on the external evaluation set, through
        # the exact metric path the internal test evaluation uses. A Stop click
        # during this optional add-on must not discard the fully-trained run, so
        # a cancellation here only skips the external numbers and lets the run
        # save its artifacts normally.
        external_result: dict[str, Any] = {}
        if external_eval is not None:
            try:
                external_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(external_eval[0]), torch.from_numpy(external_eval[1])
                    ),
                    batch_size=512,
                )
                external_loss, external_mae, _external_freq_mae, external_per_channel = _eval_metrics(
                    model,
                    external_loader,
                    device,
                    amp,
                    should_stop=should_stop,
                    target_mean=bundle.target_mean,
                    target_std=bundle.target_std,
                    channel_transforms=bundle.channel_transforms,
                    loss_fn=loss_fn,
                )
                # The external set was requested explicitly, so always break the
                # MAE down per channel — even for datasets that declare no units.
                external_labels = _channel_mae_labels(
                    bundle.channel_names, bundle.channel_units, external_per_channel, require_units=False
                )
                external_result = {
                    "eval_dataset_path": str(config.eval_dataset_path),
                    "external_eval_rows": int(len(external_eval[0])),
                    "external_eval_loss": float(external_loss),
                    "average_external_mae": float(external_mae),
                    "external_per_channel_mae": external_per_channel,
                    "external_channel_mae_with_units": external_labels,
                }
                emit_progress(
                    progress_callback,
                    event="external_evaluation_completed",
                    phase="baseline",
                    message=(
                        f"External evaluation ({Path(config.eval_dataset_path).name}, "
                        f"{external_result['external_eval_rows']} rows): "
                        + " | ".join(external_labels)
                    ),
                    **_progress_data(event_context, **external_result),
                )
            except RunCancelled:
                emit_progress(
                    progress_callback,
                    event="external_evaluation_skipped",
                    phase="baseline",
                    message="Stop requested during the external evaluation; skipping it and saving the trained run.",
                    **_progress_data(event_context, eval_dataset_path=str(config.eval_dataset_path)),
                )

        result = {
            "status": "ok",
            "device": str(device),
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val),
            "epochs_completed": len(history),
            "runtime_seconds": float(runtime_seconds),
            "evaluation_split": evaluation_split,
            # The split actually used. config.train_frac records what was REQUESTED,
            # which differs from this whenever the test fold was merged away, so both
            # the effective fraction and the row counts are recorded to keep the run
            # self-describing.
            "effective_train_frac": float(effective_train_frac),
            "test_fold_merged_into_train": bool(merge_test_into_train),
            "train_samples": int(len(bundle.split_indices["train"])),
            "val_samples": int(len(bundle.split_indices["val"])),
            "test_samples": int(len(bundle.split_indices["test"])),
            "history": history,
            "config": asdict(config),
            "active_input_feature_names": bundle.active_names,
            "dropped_input_feature_names": bundle.dropped_names,
        }
        if has_evaluation_split:
            result.update(
                {
                    "evaluation_loss": float(evaluation_loss),
                    "average_evaluation_mae": float(evaluation_mae),
                    "per_channel_mae": per_channel_mae,
                    "channel_mae_with_units": channel_mae_with_units,
                    "frequency_mae": freq_mae,
                }
            )
        result.update(external_result)
        if not save_artifacts:
            # Search trials use this fast path because they only need the metrics, not
            # a full artifact directory.
            emit_progress(
                progress_callback,
                event="completed",
                phase="baseline",
                message="Baseline trial completed.",
                **_progress_data(
                    event_context,
                    evaluation_split=evaluation_split,
                    best_epoch=int(best_epoch),
                    best_val_loss=float(best_val),
                    evaluation_loss=None if evaluation_loss is None else float(evaluation_loss),
                    average_evaluation_mae=None if evaluation_mae is None else float(evaluation_mae),
                    runtime_seconds=float(runtime_seconds),
                ),
            )
            return result

        # Stage 6: persist the checkpoint, raw history, and diagnostic plots for a
        # full training run.
        artifact_dir = Path(run_dir) if run_dir is not None else make_run_dir(config.output_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        history_path = artifact_dir / "history.json"
        summary_path = artifact_dir / "summary.json"
        best_path = artifact_dir / "best_model.pt"
        split_title = evaluation_split.capitalize()

        _save_checkpoint(best_path, model, config, bundle, best_epoch, best_val, effective_train_frac)
        history_path.write_text(json.dumps(history, indent=2))
        _plot_loss(history, artifact_dir / "loss_curve.png")

        artifact_summary = {
            "run_dir": str(artifact_dir.resolve()),
            "loss_curve_path": str((artifact_dir / "loss_curve.png").resolve()),
        }
        # Every fold-specific artifact is skipped when that fold is empty; the loss
        # curve and the checkpoint above are all a no-test-fold run produces.
        if has_evaluation_split:
            frequency_plot_path = artifact_dir / f"{evaluation_split}_frequency_mae.png"
            average_mae_plot_path = artifact_dir / f"average_{evaluation_split}_mae.png"
            _plot_frequency_mae(bundle.frequency_ghz, freq_mae, frequency_plot_path, f"{split_title} MAE by Frequency")
            _plot_average_mae_bars(
                labels=[f"Baseline {evaluation_split}"],
                values=[float(evaluation_mae)],
                path=average_mae_plot_path,
                title=f"Average {split_title} MAE",
                ylabel="Average MAE",
            )
            # Generate test sample prediction-vs-truth plots.
            test_sample_plot_paths, test_sample_data = _generate_test_sample_plots(
                model, bundle, device, amp, artifact_dir, num_samples=4,
            )
            artifact_summary.update(
                {
                    f"{evaluation_split}_loss": float(evaluation_loss),
                    f"average_{evaluation_split}_mae": float(evaluation_mae),
                    f"{evaluation_split}_frequency_mae_plot_path": str(frequency_plot_path.resolve()),
                    f"average_{evaluation_split}_mae_plot_path": str(average_mae_plot_path.resolve()),
                    "test_sample_plot_paths": test_sample_plot_paths,
                    "test_sample_data": test_sample_data,
                }
            )
        result.update(artifact_summary)
        summary_path.write_text(json.dumps(result, indent=2))
        emit_progress(
            progress_callback,
            event="artifacts_saved",
            phase="baseline",
            message="Baseline artifacts have been saved.",
            **_progress_data(
                event_context,
                run_dir=str(artifact_dir.resolve()),
                summary_path=str(summary_path.resolve()),
                best_model_path=str(best_path.resolve()),
                loss_curve_path=result["loss_curve_path"],
                # Absent when the run kept no test fold, so there is no fold plot.
                frequency_mae_plot_path=result.get(f"{evaluation_split}_frequency_mae_plot_path"),
            ),
        )
        emit_progress(
            progress_callback,
            event="completed",
            phase="baseline",
            message="Baseline training completed.",
            **_progress_data(
                event_context,
                run_dir=str(artifact_dir.resolve()),
                best_epoch=int(best_epoch),
                best_val_loss=float(best_val),
                evaluation_split=evaluation_split,
                # None when the run kept no test fold; the External Eval Set numbers
                # are then the only accuracy this event carries.
                evaluation_loss=None if evaluation_loss is None else float(evaluation_loss),
                average_evaluation_mae=None if evaluation_mae is None else float(evaluation_mae),
                runtime_seconds=float(runtime_seconds),
            ),
        )
        return result
    except RunCancelled:
        # Return a structured partial result instead of bubbling the exception out to
        # every caller. This keeps GUI and CLI code much simpler.
        runtime_seconds = perf_counter() - start_time
        stopped_result = {
            "status": "stopped",
            "device": str(device),
            "best_epoch": best_epoch if best_state is not None else None,
            "best_val_loss": float(best_val) if best_state is not None else None,
            "epochs_completed": len(history),
            "runtime_seconds": float(runtime_seconds),
            "evaluation_split": evaluation_split,
            "evaluation_loss": None,
            "average_evaluation_mae": None,
            "frequency_mae": [],
            "history": history,
            "config": asdict(config),
            "active_input_feature_names": bundle.active_names if bundle is not None else [],
            "dropped_input_feature_names": bundle.dropped_names if bundle is not None else [],
        }
        emit_progress(
            progress_callback,
            event="stopped",
            phase="baseline",
            message="Baseline training stopped.",
            **_progress_data(
                event_context,
                epochs_completed=int(len(history)),
                best_epoch=stopped_result["best_epoch"],
                best_val_loss=stopped_result["best_val_loss"],
                runtime_seconds=float(runtime_seconds),
            ),
        )
        return stopped_result


def train_baseline(
    config: TrainConfig,
    show_progress: bool = True,
    progress_callback: ProgressCallback | None = None,
    should_stop: StopChecker | None = None,
) -> dict[str, Any]:
    """Train one full-spectrum baseline model and save the best checkpoint."""
    # `run_baseline_trial` already contains the full implementation. This wrapper
    # simply fixes the evaluation split and returns a smaller top-level summary.
    result = run_baseline_trial(
        config=config,
        show_progress=show_progress,
        save_artifacts=True,
        evaluation_split="test",
        progress_callback=progress_callback,
        should_stop=should_stop,
    )
    # A cancelled run returns a partial result (status "stopped") that lacks the
    # completed-run keys (run_dir, test_loss, plot paths). Pass it straight through
    # so callers see the stopped status instead of a KeyError on those keys.
    if result.get("status") == "stopped":
        return result
    summary = {
        "run_dir": result["run_dir"],
        "device": result["device"],
        "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "active_input_feature_names": result["active_input_feature_names"],
        "dropped_input_feature_names": result["dropped_input_feature_names"],
        "loss_curve_path": result["loss_curve_path"],
        "effective_train_frac": result["effective_train_frac"],
        "test_fold_merged_into_train": result["test_fold_merged_into_train"],
        "train_samples": result["train_samples"],
        "val_samples": result["val_samples"],
        "test_samples": result["test_samples"],
    }
    # Absent when train_frac + val_frac left no test rows: the External Eval Set
    # is then the only accuracy number the run produces.
    for key in (
        "test_loss",
        "average_test_mae",
        "test_frequency_mae_plot_path",
        "average_test_mae_plot_path",
    ):
        if key in result:
            summary[key] = result[key]
    if "channel_mae_with_units" in result:
        summary["channel_mae_with_units"] = result["channel_mae_with_units"]
    if "per_channel_mae" in result:
        summary["per_channel_mae"] = result["per_channel_mae"]
    if "test_sample_data" in result:
        summary["test_sample_data"] = result["test_sample_data"]
    if "test_sample_plot_paths" in result:
        summary["test_sample_plot_paths"] = result["test_sample_plot_paths"]
    # External evaluation-set metrics, present only when the run configured one.
    for key in (
        "eval_dataset_path",
        "external_eval_rows",
        "external_eval_loss",
        "average_external_mae",
        "external_per_channel_mae",
        "external_channel_mae_with_units",
    ):
        if key in result:
            summary[key] = result[key]
    # Pass through history and runtime for GUI display.
    summary["history"] = result.get("history", [])
    summary["runtime_seconds"] = result.get("runtime_seconds")
    return summary


#
# Self-transfer flow
# -----------------------------------------------------------------------------
def run_self_transfer(
    config: TransferConfig,
    show_progress: bool = True,
    progress_callback: ProgressCallback | None = None,
    should_stop: StopChecker | None = None,
) -> dict[str, Any]:
    """Run the notebook-style forward/backward transfer experiment."""
    start_time = perf_counter()
    emit_progress(
        progress_callback,
        event="started",
        phase="transfer",
        message="Frequency-domain self-transfer started.",
        config=asdict(config),
    )
    seed_all(config.seed)
    enable_fast_cuda()
    device = resolve_device(config.device)
    limit_cpu_threads_for_gpu(device)
    amp = config.use_amp and device.type == "cuda"
    run_dir = make_run_dir(config.output_dir)
    results: list[dict[str, Any]] = []
    original_frequency_count: int | None = None
    effective_frequency_count: int | None = None
    trimmed_frequency_count: int | None = None
    try:
        request_stop(should_stop)
        # Standalone (no baseline run): load the cache and compute the split +
        # normalization from the training split, exactly as baseline training does.
        with np.load(config.cache_path, allow_pickle=False) as data:
            features = data["features"].astype(np.float32)
            targets = data["targets"].astype(np.float32)
            input_feature_names = (
                data["input_feature_names"].astype(str).tolist()
                if "input_feature_names" in data
                else data["feature_names"].astype(str).tolist()
            )
            frequency_hz = data["frequency_hz"].astype(np.float32)
            channel_names = (
                data["channel_names"].astype(str).tolist()
                if "channel_names" in data
                else [f"ch{i}" for i in range(targets.shape[1])]
            )
            channel_transforms = (
                data["channel_transforms"].astype(str).tolist()
                if "channel_transforms" in data
                else [""] * targets.shape[1]
            )
            channel_units = (
                data["channel_units"].astype(str).tolist()
                if "channel_units" in data
                else [""] * targets.shape[1]
            )
            sweep_label = str(data["sweep_label"]) if "sweep_label" in data else "Frequency (GHz)"

        if config.split_corner_columns or config.split_design_columns:
            # Design-level split (matches baseline): every corner row of a design
            # stays in the same fold so no design leaks between train and test.
            split = design_split_indices(
                features,
                input_feature_names,
                config.split_corner_columns,
                config.train_frac,
                config.val_frac,
                config.seed,
                config.split_design_columns,
            )
        else:
            split = split_indices(len(features), config.train_frac, config.val_frac, config.seed)
        if len(split["test"]) == 0:
            # Self-transfer reports accuracy only on the internal test fold and has no
            # External Eval Set option, so an empty fold would yield NaN metrics.
            raise ValueError(
                "Self-transfer reports accuracy on the internal test fold, so it needs "
                "one, but train_frac + val_frac leaves no test rows. Lower the train or "
                "validation fraction for the transfer run."
            )
        # Drop constant input features using the training split only (matches baseline).
        train_feats = features[split["train"]]
        active_mask = train_feats.max(axis=0) != train_feats.min(axis=0)
        if not active_mask.any():
            active_mask[:] = True
        active_names = [n for n, keep in zip(input_feature_names, active_mask, strict=True) if keep]
        dropped_names = [n for n, keep in zip(input_feature_names, active_mask, strict=True) if not keep]
        active_idx = [i for i, keep in enumerate(active_mask) if keep]
        # Input normalization from the training split.
        input_feature_mean = train_feats[:, active_idx].mean(axis=0).astype(np.float32)
        input_feature_std = train_feats[:, active_idx].std(axis=0).astype(np.float32)
        input_feature_std[input_feature_std == 0] = 1.0
        x = (features[:, active_idx] - input_feature_mean) / input_feature_std
        train_x, test_x = x[split["train"]], x[split["test"]]
        train_y, test_y = targets[split["train"]], targets[split["test"]]
        # Per-channel, per-frequency target normalization from the training split, with a
        # per-channel std floor — identical to baseline training so MAE is comparable.
        target_mean = train_y.mean(axis=0).astype(np.float32)
        target_std = train_y.std(axis=0).astype(np.float32)
        for ch in range(target_std.shape[0]):
            ch_global_std = float(train_y[:, ch, :].std())
            floor = max(ch_global_std * 0.01, 1e-30)
            target_std[ch][target_std[ch] < floor] = floor
        # Match load_split_bundle: only a frequency sweep axis is stored in Hz and
        # displayed in GHz; other sweep axes (e.g. a CTLE VDIFF sweep) are used as-is.
        is_frequency = "freq" in sweep_label.lower()
        freq_ghz = (frequency_hz / 1.0e9) if is_frequency else frequency_hz
        original_frequency_count = int(len(freq_ghz))
        bands = _band_indices(original_frequency_count, config.num_bands)
        effective_frequency_count = int(sum(len(band) for band in bands))
        trimmed_frequency_count = int(original_frequency_count - effective_frequency_count)
        if trimmed_frequency_count > 0:
            # Transfer learning uses equal-width bands, so trim the trailing remainder
            # when the frequency count is not perfectly divisible by the requested
            # number of bands.
            train_y = train_y[:, :, :effective_frequency_count]
            test_y = test_y[:, :, :effective_frequency_count]
            target_mean = target_mean[:, :effective_frequency_count]
            target_std = target_std[:, :effective_frequency_count]
            freq_ghz = freq_ghz[:effective_frequency_count]
            transfer_data_message = (
                "Transfer-learning inputs are ready. "
                f"Trimming the last {trimmed_frequency_count} frequency point(s) so "
                f"{config.num_bands} equal bands can be formed."
            )
        else:
            transfer_data_message = "Transfer-learning inputs are ready."
        emit_progress(
            progress_callback,
            event="data_ready",
            phase="transfer",
            message=transfer_data_message,
            train_samples=int(len(train_x)),
            test_samples=int(len(test_x)),
            frequency_count=original_frequency_count,
            effective_frequency_count=effective_frequency_count,
            trimmed_frequency_count=trimmed_frequency_count,
            frequency_min_ghz=float(freq_ghz.min()),
            frequency_max_ghz=float(freq_ghz.max()),
        )

        # Band-model architecture comes from the transfer config (standalone — no baseline).
        saved_model_type = config.model_type
        model_kwargs = {
            "input_feature_dim": len(active_names),
            "ground_truth_channels": int(targets.shape[1]),
            "width": int(config.width),
            "depth": int(config.depth),
        }
        # SpectraHydraProj: resolve the corner columns once, here, so every band
        # submodel, the stitch evaluator, and the saved final_submodels.pt share
        # byte-identical projection kwargs (band-to-band warm starts load state
        # dicts strictly, so all band architectures must match exactly).
        model_kwargs.update(
            resolve_projection_kwargs(
                saved_model_type,
                config.projection_columns,
                config.projection_dim,
                active_names,
                dropped_names,
            )
        )
        weight_decay = 1e-4 if config.weight_decay is None else float(config.weight_decay)
        # Each band predicts only its own frequencies, so band models are built at the
        # band width (the per-frequency output layer is what differs between bands).
        band_width = int(len(bands[0]))
        # Train in normalized target space; MAE is denormalized for reports.
        train_y_norm = (train_y - target_mean) / target_std
        loaders = _band_loaders(train_x, train_y_norm, bands, config.batch_size, device.type == "cuda")
        eval_model = build_model(saved_model_type, num_frequencies=band_width, **model_kwargs).to(device)
        # `states[i]` will hold the latest checkpoint for band `i`.
        states: list[dict[str, torch.Tensor]] = [None] * config.num_bands  # type: ignore[list-item]
        total_band_runs = 1 + config.iterations * max(2 * (config.num_bands - 1), 0)
        band_run_index = 1
        # Initialize the first band by training a fresh (randomly initialized) model on
        # its frequency slice — no baseline seeding.
        states[0] = _train_band(
            model_kwargs,
            None,
            loaders[0],
            device,
            amp,
            config.transfer_epochs,
            config.learning_rate,
            weight_decay,
            show_progress=show_progress,
            progress_callback=progress_callback,
            should_stop=should_stop,
            event_context={
                "transfer_iteration": 0,
                "direction": "initial",
                "band_index": 0,
                "num_bands": config.num_bands,
                "band_run_index": band_run_index,
                "total_band_runs": total_band_runs,
            },
            run_start_time=start_time,
            model_type=saved_model_type,
            num_frequencies=band_width,
        )

        for t in range(1, config.iterations + 1):
            request_stop(should_stop)
            emit_progress(
                progress_callback,
                event="iteration_started",
                phase="transfer",
                message=f"Transfer iteration {t} started.",
                transfer_iteration=t,
                total_iterations=config.iterations,
            )
            if show_progress:
                print(f"T={t}: forward loop")
            # Forward pass: each band starts from the checkpoint produced by the band
            # immediately below it in frequency.
            for i in range(1, config.num_bands):
                band_run_index += 1
                states[i] = _train_band(
                    model_kwargs,
                    states[i - 1],
                    loaders[i],
                    device,
                    amp,
                    config.transfer_epochs,
                    config.learning_rate,
                    weight_decay,
                    show_progress=show_progress,
                    progress_callback=progress_callback,
                    should_stop=should_stop,
                    event_context={
                        "transfer_iteration": t,
                        "direction": "forward",
                        "band_index": i,
                        "num_bands": config.num_bands,
                        "band_run_index": band_run_index,
                        "total_band_runs": total_band_runs,
                    },
                    run_start_time=start_time,
                    model_type=saved_model_type,
                    num_frequencies=band_width,
                )
            if show_progress:
                print(f"T={t}: backward loop")
            # Backward pass: sweep back across the bands so information can propagate
            # in the opposite direction as well.
            for i in range(config.num_bands - 2, -1, -1):
                band_run_index += 1
                states[i] = _train_band(
                    model_kwargs,
                    states[i + 1],
                    loaders[i],
                    device,
                    amp,
                    config.transfer_epochs,
                    config.learning_rate,
                    weight_decay,
                    show_progress=show_progress,
                    progress_callback=progress_callback,
                    should_stop=should_stop,
                    event_context={
                        "transfer_iteration": t,
                        "direction": "backward",
                        "band_index": i,
                        "num_bands": config.num_bands,
                        "band_run_index": band_run_index,
                        "total_band_runs": total_band_runs,
                    },
                    run_start_time=start_time,
                    model_type=saved_model_type,
                    num_frequencies=band_width,
                )
            # Reassemble the per-band models into one full-spectrum prediction and
            # record its error profile.
            average_mae, freq_mae, band_mae, per_channel_mae = _stitched_mae(
                eval_model, states, bands, test_x, test_y, device, amp, config.batch_size,
                target_mean=target_mean, target_std=target_std, channel_transforms=channel_transforms,
            )
            result = {
                "transfer_iteration": t,
                "average_mae": average_mae,
                "frequency_mae": freq_mae,
                "band_mae": band_mae,
                "per_channel_mae": per_channel_mae,
            }
            results.append(result)
            (run_dir / f"iteration_{t:02d}.json").write_text(json.dumps(result, indent=2))
            elapsed_seconds = perf_counter() - start_time
            progress_fraction = band_run_index / max(total_band_runs, 1) if total_band_runs > 0 else 0.0
            eta_seconds = (
                0.0
                if progress_fraction <= 0.0
                else max(elapsed_seconds * (1.0 - progress_fraction) / progress_fraction, 0.0)
            )
            emit_progress(
                progress_callback,
                event="iteration_completed",
                phase="transfer",
                message=f"Transfer iteration {t} completed.",
                transfer_iteration=t,
                total_iterations=config.iterations,
                average_mae=float(average_mae),
                frequency_mae=freq_mae,
                band_mae=band_mae,
                per_channel_mae=per_channel_mae,
                channel_names=channel_names,
                frequency_ghz=freq_ghz.tolist(),
                elapsed_seconds=float(elapsed_seconds),
                eta_seconds=float(eta_seconds),
            )

        # Persist summary plots that show how the transfer procedure evolved.
        _plot_transfer_curves(
            freq_ghz,
            results,
            run_dir / "mae_vs_frequency_by_iteration.png",
            base_frequency_mae=None,
        )
        _plot_average_curve(results, run_dir / "average_mae_vs_iteration.png", base_average=None)
        _plot_band_curve(results, bands, freq_ghz, run_dir / "band_mae_vs_iteration.png", base_band_mae=None)
        final_average_plot_path = run_dir / "final_average_mae.png"
        final_average = float(results[-1]["average_mae"]) if results else 0.0
        _plot_average_mae_bars(
            labels=[f"Final (T={config.iterations})"],
            values=[final_average],
            path=final_average_plot_path,
            title="Average MAE at End of Transfer",
            ylabel="Average MAE",
        )
        # Save all band submodels plus the normalization + metadata needed to use them
        # standalone (e.g. ONNX export), since there is no baseline checkpoint to read.
        torch.save(
            {
                "states": states,
                "model_kwargs": model_kwargs,
                "bands": bands,
                "model_type": saved_model_type,
                "active_input_feature_names": active_names,
                "target_channel_names": channel_names,
                "channel_transforms": channel_transforms,
                "channel_units": channel_units,
                "input_feature_mean": input_feature_mean,
                "input_feature_std": input_feature_std,
                "target_mean": target_mean,
                "target_std": target_std,
                "frequency_hz": frequency_hz,
            },
            run_dir / "final_submodels.pt",
        )

        summary = {
            "status": "ok",
            "run_dir": str(run_dir.resolve()),
            "channel_names": channel_names,
            "frequency_count": original_frequency_count,
            "effective_frequency_count": effective_frequency_count,
            "trimmed_frequency_count": trimmed_frequency_count,
            "final_average_mae": float(final_average),
            "iteration_results": results,
            "frequency_mae_plot_path": str((run_dir / "mae_vs_frequency_by_iteration.png").resolve()),
            "average_mae_plot_path": str((run_dir / "average_mae_vs_iteration.png").resolve()),
            "band_mae_plot_path": str((run_dir / "band_mae_vs_iteration.png").resolve()),
            "final_average_mae_plot_path": str(final_average_plot_path.resolve()),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        emit_progress(
            progress_callback,
            event="artifacts_saved",
            phase="transfer",
            message="Transfer-learning artifacts have been saved.",
            run_dir=str(run_dir.resolve()),
            summary_path=str((run_dir / "summary.json").resolve()),
            frequency_mae_plot_path=summary["frequency_mae_plot_path"],
            average_mae_plot_path=summary["average_mae_plot_path"],
            band_mae_plot_path=summary["band_mae_plot_path"],
        )
        emit_progress(
            progress_callback,
            event="completed",
            phase="transfer",
            message="Frequency-domain self-transfer completed.",
            run_dir=str(run_dir.resolve()),
            final_average_mae=float(final_average),
            total_iterations=config.iterations,
            elapsed_seconds=float(perf_counter() - start_time),
            eta_seconds=0.0,
        )
        return summary
    except RunCancelled:
        # Save whatever partial information exists so an interrupted transfer run still
        # leaves a useful artifact trail behind.
        summary = {
            "status": "stopped",
            "run_dir": str(run_dir.resolve()),
            "frequency_count": original_frequency_count,
            "effective_frequency_count": effective_frequency_count,
            "trimmed_frequency_count": trimmed_frequency_count,
            "final_average_mae": float(results[-1]["average_mae"]) if results else None,
            "iteration_results": results,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        emit_progress(
            progress_callback,
            event="stopped",
            phase="transfer",
            message="Frequency-domain self-transfer stopped.",
            run_dir=str(run_dir.resolve()),
            completed_iterations=len(results),
            final_average_mae=summary["final_average_mae"],
            elapsed_seconds=float(perf_counter() - start_time),
            eta_seconds=0.0,
        )
        return summary


#
# Core training/evaluation helpers
# -----------------------------------------------------------------------------
def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    should_stop: StopChecker | None = None,
    loss_fn=frequency_rmse,
) -> float:
    model.train()
    # Accumulate on-device and read back once after the loop. A per-batch
    # float(loss.cpu()) blocks the host until every kernel queued for that step has
    # finished, so the CPU can never run ahead to enqueue the next step's kernels --
    # the loop degenerates from pipelined to strictly alternating and the GPU idles
    # through all the Python. The returned value is unchanged.
    total = torch.zeros((), dtype=torch.float32, device=device)
    count = 0
    for x, y in loader:
        request_stop(should_stop)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.detach().float() * x.shape[0]
        count += x.shape[0]
    return float(total) / max(count, 1)


def _eval_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    should_stop: StopChecker | None = None,
    loss_fn=frequency_rmse,
) -> float:
    model.eval()
    # Same deferred read-back as _run_epoch: one host sync per call, not per batch.
    total = torch.zeros((), dtype=torch.float32, device=device)
    count = 0
    with torch.no_grad():
        for x, y in loader:
            request_stop(should_stop)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(x), y)
            total += loss.detach().float() * x.shape[0]
            count += x.shape[0]
    if count == 0:
        # Returning 0.0 here would look like a perfect epoch, so the first
        # checkpoint would never be beaten and training would silently keep
        # epoch-1 weights.
        raise ValueError(
            "The validation split is empty, so no validation loss can be computed. "
            "Lower the train fraction so validation keeps at least one sample."
        )
    return float(total) / max(count, 1)


def _channel_mae_labels(
    channel_names: list[str],
    channel_units: list[str],
    per_channel_mae: list[float],
    require_units: bool = True,
) -> list[str]:
    """Format per-channel MAE values as display strings like ``"gain: 0.1291 dB"``.

    With ``require_units`` (the default), datasets that declare no channel units
    get an empty list so the GUI keeps its single averaged card; pass ``False``
    to always produce the breakdown.
    """
    if require_units and not any(unit for unit in channel_units):
        return []
    labels = []
    for ch_idx, ch_name in enumerate(channel_names):
        unit = channel_units[ch_idx] if ch_idx < len(channel_units) else ""
        mae_val = per_channel_mae[ch_idx] if ch_idx < len(per_channel_mae) else 0.0
        mae_str = f"{mae_val:.4e}" if mae_val != 0 and abs(mae_val) < 1e-3 else f"{mae_val:.4f}"
        labels.append(f"{ch_name}: {mae_str} {unit}".strip())
    return labels


def _load_external_eval(path_str: str, bundle: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load an external evaluation .npz and align it to a training bundle.

    The file must hold ``features`` (rows x columns), ``targets`` (rows x
    channels x frequencies), and ``feature_names`` (or the engine cache's
    ``input_feature_names``) — a dataset bundle.npz or cache.npz qualifies.
    Feature columns are matched to the model's active inputs by name and
    normalized with the training statistics; targets are normalized with the
    training target statistics so ``_eval_metrics`` reports MAE in the original
    units, exactly like the internal test evaluation. Targets must therefore be
    in the same value space as the training dataset's stored targets; when the
    file carries its own channel/frequency metadata it is cross-checked against
    the training bundle so a mismatch cannot score silently. Every problem
    raises ``ValueError`` so callers can fail before training starts. Returns
    ``(x, y, notes)`` where ``notes`` are non-fatal warnings for the run log.
    """
    path = Path(path_str)
    if not path.is_file():
        raise ValueError(f"External evaluation set not found: {path}")
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"Could not read external evaluation set {path}: {error}") from error
    with archive as data:
        names_key = "feature_names" if "feature_names" in data else "input_feature_names"
        missing_keys = [key for key in ("features", "targets", names_key) if key not in data]
        if missing_keys:
            raise ValueError(
                f"External evaluation set {path.name} is missing arrays: {', '.join(missing_keys)} "
                "(expected an .npz holding features, targets, and feature_names "
                "or input_feature_names)."
            )
        features = np.asarray(data["features"], dtype=np.float32)
        targets = np.asarray(data["targets"], dtype=np.float32)
        names = [str(name) for name in data[names_key]]
        # Optional metadata a dataset bundle/cache carries about its own targets.
        file_channel_names = (
            [str(name) for name in data["channel_names"]] if "channel_names" in data else None
        )
        file_channel_transforms = (
            [str(item) for item in data["channel_transforms"]]
            if "channel_transforms" in data
            else None
        )
        file_channel_units = (
            [str(unit) for unit in data["channel_units"]] if "channel_units" in data else None
        )
        file_frequency_hz = (
            np.asarray(data["frequency_hz"], dtype=np.float64) if "frequency_hz" in data else None
        )
    if features.ndim != 2 or targets.ndim != 3:
        raise ValueError(
            f"External evaluation set {path.name} has features of shape {features.shape} and "
            f"targets of shape {targets.shape}; expected (rows, columns) and "
            "(rows, channels, frequencies)."
        )
    if len(features) == 0 or len(features) != len(targets):
        raise ValueError(
            f"External evaluation set {path.name} has {len(features)} feature rows and "
            f"{len(targets)} target rows; both must match and be non-empty."
        )
    if len(names) != features.shape[1]:
        raise ValueError(
            f"External evaluation set {path.name} names {len(names)} feature columns but its "
            f"features array has {features.shape[1]}; the file is malformed."
        )
    missing_features = [name for name in bundle.active_names if name not in names]
    if missing_features:
        raise ValueError(
            f"External evaluation set {path.name} lacks input columns the model needs: "
            f"{', '.join(missing_features)}."
        )
    duplicated = [name for name in bundle.active_names if names.count(name) > 1]
    if duplicated:
        raise ValueError(
            f"External evaluation set {path.name} lists input columns more than once: "
            f"{', '.join(duplicated)}; column matching by name would be ambiguous."
        )
    if targets.shape[1] != len(bundle.channel_names):
        raise ValueError(
            f"External evaluation set {path.name} has {targets.shape[1]} target channels; "
            f"the training dataset has {len(bundle.channel_names)} ({', '.join(bundle.channel_names)})."
        )
    if targets.shape[2] != len(bundle.frequency_hz):
        raise ValueError(
            f"External evaluation set {path.name} has {targets.shape[2]} frequency points; "
            f"the training dataset has {len(bundle.frequency_hz)}."
        )
    # Cross-check the file's own target metadata against the training dataset.
    # These mismatches pass every shape check yet make the score meaningless: a
    # value-space mismatch (log10 vs raw) exponentiates raw values into garbage,
    # swapped channels score each channel against the other's statistics, and a
    # different grid at the same point count compares the wrong frequencies.
    if file_channel_names is not None and file_channel_names != list(bundle.channel_names):
        raise ValueError(
            f"External evaluation set {path.name} declares channels "
            f"[{', '.join(file_channel_names)}] but the training dataset has "
            f"[{', '.join(bundle.channel_names)}]; the order and names must match."
        )
    if file_channel_transforms is not None and file_channel_transforms != list(bundle.channel_transforms):
        raise ValueError(
            f"External evaluation set {path.name} stores targets with channel transforms "
            f"{file_channel_transforms} but the training dataset uses "
            f"{list(bundle.channel_transforms)}; the target value spaces differ, so the "
            "score would be meaningless. Re-export the evaluation set in the training "
            "dataset's target space."
        )
    if file_frequency_hz is not None and (
        file_frequency_hz.shape != np.asarray(bundle.frequency_hz).shape
        or not np.allclose(file_frequency_hz, np.asarray(bundle.frequency_hz, dtype=np.float64), rtol=1e-5)
    ):
        raise ValueError(
            f"External evaluation set {path.name} was sampled on a different frequency grid "
            "than the training dataset; equal point counts are not enough — the grids "
            "must match point-for-point."
        )
    notes: list[str] = []
    if file_channel_units is not None and file_channel_units != list(bundle.channel_units):
        notes.append(
            f"channel units differ from the training dataset ({file_channel_units} vs "
            f"{list(bundle.channel_units)}); results are labeled with the training units."
        )
    # A column that was constant in training was dropped from the model. If it
    # varies in the external set, the model cannot respond to it — the score is
    # still a true measure of this checkpoint, but the reader should know.
    for name in bundle.dropped_names:
        if name in names:
            column = features[:, names.index(name)]
            if np.ptp(column) > 0:
                notes.append(
                    f"column '{name}' was constant in training (dropped from the model) "
                    "but varies in the external set; the model cannot respond to it."
                )
    column_index = [names.index(name) for name in bundle.active_names]
    selected = features[:, column_index]
    if not np.isfinite(selected).all() or not np.isfinite(targets).all():
        bad_rows = int(
            (~np.isfinite(selected).all(axis=1) | ~np.isfinite(targets).all(axis=(1, 2))).sum()
        )
        raise ValueError(
            f"External evaluation set {path.name} contains NaN or infinite values in "
            f"{bad_rows} row(s); clean or drop those rows before using it."
        )
    x = (selected - bundle.input_feature_mean) / bundle.input_feature_std
    y = (targets - bundle.target_mean) / bundle.target_std
    return x.astype(np.float32), y.astype(np.float32), notes


def _eval_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    should_stop: StopChecker | None = None,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
    channel_transforms: list[str] | None = None,
    loss_fn=frequency_rmse,
) -> tuple[float, float, list[float], list[float]]:
    """Evaluate and return (loss, avg_mae, freq_mae, per_channel_mae).

    When ``target_mean`` / ``target_std`` are provided the MAE values are
    computed in the original (denormalized) units.  When ``channel_transforms``
    contains entries like ``"log10"``, the inverse transform (10^x) is applied
    after denormalization so the MAE is in the original data space.
    """
    from .data import _inverse_channel_transforms

    model.eval()
    preds: list[torch.Tensor] = []
    trues: list[torch.Tensor] = []
    # The loss is accumulated inside the prediction pass; this used to call
    # _eval_loss first, walking the whole split a second time for a value the
    # prediction loop can produce for free.
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    count = 0
    with torch.no_grad():
        for x, y in loader:
            request_stop(should_stop)
            x = x.to(device, non_blocking=True)
            y_device = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                pred = model(x)
                batch_loss = loss_fn(pred, y_device)
            total_loss += batch_loss.detach().float() * x.shape[0]
            count += x.shape[0]
            preds.append(pred.cpu())
            trues.append(y)
    loss = float(total_loss) / max(count, 1)
    if not preds:
        raise ValueError("Evaluation split is empty; cannot compute metrics.")
    pred_all = torch.cat(preds, dim=0).numpy()
    true_all = torch.cat(trues, dim=0).numpy()
    # Denormalize predictions and targets to compute MAE in original units.
    if target_mean is not None and target_std is not None:
        pred_all = pred_all * target_std + target_mean
        true_all = true_all * target_std + target_mean
    # Undo log10 or other transforms so MAE is in the original data space.
    if channel_transforms and any(t for t in channel_transforms):
        pred_all = _inverse_channel_transforms(pred_all, channel_transforms)
        true_all = _inverse_channel_transforms(true_all, channel_transforms)
    abs_err = np.abs(pred_all - true_all)
    # Per-channel MAE: average over samples and frequencies for each channel.
    per_channel_mae = abs_err.mean(axis=(0, 2)).tolist()  # (channels,)
    return loss, float(abs_err.mean()), abs_err.mean(axis=(0, 1)).tolist(), per_channel_mae


def _select_loader(bundle: Any, split_name: str) -> DataLoader:
    # Keep the mapping in one place so callers can request a split by name.
    if split_name == "train":
        return bundle.train_loader
    if split_name == "val":
        return bundle.val_loader
    if split_name == "test":
        return bundle.test_loader
    raise ValueError(f"Unknown split name: {split_name}")


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainConfig,
    bundle: Any,
    best_epoch: int,
    best_val: float,
    effective_train_frac: float,
) -> None:
    # Save not only the model weights, but also the normalization statistics and
    # active feature list required to reuse the checkpoint later.
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            # Sibling of "config", never inside it: two call sites rebuild a
            # TrainConfig from that dict with **kwargs and would raise on a new key.
            "effective_train_frac": float(effective_train_frac),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "active_input_feature_names": bundle.active_names,
            "dropped_input_feature_names": bundle.dropped_names,
            "target_channel_names": bundle.channel_names,
            "input_feature_mean": bundle.input_feature_mean,
            "input_feature_std": bundle.input_feature_std,
            "target_mean": bundle.target_mean,
            "target_std": bundle.target_std,
        },
        path,
    )


def _band_indices(num_freq: int, num_bands: int) -> list[np.ndarray]:
    # Transfer learning still assumes equal-width bands, but the last few frequency
    # points can be discarded when the count is not perfectly divisible.
    if num_bands <= 0:
        raise ValueError("Band count must be positive.")
    if num_bands > num_freq:
        raise ValueError("Band count cannot exceed the number of available frequency points.")
    usable_num_freq = num_freq - (num_freq % num_bands)
    if usable_num_freq <= 0:
        raise ValueError("No frequency points remain after trimming for equal-width transfer bands.")
    size = usable_num_freq // num_bands
    return [np.arange(i * size, (i + 1) * size, dtype=np.int64) for i in range(num_bands)]


def _band_loaders(train_x: np.ndarray, train_y: np.ndarray, bands: list[np.ndarray], batch_size: int, pin_memory: bool) -> list[DataLoader]:
    # Build one loader per band so each fine-tuning step only sees its slice of the
    # frequency axis while still sharing the same sample features.
    x = torch.from_numpy(train_x.astype(np.float32))
    loaders = []
    for band in bands:
        y = torch.from_numpy(train_y[:, :, band].astype(np.float32))
        loaders.append(
            DataLoader(
                TensorDataset(x, y),
                batch_size=min(batch_size, len(train_x)),
                shuffle=True,
                num_workers=0,
                pin_memory=pin_memory,
            )
        )
    return loaders


def _train_band(
    model_kwargs: dict[str, Any],
    init_state: dict[str, torch.Tensor] | None,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    epochs: int,
    lr: float,
    weight_decay: float,
    show_progress: bool = True,
    progress_callback: ProgressCallback | None = None,
    should_stop: StopChecker | None = None,
    event_context: dict[str, Any] | None = None,
    run_start_time: float | None = None,
    model_type: str = "SpectraNet",
    num_frequencies: int = 0,
) -> dict[str, torch.Tensor]:
    # Each band trains a model instance. When ``init_state`` is given the model starts
    # from those weights (a neighbour band during the sweep); when it is None the model
    # keeps its fresh random initialization (band 0).
    band_start_time = perf_counter()
    model = build_model(model_type, num_frequencies=num_frequencies, **model_kwargs).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    emit_progress(
        progress_callback,
        event="band_started",
        phase="transfer",
        message="Transfer band training started.",
        **_progress_data(event_context, total_epochs=epochs),
    )
    for epoch in range(1, epochs + 1):
        request_stop(should_stop)
        train_loss = _run_epoch(
            model,
            loader,
            optimizer,
            scaler,
            device,
            amp,
            should_stop=should_stop,
        )
        if show_progress:
            direction = event_context.get("direction", "transfer") if event_context is not None else "transfer"
            band_index = event_context.get("band_index", "?") if event_context is not None else "?"
            print(f"{direction} band={band_index} epoch={epoch:03d} train_loss={train_loss:.6f}")
        elapsed_seconds = perf_counter() - (run_start_time if run_start_time is not None else band_start_time)
        total_band_runs = int(event_context.get("total_band_runs", 0)) if event_context is not None else 0
        band_run_index = int(event_context.get("band_run_index", 0)) if event_context is not None else 0
        if total_band_runs > 0 and band_run_index > 0:
            progress_fraction = ((band_run_index - 1) + epoch / max(epochs, 1)) / max(total_band_runs, 1)
        else:
            progress_fraction = epoch / max(epochs, 1)
        eta_seconds = (
            0.0
            if progress_fraction <= 0.0
            else max(elapsed_seconds * (1.0 - progress_fraction) / progress_fraction, 0.0)
        )
        emit_progress(
            progress_callback,
            event="band_epoch_end",
            phase="transfer",
            message=f"Transfer band epoch {epoch}/{epochs} complete.",
            **_progress_data(
                event_context,
                epoch=epoch,
                total_epochs=epochs,
                train_loss=float(train_loss),
                elapsed_seconds=float(elapsed_seconds),
                eta_seconds=float(eta_seconds),
            ),
        )
    emit_progress(
        progress_callback,
        event="band_completed",
        phase="transfer",
        message="Transfer band training completed.",
        **_progress_data(event_context, total_epochs=epochs),
    )
    return clone_state(model.state_dict())


def _stitched_mae(
    model: nn.Module,
    states: list[dict[str, torch.Tensor]],
    bands: list[np.ndarray],
    test_x: np.ndarray,
    test_y: np.ndarray,
    device: torch.device,
    amp: bool,
    batch_size: int,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
    channel_transforms: list[str] | None = None,
) -> tuple[float, list[float], list[float], list[float]]:
    # Evaluate each band-specific state on its own frequency slice, then stitch the
    # predictions back together into a full-spectrum tensor.
    #
    # Band models predict in the normalized target space. When the normalization stats
    # are supplied, predictions are denormalized per band so the MAE is reported in the
    # original (raw) units of ``test_y`` — matching the baseline's MAE calculation,
    # including per-channel values (e.g. gain and phase reported separately).
    from .data import _inverse_channel_transforms

    pred = np.zeros_like(test_y, dtype=np.float32)
    x = torch.from_numpy(test_x.astype(np.float32))
    for state, band in zip(states, bands, strict=True):
        model.load_state_dict(state)
        model.eval()
        for start in range(0, len(test_x), batch_size):
            stop = min(start + batch_size, len(test_x))
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    out = model(x[start:stop].to(device))
            out_np = out.detach().cpu().numpy().astype(np.float32)
            if target_mean is not None and target_std is not None:
                out_np = out_np * target_std[:, band][np.newaxis, :, :] + target_mean[:, band][np.newaxis, :, :]
            pred[start:stop, :, band] = out_np
    true = test_y
    # Undo per-channel transforms (e.g. log10) so MAE is in the original data space,
    # exactly as the baseline evaluation does.
    if channel_transforms and any(t for t in channel_transforms):
        pred = _inverse_channel_transforms(pred, channel_transforms)
        true = _inverse_channel_transforms(test_y, channel_transforms)
    abs_err = np.abs(pred - true)
    # per_channel_mae averages over samples and frequencies for each channel — same
    # formula as the baseline's _eval_metrics (axis=(0, 2)).
    per_channel_mae = abs_err.mean(axis=(0, 2)).tolist()
    return (
        float(abs_err.mean()),
        abs_err.mean(axis=(0, 1)).tolist(),
        [float(abs_err[:, :, band].mean()) for band in bands],
        per_channel_mae,
    )


#
# Miscellaneous utilities
# -----------------------------------------------------------------------------
def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_fast_cuda() -> None:
    """Enable TF32 on supported CUDA hardware."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Detach a state dict so later updates do not mutate the source tensors."""
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def open_in_vscode(paths: list[str | Path]) -> bool:
    """Open generated artifact files in VS Code when the CLI is available."""
    code_path = shutil.which("code")
    if code_path is None:
        return False
    resolved_paths = [str(Path(path).resolve()) for path in paths if Path(path).exists()]
    if not resolved_paths:
        return False
    subprocess.run([code_path, "-r", *resolved_paths], check=False)
    return True


def make_run_dir(root: str | Path) -> Path:
    """Create one timestamped artifact folder."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Timestamps have second resolution; suffix on collision so two runs started
    # within the same second cannot share (and overwrite) one artifact folder.
    for attempt in range(1000):
        path = Path(root) / (stamp if attempt == 0 else f"{stamp}-{attempt + 1}")
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a unique run directory under {root}.")


#
# Plot helpers
# -----------------------------------------------------------------------------
def _plot_loss(history: list[dict[str, float]], path: Path) -> None:
    # Use a local import so environments that only train without plotting do not
    # import matplotlib until it is actually needed.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train", linewidth=2)
    ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_frequency_mae(freq_ghz: np.ndarray, mae: list[float], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freq_ghz, mae, linewidth=2)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("MAE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_average_mae_bars(
    labels: list[str],
    values: list[float],
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = ["#355070", "#6d597a", "#b56576", "#e56b6f"][: len(values)]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.6f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_transfer_curves(
    freq_ghz: np.ndarray,
    results: list[dict[str, Any]],
    path: Path,
    base_frequency_mae: list[float] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    if base_frequency_mae is not None:
        ax.plot(freq_ghz, base_frequency_mae, linewidth=2.2, linestyle="--", color="black", label="Base")
    for result in results:
        ax.plot(freq_ghz, result["frequency_mae"], linewidth=1.8, label=f"T={result['transfer_iteration']}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("MAE")
    ax.set_title("MAE vs Frequency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_average_curve(results: list[dict[str, Any]], path: Path, base_average: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [r["transfer_iteration"] for r in results]
    mae = [r["average_mae"] for r in results]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(t, mae, marker="o", linewidth=2)
    if base_average is not None and t:
        ax.axhline(base_average, linestyle="--", linewidth=1.8, color="black", label=f"Base = {base_average:.6f}")
    ax.set_xlabel("Transfer Iteration T")
    ax.set_ylabel("Average MAE")
    ax.set_title("Average MAE vs T")
    ax.set_xticks(t)
    ax.grid(True, alpha=0.3)
    if base_average is not None and t:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_band_curve(
    results: list[dict[str, Any]],
    bands: list[np.ndarray],
    freq_ghz: np.ndarray,
    path: Path,
    base_band_mae: list[float] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{freq_ghz[band[0]]:.0f}-{freq_ghz[band[-1]]:.0f}" for band in bands]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5))
    series_count = len(results) + (1 if base_band_mae is not None else 0)
    width = 0.8 / max(series_count, 1)
    start_idx = 0
    if base_band_mae is not None:
        ax.bar(x - 0.4 + width / 2, base_band_mae, width=width, label="Base")
        start_idx = 1
    for idx, result in enumerate(results):
        ax.bar(
            x - 0.4 + width / 2 + (idx + start_idx) * width,
            result["band_mae"],
            width=width,
            label=f"T={result['transfer_iteration']}",
        )
    ax.set_xlabel("Frequency Band (GHz)")
    ax.set_ylabel("Band MAE")
    ax.set_title("Band MAE vs T")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _generate_test_sample_plots(
    model: nn.Module,
    bundle: Any,
    device: torch.device,
    amp: bool,
    artifact_dir: Path,
    num_samples: int = 4,
) -> tuple[list[str], dict]:
    """Generate prediction-vs-truth plots for a few test samples.

    Returns (plot_file_paths, test_sample_data) where test_sample_data contains
    the raw arrays for GUI display.
    """
    model.eval()
    from .data import _inverse_channel_transforms

    loader = bundle.test_loader
    if len(loader.dataset) == 0:
        # No test fold: nothing to plot, and torch.cat would raise on empty lists.
        return [], {}
    preds_list: list[torch.Tensor] = []
    trues_list: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            x_dev = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                pred = model(x_dev)
            preds_list.append(pred.cpu())
            trues_list.append(y)

    pred_all = torch.cat(preds_list, dim=0).numpy()  # (N, C, F)
    true_all = torch.cat(trues_list, dim=0).numpy()

    # Denormalize.
    if bundle.target_mean is not None and bundle.target_std is not None:
        pred_all = pred_all * bundle.target_std + bundle.target_mean
        true_all = true_all * bundle.target_std + bundle.target_mean
    if bundle.channel_transforms and any(t for t in bundle.channel_transforms):
        pred_all = _inverse_channel_transforms(pred_all, bundle.channel_transforms)
        true_all = _inverse_channel_transforms(true_all, bundle.channel_transforms)

    freq_ghz = bundle.frequency_ghz
    channel_names = bundle.channel_names

    # Pick samples evenly spaced through the test set.
    n_test = pred_all.shape[0]
    indices = np.linspace(0, n_test - 1, min(num_samples, n_test), dtype=int)

    plot_dir = artifact_dir / "test_sample_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    for sample_idx in indices:
        path = plot_dir / f"test_sample_{int(sample_idx):04d}.png"
        _plot_test_sample(
            freq_ghz=freq_ghz,
            pred=pred_all[sample_idx],
            true=true_all[sample_idx],
            channel_names=channel_names,
            sample_idx=int(sample_idx),
            path=path,
        )
        saved_paths.append(str(path.resolve()))

    # Build data dict for GUI display.
    test_sample_data = {
        "freq_ghz": freq_ghz.tolist(),
        "channel_names": list(channel_names),
        "samples": [
            {
                "index": int(idx),
                "pred": pred_all[idx].tolist(),   # [C][F]
                "true": true_all[idx].tolist(),
            }
            for idx in indices
        ],
    }

    return saved_paths, test_sample_data


def _plot_test_sample(
    freq_ghz: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    channel_names: list[str],
    sample_idx: int,
    path: Path,
) -> None:
    """Plot predicted vs true curves for one test sample across all channels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_channels = pred.shape[0]
    ncols = min(4, num_channels)
    nrows = (num_channels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for ch_idx in range(num_channels):
        row, col = divmod(ch_idx, ncols)
        ax = axes[row][col]
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx}"
        ax.plot(freq_ghz, true[ch_idx], label="True", linewidth=1.2, color="#2196F3")
        ax.plot(freq_ghz, pred[ch_idx], label="Pred", linewidth=1.2, color="#FF5722", linestyle="--")
        ax.set_title(ch_name, fontsize=10)
        ax.set_xlabel("Freq (GHz)", fontsize=8)
        ax.grid(True, alpha=0.3)
        if ch_idx == 0:
            ax.legend(fontsize=8)

    for ch_idx in range(num_channels, nrows * ncols):
        row, col = divmod(ch_idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"Test Sample #{sample_idx} — Predicted vs True", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
