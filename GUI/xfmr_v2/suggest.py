"""Scan-only heuristic suggestions for baseline and transfer training settings.

This module answers the question:
"Given only the cached dataset statistics, what settings are reasonable starters?"

It does not train any models. Instead it:
- prepares or reuses the cache
- computes dataset diagnostics from the training split
- translates those diagnostics into heuristic baseline and transfer settings

That makes it a fast first pass before a real training run or quick search.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from .data import CACHE_PATH, DATA_ROOT, ensure_cache, split_indices
from .progress import ProgressCallback, StopChecker, emit_progress, request_stop
from .runner import TrainConfig


@dataclass
class SuggestConfig:
    """Configuration for the scan-only initial setting recommender."""

    # Data-location fields mirror the training/search configs for consistency.
    data_root: str | None = str(DATA_ROOT)
    input_feature_path: str | None = None
    ground_truth_data_dir: str | None = None
    cache_path: str = str(CACHE_PATH)
    seed: int = 42
    train_frac: float = 0.8
    val_frac: float = 0.1
    max_samples: int | None = None
    variance_threshold: float = 0.95


def suggest_initial_settings(
    config: SuggestConfig,
    show_progress: bool = False,
    progress_callback: ProgressCallback | None = None,
    should_stop: StopChecker | None = None,
) -> dict[str, Any]:
    """Suggest strong starting hyperparameters without running any training."""
    _validate_suggest_config(config)
    emit_progress(
        progress_callback,
        event="started",
        phase="suggest",
        message="Preparing cached arrays for initial-setting suggestions.",
        config=asdict(config),
    )
    if show_progress:
        print("Preparing cached arrays for initial-setting suggestions...")

    # Reuse the exact same cache-building logic that training uses. That ensures
    # the suggestion pass and the actual training run see the same data layout.
    cache_path = ensure_cache(
        data_root=config.data_root,
        cache_path=config.cache_path,
        max_samples=config.max_samples,
        input_feature_path=config.input_feature_path,
        ground_truth_data_dir=config.ground_truth_data_dir,
    )
    emit_progress(
        progress_callback,
        event="cache_ready",
        phase="suggest",
        message="Cached arrays are ready for suggestion generation.",
        cache_path=str(cache_path),
    )
    request_stop(should_stop)

    with np.load(cache_path, allow_pickle=False) as data:
        features = data["features"].astype(np.float32)
        targets = data["targets"].astype(np.float32)
        frequency_hz = data["frequency_hz"].astype(np.float32)
        input_feature_names = data["input_feature_names"].astype(str).tolist()
        channel_names = data["channel_names"].astype(str).tolist()
        target_names = data["target_names"].astype(str).tolist()

    if config.max_samples is not None:
        features = features[: config.max_samples]
        targets = targets[: config.max_samples]

    # All statistics are computed from the training split only so the heuristic mirrors
    # the information a real training pipeline should legitimately use.
    split = split_indices(len(features), config.train_frac, config.val_frac, config.seed)
    train_features_raw = features[split["train"]]
    active_mask = train_features_raw.std(axis=0) > 1e-8
    if not np.any(active_mask):
        raise ValueError("All input-feature columns are constant in the training split.")

    active_names = [name for name, keep in zip(input_feature_names, active_mask, strict=True) if keep]
    dropped_names = [name for name, keep in zip(input_feature_names, active_mask, strict=True) if not keep]
    input_feature_mean = train_features_raw[:, active_mask].mean(axis=0).astype(np.float32)
    input_feature_std = train_features_raw[:, active_mask].std(axis=0).astype(np.float32)
    input_feature_std[input_feature_std < 1e-8] = 1.0

    train_features = (train_features_raw[:, active_mask] - input_feature_mean) / input_feature_std
    train_targets = targets[split["train"]]

    # Flatten `(samples, channels, frequency)` into one matrix when estimating overall
    # output complexity with linear-algebra diagnostics such as effective rank.
    ground_truth_matrix = train_targets.reshape(train_targets.shape[0], -1)
    frequency_ghz = frequency_hz / 1.0e9

    if show_progress:
        print("Computing dataset diagnostics...")
    emit_progress(
        progress_callback,
        event="diagnostics_started",
        phase="suggest",
        message="Computing dataset diagnostics from the training split.",
    )
    input_rank = _effective_rank(train_features, config.variance_threshold)
    ground_truth_rank = _effective_rank(ground_truth_matrix, config.variance_threshold)
    spectral_stats = _spectral_complexity(train_targets)
    sample_to_ground_truth_rank_ratio = float(len(split["train"]) / max(ground_truth_rank, 1))
    capacity_tier, overfit_risk = _capacity_and_risk(sample_to_ground_truth_rank_ratio)
    gpu_info = _detect_gpu()
    request_stop(should_stop)

    if show_progress:
        print("Building heuristic recommendations...")
    emit_progress(
        progress_callback,
        event="recommendation_started",
        phase="suggest",
        message="Building heuristic baseline and transfer recommendations.",
    )
    baseline = _suggest_baseline(
        request=config,
        cache_path=str(cache_path),
        train_count=len(split["train"]),
        active_input_feature_dim=len(active_names),
        ground_truth_rank=ground_truth_rank,
        frequency_point_count=int(train_targets.shape[-1]),
        spectral_tier=spectral_stats["tier"],
        capacity_tier=capacity_tier,
        overfit_risk=overfit_risk,
        gpu_info=gpu_info,
    )
    transfer = _suggest_transfer(
        frequency_point_count=int(train_targets.shape[-1]),
        spectral_tier=spectral_stats["tier"],
        baseline_learning_rate=float(baseline["config"]["learning_rate"]),
        baseline_batch_size=int(baseline["config"]["batch_size"]),
        gpu_info=gpu_info,
    )

    confidence, confidence_reason = _confidence_summary(
        train_count=len(split["train"]),
        ground_truth_rank=ground_truth_rank,
        flattened_ground_truth_dim=int(ground_truth_matrix.shape[1]),
        sample_to_ground_truth_rank_ratio=sample_to_ground_truth_rank_ratio,
    )

    diagnostics = {
        "cache_path": str(cache_path),
        "num_samples_total": int(len(features)),
        "num_samples_train": int(len(split["train"])),
        "num_samples_val": int(len(split["val"])),
        "num_samples_test": int(len(split["test"])),
        "active_input_feature_dim": int(len(active_names)),
        "dropped_constant_input_count": int(len(dropped_names)),
        "input_feature_effective_rank_95": int(input_rank),
        "ground_truth_effective_rank_95": int(ground_truth_rank),
        "ground_truth_channels": int(len(channel_names)),
        "target_names": target_names,
        "channel_names": channel_names,
        "frequency_point_count": int(train_targets.shape[-1]),
        "frequency_min_ghz": float(frequency_ghz.min()),
        "frequency_max_ghz": float(frequency_ghz.max()),
        "spectral_complexity_tier": spectral_stats["tier"],
        "spectral_complexity_score": float(spectral_stats["score"]),
        "spectral_variation_ratio": float(spectral_stats["variation_ratio"]),
        "spectral_curvature_ratio": float(spectral_stats["curvature_ratio"]),
        "sample_to_ground_truth_rank_ratio": sample_to_ground_truth_rank_ratio,
        "capacity_tier": capacity_tier,
        "estimated_overfit_risk": overfit_risk,
        "gpu_available": bool(gpu_info["available"]),
        "gpu_name": gpu_info["name"],
        "gpu_memory_gb": gpu_info["memory_gb"],
    }
    emit_progress(
        progress_callback,
        event="diagnostics_ready",
        phase="suggest",
        message="Dataset diagnostics are ready.",
        diagnostics=diagnostics,
    )
    warnings = _build_warnings(diagnostics, confidence, transfer)

    result = {
        "status": "ok",
        "scan_mode": "no_training",
        "confidence": confidence,
        "confidence_reason": confidence_reason,
        "diagnostics": diagnostics,
        "active_input_feature_names": active_names,
        "dropped_input_feature_names": dropped_names,
        "suggested_baseline_config": baseline["config"],
        "suggested_baseline_ranges": baseline["ranges"],
        "baseline_rationale": baseline["rationale"],
        "suggested_transfer_config": transfer["config"],
        "suggested_transfer_ranges": transfer["ranges"],
        "transfer_rationale": transfer["rationale"],
        "quick_search_hints": baseline["quick_search_hints"],
        "warnings": warnings,
    }
    emit_progress(
        progress_callback,
        event="completed",
        phase="suggest",
        message="Initial setting suggestion is ready.",
        confidence=confidence,
        confidence_reason=confidence_reason,
        diagnostics=diagnostics,
        suggested_baseline_config=result["suggested_baseline_config"],
        suggested_transfer_config=result["suggested_transfer_config"],
    )
    return result


def _validate_suggest_config(config: SuggestConfig) -> None:
    # These checks are cheap and save the user from confusing downstream errors.
    if not 0.0 < config.train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0.0 <= config.val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")
    if config.train_frac + config.val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must leave room for a test split.")
    if not 0.5 < config.variance_threshold < 1.0:
        raise ValueError("variance_threshold must be between 0.5 and 1.0.")


def _effective_rank(matrix: np.ndarray, variance_threshold: float) -> int:
    # Effective rank answers: "How many singular directions explain most of the variance?"
    # It is a compact proxy for how much independent structure the data seems to contain.
    if matrix.ndim != 2 or matrix.shape[0] <= 1 or matrix.shape[1] == 0:
        return 1
    centered = matrix.astype(np.float64, copy=False) - matrix.mean(axis=0, keepdims=True, dtype=np.float64)
    singular_values = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    energy = singular_values**2
    total_energy = float(energy.sum())
    if total_energy <= 1e-12:
        return 1
    cumulative = np.cumsum(energy) / total_energy
    return int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)


def _spectral_complexity(targets: np.ndarray) -> dict[str, float | str]:
    # View every output channel for every sample as one curve over frequency, then
    # measure how wiggly those curves are using first and second differences.
    curves = targets.reshape(-1, targets.shape[-1]).astype(np.float64, copy=False)
    if curves.shape[-1] < 3:
        return {"variation_ratio": 0.0, "curvature_ratio": 0.0, "score": 0.0, "tier": "low"}

    first_diff = np.diff(curves, axis=1)
    second_diff = np.diff(curves, n=2, axis=1)
    curve_range = np.ptp(curves, axis=1)
    curve_range[curve_range < 1e-8] = 1.0
    total_variation = np.sum(np.abs(first_diff), axis=1)
    total_variation[total_variation < 1e-8] = 1.0

    variation_ratio = float(np.median(total_variation / curve_range))
    curvature_ratio = float(np.median(np.sum(np.abs(second_diff), axis=1) / total_variation))
    score = variation_ratio + 1.5 * curvature_ratio

    if score < 1.6 and curvature_ratio < 0.35:
        tier = "low"
    elif score < 3.2 and curvature_ratio < 0.9:
        tier = "medium"
    else:
        tier = "high"
    return {
        "variation_ratio": variation_ratio,
        "curvature_ratio": curvature_ratio,
        "score": float(score),
        "tier": tier,
    }


def _capacity_and_risk(sample_to_ground_truth_rank_ratio: float) -> tuple[str, str]:
    # When the dataset has many samples relative to output complexity, we can afford
    # a more expressive model. When that ratio is low, we should be more conservative.
    if sample_to_ground_truth_rank_ratio < 8.0:
        return "conservative", "high"
    if sample_to_ground_truth_rank_ratio < 25.0:
        return "balanced", "medium"
    return "aggressive", "low"


def _detect_gpu() -> dict[str, Any]:
    # The heuristic uses GPU availability only to size batch recommendations and AMP.
    if not torch.cuda.is_available():
        return {"available": False, "name": None, "memory_gb": None}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "memory_gb": round(props.total_memory / (1024**3), 1),
    }


def _suggest_baseline(
    request: SuggestConfig,
    cache_path: str,
    train_count: int,
    active_input_feature_dim: int,
    ground_truth_rank: int,
    frequency_point_count: int,
    spectral_tier: str,
    capacity_tier: str,
    overfit_risk: str,
    gpu_info: dict[str, Any],
) -> dict[str, Any]:
    # These ordered tiers make the heuristic easier to reason about than a fully
    # continuous formula because the output lands on familiar, hand-checked values.
    latent_tiers = [48, 64, 96, 128, 160]
    width_tiers = [128, 192, 256, 384, 512, 768]
    depth_tiers = [3, 4, 5, 6]
    band_tiers = [8, 12, 16, 20, 24]
    batch_tiers = [8, 16, 32, 64]
    lr_tiers = [7e-5, 1e-4, 2e-4]
    dropout_tiers = [0.02, 0.05, 0.10, 0.15]
    weight_decay_tiers = [5e-5, 1e-4, 3e-4]
    epoch_tiers = [200, 300, 400]
    patience_tiers = [15, 20, 30]

    if ground_truth_rank <= 24:
        latent_dim = 48
    elif ground_truth_rank <= 48:
        latent_dim = 64
    elif ground_truth_rank <= 80:
        latent_dim = 96
    elif ground_truth_rank <= 128:
        latent_dim = 128
    else:
        latent_dim = 160
    if capacity_tier == "conservative":
        latent_dim = _shift_tier(latent_tiers, latent_dim, -1)
    elif capacity_tier == "aggressive" and spectral_tier == "high":
        latent_dim = _shift_tier(latent_tiers, latent_dim, 1)

    # Width grows roughly with latent dimension, then gets adjusted for richer inputs
    # or more complex spectra.
    width = _nearest_tier(width_tiers, latent_dim * 4)
    width_shift = 0
    if capacity_tier == "conservative":
        width_shift -= 1
    elif capacity_tier == "aggressive":
        width_shift += 1
    if active_input_feature_dim > 16:
        width_shift += 1
    if spectral_tier == "high":
        width_shift += 1
    width = _shift_tier(width_tiers, width, width_shift)

    if train_count >= 5000 and spectral_tier == "high" and active_input_feature_dim > 12:
        depth = 6
    elif spectral_tier == "high" and capacity_tier != "conservative":
        depth = 5
    elif spectral_tier == "medium" and capacity_tier == "aggressive":
        depth = 5
    elif spectral_tier == "low" and capacity_tier == "conservative":
        depth = 3
    else:
        depth = 4

    if spectral_tier == "low":
        fourier_bands = 8 if frequency_point_count < 300 else 12
    elif spectral_tier == "medium":
        fourier_bands = 12 if frequency_point_count < 300 else 16
    else:
        fourier_bands = 16 if frequency_point_count < 300 else 20
    if frequency_point_count >= 800 and spectral_tier == "high":
        fourier_bands = 24

    if overfit_risk == "high":
        dropout = 0.15
        weight_decay = 3e-4
    elif overfit_risk == "medium":
        dropout = 0.10
        weight_decay = 1e-4
    else:
        dropout = 0.05
        weight_decay = 5e-5
    if train_count >= 4000 and spectral_tier == "low":
        dropout = 0.02

    if width <= 256 and depth <= 4:
        learning_rate = 2e-4
    elif capacity_tier == "conservative" and width >= 512:
        learning_rate = 7e-5
    else:
        learning_rate = 1e-4

    if not gpu_info["available"]:
        batch_size = 16 if train_count >= 256 else 8
    elif float(gpu_info["memory_gb"]) < 8.0:
        batch_size = 16
    elif float(gpu_info["memory_gb"]) <= 16.0:
        batch_size = 32
    else:
        batch_size = 64
    if frequency_point_count >= 600 or width >= 512:
        batch_size = _shift_tier(batch_tiers, batch_size, -1)

    if train_count < 800:
        epochs = 400
        patience = 30
    elif train_count < 4000:
        epochs = 300
        patience = 20
    else:
        epochs = 200
        patience = 15

    # Start from the baseline training defaults, then override only the fields the
    # heuristic is actually choosing.
    defaults = TrainConfig()
    suggested = TrainConfig(
        data_root=(
            request.data_root if request.input_feature_path is None and request.ground_truth_data_dir is None else None
        ),
        input_feature_path=request.input_feature_path,
        ground_truth_data_dir=request.ground_truth_data_dir,
        cache_path=cache_path,
        output_dir=defaults.output_dir,
        seed=request.seed,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip=defaults.gradient_clip,
        train_frac=request.train_frac,
        val_frac=request.val_frac,
        latent_dim=latent_dim,
        width=width,
        depth=depth,
        fourier_bands=fourier_bands,
        dropout=dropout,
        use_amp=bool(gpu_info["available"]),
        max_samples=request.max_samples,
    )

    ranges = {
        # Expose nearby alternatives so the quick-search module can explore around
        # the heuristic recommendation without rebuilding its own search space logic.
        "latent_dim": _neighbor_range(latent_tiers, latent_dim),
        "width": _neighbor_range(width_tiers, width),
        "depth": _neighbor_range(depth_tiers, depth),
        "fourier_bands": _neighbor_range(band_tiers, fourier_bands),
        "batch_size": _neighbor_range(batch_tiers, batch_size),
        "learning_rate": _neighbor_range(lr_tiers, learning_rate),
        "dropout": _neighbor_range(dropout_tiers, dropout),
        "weight_decay": _neighbor_range(weight_decay_tiers, weight_decay),
        "epochs": _neighbor_range(epoch_tiers, epochs),
        "patience": _neighbor_range(patience_tiers, patience),
    }
    rationale = {
        "latent_dim": f"Ground-truth effective rank is {ground_truth_rank}, so latent capacity is anchored near that scale.",
        "width": f"Width is tied to latent dimension and adjusted for {capacity_tier} capacity with {spectral_tier} spectral complexity.",
        "depth": f"Depth {depth} balances train-set size {train_count} with {spectral_tier} spectral complexity.",
        "fourier_bands": f"Frequency encoding uses {fourier_bands} bands for {frequency_point_count} frequency points and {spectral_tier} curve complexity.",
        "batch_size": _batch_rationale(batch_size, gpu_info, width, frequency_point_count),
        "learning_rate": f"Learning rate {learning_rate:.1e} is a stable starting point for a {width}-wide, depth-{depth} network.",
        "dropout": f"Dropout {dropout:.2f} reflects an estimated {overfit_risk} overfit risk.",
        "weight_decay": f"Weight decay {weight_decay:.1e} complements the same {overfit_risk} overfit estimate.",
        "epochs": f"Epoch budget {epochs} follows the current train split size of {train_count} samples.",
        "patience": f"Patience {patience} pairs with the suggested epoch budget to stop early if validation plateaus.",
    }
    quick_search_hints = {
        "latent_dim": ranges["latent_dim"]["candidates"],
        "width": ranges["width"]["candidates"],
        "depth": ranges["depth"]["candidates"],
        "learning_rate": ranges["learning_rate"]["candidates"],
        "dropout": ranges["dropout"]["candidates"],
    }
    return {
        "config": asdict(suggested),
        "ranges": ranges,
        "rationale": rationale,
        "quick_search_hints": quick_search_hints,
    }


def _suggest_transfer(
    frequency_point_count: int,
    spectral_tier: str,
    baseline_learning_rate: float,
    baseline_batch_size: int,
    gpu_info: dict[str, Any],
) -> dict[str, Any]:
    # Transfer suggestions depend mostly on the frequency axis structure and on the
    # baseline settings that the user is likely to run first.
    viable_bands = [value for value in range(2, min(frequency_point_count, 24) + 1)]
    if viable_bands:
        num_bands = min(
            viable_bands,
            key=lambda value: (
                abs(((frequency_point_count - (frequency_point_count % value)) / value) - 64.0),
                frequency_point_count % value,
            ),
        )
    else:
        num_bands = 1

    if spectral_tier == "low":
        iterations = 6
        transfer_epochs = 60
    elif spectral_tier == "medium":
        iterations = 8
        transfer_epochs = 80
    else:
        iterations = 10
        transfer_epochs = 100

    learning_rate = float(min(max(baseline_learning_rate * 0.5, 5e-5), 1e-4))
    if gpu_info["available"]:
        batch_size = max(32, min(64, baseline_batch_size * 2))
    else:
        batch_size = baseline_batch_size

    config = {
        "enabled_by_default": False,
        "num_bands": int(num_bands),
        "iterations": int(iterations),
        "transfer_epochs": int(transfer_epochs),
        "batch_size": int(batch_size),
        "learning_rate": learning_rate,
        "weight_decay": None,
        "gradient_clip": 1.0,
        "use_amp": bool(gpu_info["available"]),
    }
    ranges = {
        "num_bands": _candidate_span([value for value in viable_bands if value >= 2], num_bands),
        "iterations": _candidate_span([4, 6, 8, 10, 12], iterations),
        "transfer_epochs": _candidate_span([40, 60, 80, 100, 120], transfer_epochs),
        "batch_size": _candidate_span([8, 16, 32, 64], batch_size),
        "learning_rate": _candidate_span([5e-5, 7e-5, 1e-4], learning_rate),
    }
    rationale = {
        "num_bands": (
            f"{num_bands} band(s) keep about "
            f"{(frequency_point_count - (frequency_point_count % max(num_bands, 1))) / max(num_bands, 1):.0f} "
            "frequency points per band. Any trailing remainder can be trimmed for transfer."
        ),
        "iterations": f"Transfer iterations increase with {spectral_tier} spectral complexity.",
        "transfer_epochs": f"Per-band transfer epochs scale with the same {spectral_tier} complexity estimate.",
        "batch_size": "Transfer learning can usually use a larger batch size because each band sees a smaller slice of frequency space.",
        "learning_rate": f"Transfer learning rate {learning_rate:.1e} starts at half of the baseline rate.",
    }
    return {"config": config, "ranges": ranges, "rationale": rationale}


def _confidence_summary(
    train_count: int,
    ground_truth_rank: int,
    flattened_ground_truth_dim: int,
    sample_to_ground_truth_rank_ratio: float,
) -> tuple[str, str]:
    # Confidence is intentionally simple: it reflects how much apparent output
    # complexity is supported by the amount of training data available.
    rank_fraction = ground_truth_rank / max(1, min(train_count, flattened_ground_truth_dim))
    if train_count >= 1000 and sample_to_ground_truth_rank_ratio >= 15.0 and rank_fraction <= 0.65:
        return "high", "Plenty of train samples are available relative to the estimated ground-truth complexity."
    if train_count >= 250 and sample_to_ground_truth_rank_ratio >= 6.0:
        return "medium", "The scan captures a useful amount of structure, but optimization still matters."
    return "low", "The dataset looks sparse relative to ground-truth complexity, so these values should be treated as rough starters."


def _build_warnings(diagnostics: dict[str, Any], confidence: str, transfer: dict[str, Any]) -> list[str]:
    # Keep caveats in a dedicated list so callers can surface them prominently in a UI.
    warnings: list[str] = [
        "These suggestions come from a data scan only; no training run has been performed.",
        "Learning rate and regularization are less predictable from static data than model size is.",
    ]
    if confidence != "high":
        warnings.append("Run a quick hyperparameter search after accepting these values to refine them.")
    if not diagnostics["gpu_available"]:
        warnings.append("CUDA was not detected, so training will fall back to CPU unless the environment changes.")
    if diagnostics["estimated_overfit_risk"] == "high":
        warnings.append(
            "The current train split looks data-limited relative to ground-truth complexity, so watch validation loss closely."
        )
    if int(transfer["config"]["num_bands"]) <= 1:
        warnings.append(
            "Frequency-band transfer suggestions are limited because the frequency count is too small for multiple equal-width bands."
        )
    return warnings


def _neighbor_range(ordered_values: list[Any], value: Any) -> dict[str, Any]:
    # Return a small local window around the selected value.
    candidates = _candidate_span(ordered_values, value)
    return {
        "lower": candidates["candidates"][0],
        "upper": candidates["candidates"][-1],
        "candidates": candidates["candidates"],
    }


def _candidate_span(ordered_values: list[Any], value: Any) -> dict[str, list[Any]]:
    # Pick the nearest tier and expose it together with its immediate neighbors.
    if not ordered_values:
        return {"candidates": [value]}
    idx = min(range(len(ordered_values)), key=lambda index: abs(float(ordered_values[index]) - float(value)))
    start = max(idx - 1, 0)
    stop = min(idx + 1, len(ordered_values) - 1)
    return {"candidates": ordered_values[start : stop + 1]}


def _nearest_tier(ordered_values: list[int], value: float) -> int:
    # Snap a continuous heuristic value to the nearest hand-picked tier.
    return min(ordered_values, key=lambda candidate: abs(candidate - value))


def _shift_tier(ordered_values: list[Any], value: Any, shift: int) -> Any:
    # Move left or right within an ordered tier list while clamping at the ends.
    idx = ordered_values.index(value)
    idx = max(0, min(len(ordered_values) - 1, idx + shift))
    return ordered_values[idx]


def _divisors(value: int) -> set[int]:
    # Exact divisors matter because the transfer workflow splits the frequency axis
    # into equal-width bands with no leftover points.
    divisors: set[int] = set()
    for candidate in range(1, int(value**0.5) + 1):
        if value % candidate == 0:
            divisors.add(candidate)
            divisors.add(value // candidate)
    return divisors


def _batch_rationale(batch_size: int, gpu_info: dict[str, Any], width: int, frequency_point_count: int) -> str:
    # Keep the explanation human-readable so the saved suggestion report is self-explanatory.
    if gpu_info["available"]:
        return (
            f"Batch size {batch_size} fits a {gpu_info['memory_gb']:.1f} GB GPU while leaving room for "
            f"{width}-wide layers and {frequency_point_count} frequency points."
        )
    return f"Batch size {batch_size} is conservative for CPU training."
