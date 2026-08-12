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

from .data import CACHE_PATH, DATA_ROOT, design_split_indices, ensure_cache, split_indices
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
    # Same semantics as ``TrainConfig.split_corner_columns``: when set, the
    # diagnostics and heuristics are computed from the design-level split so they
    # describe the split the recommended run will actually train on.
    split_corner_columns: list[str] | None = None
    # Names the columns that IDENTIFY a design instead (safer; see TrainConfig).
    split_design_columns: list[str] | None = None
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
        should_stop=should_stop,
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
    # the information a real training pipeline should legitimately use. The split mode
    # must match the recommended run's, or the persisted diagnostics (split sizes,
    # train-count-driven epoch/capacity tiers) would describe a split it never uses.
    if config.split_corner_columns or config.split_design_columns:
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
    train_features_raw = features[split["train"]]
    # Match the training pipeline's active-feature criterion (data.py / runner.py):
    # max != min keeps features with tiny but meaningful SI-unit values (e.g.
    # capacitance in farads ~1e-13) that an absolute std threshold would drop.
    active_mask = train_features_raw.max(axis=0) != train_features_raw.min(axis=0)
    if not np.any(active_mask):
        raise ValueError("All input-feature columns are constant in the training split.")

    active_names = [name for name, keep in zip(input_feature_names, active_mask, strict=True) if keep]
    dropped_names = [name for name, keep in zip(input_feature_names, active_mask, strict=True) if not keep]
    input_feature_mean = train_features_raw[:, active_mask].mean(axis=0).astype(np.float32)
    input_feature_std = train_features_raw[:, active_mask].std(axis=0).astype(np.float32)
    input_feature_std[input_feature_std == 0] = 1.0

    train_features = (train_features_raw[:, active_mask] - input_feature_mean) / input_feature_std
    train_targets = targets[split["train"]]

    # Standardize each ground-truth channel before flattening so the effective-rank
    # diagnostic is scale-invariant: otherwise a single large-magnitude channel (e.g.
    # dB vs degrees) would dominate the singular spectrum and distort the estimate.
    ground_truth_matrix = _standardize_per_channel(train_targets)
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
    capacity_tier = _capacity_tier(sample_to_ground_truth_rank_ratio)
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
        ground_truth_channels=len(channel_names),
        ground_truth_rank=ground_truth_rank,
        frequency_point_count=int(train_targets.shape[-1]),
        spectral_tier=spectral_stats["tier"],
        capacity_tier=capacity_tier,
        gpu_info=gpu_info,
    )
    # The overfit estimate is now derived from the *chosen* model size relative to the
    # supervised signal available, so it is read back from the baseline result.
    overfit_risk = baseline["overfit_risk"]
    param_count = baseline["param_count"]
    signal_per_param = baseline["signal_per_param"]
    transfer = _suggest_transfer(
        frequency_point_count=int(train_targets.shape[-1]),
        spectral_tier=spectral_stats["tier"],
        baseline_learning_rate=float(baseline["config"]["learning_rate"]),
        baseline_batch_size=int(baseline["config"]["batch_size"]),
        gpu_info=gpu_info,
    )

    confidence, confidence_reason = _confidence_summary(
        train_count=len(split["train"]),
        signal_per_param=signal_per_param,
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
        "estimated_param_count": int(param_count),
        "supervised_signal_per_param": float(signal_per_param),
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


def _standardize_per_channel(targets: np.ndarray) -> np.ndarray:
    # Flatten (samples, channels, frequency) -> (samples, channels*frequency) with each
    # channel z-scored across samples and frequency. This makes the effective-rank
    # diagnostic invariant to per-channel scale/units so no single large-magnitude
    # channel dominates the singular spectrum.
    #
    # Caveat: because every channel is rescaled to unit variance, a pure-noise channel
    # is weighted the same as a signal channel and will inflate the effective rank. That
    # is acceptable here because ground-truth channels are smooth physical curves, but it
    # means the rank (and the width anchored to it) is not robust to a garbage channel.
    arr = targets.astype(np.float64, copy=False)
    if arr.ndim != 3:
        return arr.reshape(arr.shape[0], -1)
    mean = arr.mean(axis=(0, 2), keepdims=True)
    std = arr.std(axis=(0, 2), keepdims=True)
    # Treat a channel as constant only when its std is negligible *relative to its own
    # magnitude* (a units-independent test), so a structured but tiny-magnitude channel
    # is still standardized rather than silently dropped by an absolute floor.
    scale = np.abs(arr).max(axis=(0, 2), keepdims=True)
    constant = std <= 1e-9 * (scale + 1e-30)
    std = np.where(constant, 1.0, std)
    return ((arr - mean) / std).reshape(arr.shape[0], -1)


def _resample_curves(curves: np.ndarray, n: int) -> np.ndarray:
    # Linearly resample each curve to a fixed number of points so the wiggliness
    # diagnostic measures curve *shape* and does not depend on how densely the
    # frequency axis happens to be sampled. Note: linear resampling has no
    # anti-aliasing, so a feature much narrower than ~1/n of the axis (when the native
    # resolution F >> n) can be missed; wider sharp features survive.
    f = curves.shape[1]
    if f == n:
        return curves
    src = np.linspace(0.0, 1.0, f)
    dst = np.linspace(0.0, 1.0, n)
    return np.vstack([np.interp(dst, src, row) for row in curves])


# Spectral-complexity tier thresholds, calibrated on synthetic curves at the fixed
# resample resolution: smooth/broad-hump -> low, single sharp resonance -> medium,
# multi-resonance -> high, and the same tier across sampling densities. (Note: heavy
# measurement noise also reads as "high" -- this proxy cannot separate noise from
# genuine high-frequency structure.)
_SPECTRAL_LOW_SCORE = 2.3
_SPECTRAL_LOW_CURVATURE = 0.15
_SPECTRAL_HIGH_SCORE = 5.0
_SPECTRAL_HIGH_CURVATURE = 1.0


def _spectral_complexity(targets: np.ndarray, resample_points: int = 64) -> dict[str, float | str]:
    # View every output channel for every sample as one curve over frequency, then
    # measure how wiggly those curves are using first and second differences.
    curves = targets.reshape(-1, targets.shape[-1]).astype(np.float64, copy=False)
    if curves.shape[-1] < 3:
        return {"variation_ratio": 0.0, "curvature_ratio": 0.0, "score": 0.0, "tier": "low"}

    # Resample to a fixed resolution (density-invariant), then summarize with a high
    # percentile across curves -- not the median -- so a minority of sharp-resonance
    # channels is not averaged away by many smooth ones.
    curves = _resample_curves(curves, resample_points)
    first_diff = np.diff(curves, axis=1)
    second_diff = np.diff(curves, n=2, axis=1)
    curve_range = np.ptp(curves, axis=1)
    curve_range[curve_range < 1e-8] = 1.0
    total_variation = np.sum(np.abs(first_diff), axis=1)
    total_variation[total_variation < 1e-8] = 1.0

    variation_ratio = float(np.percentile(total_variation / curve_range, 90))
    curvature_ratio = float(np.percentile(np.sum(np.abs(second_diff), axis=1) / total_variation, 90))
    score = variation_ratio + 1.5 * curvature_ratio

    if score < _SPECTRAL_LOW_SCORE and curvature_ratio < _SPECTRAL_LOW_CURVATURE:
        tier = "low"
    elif score < _SPECTRAL_HIGH_SCORE and curvature_ratio < _SPECTRAL_HIGH_CURVATURE:
        tier = "medium"
    else:
        tier = "high"
    return {
        "variation_ratio": variation_ratio,
        "curvature_ratio": curvature_ratio,
        "score": float(score),
        "tier": tier,
    }


def _capacity_tier(sample_to_ground_truth_rank_ratio: float) -> str:
    # When the dataset has many samples relative to output complexity, we can afford
    # a more expressive model. When that ratio is low, we should be more conservative.
    if sample_to_ground_truth_rank_ratio < 8.0:
        return "conservative"
    if sample_to_ground_truth_rank_ratio < 25.0:
        return "balanced"
    return "aggressive"


def _estimate_spectranet_params(
    input_dim: int, channels: int, frequency_point_count: int, width: int, depth: int
) -> int:
    # Parameter count of the default SpectraNet (a dense network from input features
    # straight to the flattened channels*frequency output). Used purely as a model-size
    # proxy for the overfit-risk and confidence estimates. NOTE: a SpectraHydra has
    # ~2-5x more parameters (its encoder expands to 4*width at the midpoint), so for that
    # model this understates the parameter count and therefore the overfit risk.
    out_dim = max(channels * frequency_point_count, 1)
    if depth <= 1:
        return input_dim * out_dim + out_dim
    params = input_dim * width + width                 # first hidden layer
    params += (depth - 2) * (width * width + width)    # intermediate hidden layers
    params += width * out_dim + out_dim                # output projection
    return int(params)


def _estimate_spectrahydra_params(
    input_dim: int, channels: int, frequency_point_count: int, width: int, depth: int
) -> int:
    # Parameter count of the SpectraHydra: a Linear->LayerNorm->GELU encoder whose hidden
    # sizes follow the same expand-then-contract schedule the model uses, plus one
    # Linear output head per channel. Mirrors xfmr_v2.model so the size proxy is honest
    # for this (much larger) architecture.
    from .model import _symmetric_hidden_sizes

    out_dim = max(frequency_point_count, 1)
    params = 0
    dim = input_dim
    for hidden in _symmetric_hidden_sizes(width, depth):
        params += dim * hidden + hidden    # Linear
        params += 2 * hidden               # LayerNorm weight + bias
        dim = hidden
    params += channels * (dim * out_dim + out_dim)  # one output head per channel
    return int(params)


def _estimate_params(
    model_type: str, input_dim: int, channels: int, frequency_point_count: int, width: int, depth: int
) -> int:
    # Dispatch to the right size proxy so the overfit/confidence estimates reflect the
    # architecture the heuristic actually recommends. SpectraHydraProj is a SpectraHydra
    # plus a small corner projection (Linear(corners -> 16), negligible next to the
    # trunk), so it shares the SpectraHydra estimate.
    if model_type in ("SpectraHydra", "SpectraHydraProj"):
        return _estimate_spectrahydra_params(input_dim, channels, frequency_point_count, width, depth)
    return _estimate_spectranet_params(input_dim, channels, frequency_point_count, width, depth)


def _overfit_risk_from_signal(signal_per_param: float) -> str:
    # signal_per_param = (train_samples * channels * frequency_points) / parameter_count:
    # how many supervised target scalars the data provides per model parameter. Fewer
    # supervised scalars per parameter -> more overfitting headroom is needed.
    if signal_per_param < 1.0:
        return "high"
    if signal_per_param < 5.0:
        return "medium"
    return "low"


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
    ground_truth_channels: int,
    ground_truth_rank: int,
    frequency_point_count: int,
    spectral_tier: str,
    capacity_tier: str,
    gpu_info: dict[str, Any],
) -> dict[str, Any]:
    # These ordered tiers make the heuristic easier to reason about than a fully
    # continuous formula because the output lands on familiar, hand-checked values.
    width_tiers = [128, 192, 256, 384, 512, 768]
    depth_tiers = [3, 4, 5, 6]
    batch_tiers = [8, 16, 32, 64]
    lr_tiers = [7e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    weight_decay_tiers = [5e-5, 1e-4, 3e-4]
    epoch_tiers = [200, 300, 400]

    # Anchor width directly to the ground-truth effective rank, then adjust for
    # richer inputs or more complex spectra.
    if ground_truth_rank <= 24:
        width = 192
    elif ground_truth_rank <= 48:
        width = 256
    elif ground_truth_rank <= 80:
        width = 384
    elif ground_truth_rank <= 128:
        width = 512
    else:
        width = 768
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

    # Architecture choice. The shared-encoder SpectraHydra (Linear->LayerNorm->GELU blocks
    # that expand to 4x width at the midpoint, then contract, with one output head per
    # channel) is substantially more accurate than the flat SpectraNet whenever there is
    # enough data to support it. SpectraNet is reserved for data-starved (conservative)
    # cases where the larger model would mostly add overfitting risk.
    if capacity_tier == "conservative":
        model_type = "SpectraNet"
        loss_function = "rmse"
        scheduler = "plateau"
    else:
        model_type = "SpectraHydra"
        depth = 5  # yields the validated [w, 2w, 4w, 2w, w] encoder
        loss_function = "mse"
        scheduler = "cosine"

    # Estimate the chosen model's size and derive the overfit risk from how much
    # supervised signal the data provides per parameter. This closes the loop so a
    # larger suggested model correctly calls for more regularization.
    #
    # The supervised-signal term counts every channel*frequency output scalar as an
    # independent constraint. Because a smooth curve carries only ~rank effective
    # degrees of freedom over frequency, this OVERSTATES the true signal (and so
    # understates overfit risk) for smooth data -- the cutoffs below are tuned with that
    # optimism in mind, and the warnings still steer the user to a real validation check.
    param_count = _estimate_params(
        model_type, active_input_feature_dim, ground_truth_channels, frequency_point_count, width, depth
    )
    signal_per_param = (train_count * ground_truth_channels * frequency_point_count) / max(param_count, 1)
    overfit_risk = _overfit_risk_from_signal(signal_per_param)

    if model_type == "SpectraHydra":
        # Notebook-validated AdamW recipe for the shared-encoder model (with cosine LR).
        learning_rate = 1e-3
        weight_decay = 1e-4
    else:
        if overfit_risk == "high":
            weight_decay = 3e-4
        elif overfit_risk == "medium":
            weight_decay = 1e-4
        else:
            weight_decay = 5e-5
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
    elif train_count < 4000:
        epochs = 300
    else:
        epochs = 200
    if model_type == "SpectraHydra":
        # The larger encoder under a cosine schedule needs a longer budget to converge;
        # with early stopping removed, the validated recipe trains the full 500 epochs.
        epochs = max(epochs, 500)

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
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        train_frac=request.train_frac,
        val_frac=request.val_frac,
        split_corner_columns=request.split_corner_columns,
        split_design_columns=request.split_design_columns,
        model_type=model_type,
        width=width,
        depth=depth,
        loss_function=loss_function,
        scheduler=scheduler,
        use_amp=bool(gpu_info["available"]),
        max_samples=request.max_samples,
    )

    ranges = {
        # Expose nearby alternatives so the quick-search module can explore around
        # the heuristic recommendation without rebuilding its own search space logic.
        "width": _neighbor_range(width_tiers, width),
        "depth": _neighbor_range(depth_tiers, depth),
        "batch_size": _neighbor_range(batch_tiers, batch_size),
        "learning_rate": _neighbor_range(lr_tiers, learning_rate),
        "weight_decay": _neighbor_range(weight_decay_tiers, weight_decay),
        "epochs": _neighbor_range(epoch_tiers, epochs),
    }
    rationale = {
        "model_type": (
            f"{model_type} chosen for {capacity_tier} capacity: the shared-encoder SpectraHydra "
            "(with MSE loss + cosine LR) is used whenever the data supports it, and the flat "
            "SpectraNet only for data-starved cases."
        ),
        "width": f"Width is anchored to the ground-truth effective rank ({ground_truth_rank}) and adjusted for {capacity_tier} capacity with {spectral_tier} spectral complexity.",
        "depth": (
            f"Depth {depth} gives the validated expand-then-contract SpectraHydra encoder."
            if model_type == "SpectraHydra"
            else f"Depth {depth} balances train-set size {train_count} with {spectral_tier} spectral complexity."
        ),
        "batch_size": _batch_rationale(batch_size, gpu_info, width, frequency_point_count),
        "learning_rate": (
            f"Learning rate {learning_rate:.1e} is the notebook-validated AdamW rate for the SpectraHydra encoder."
            if model_type == "SpectraHydra"
            else f"Learning rate {learning_rate:.1e} is a safe AdamW default; it cannot be inferred from a static scan, so refine it with a quick search."
        ),
        "weight_decay": f"Weight decay {weight_decay:.1e} reflects a {overfit_risk} overfit estimate ({signal_per_param:.1f} supervised values per parameter).",
        "epochs": f"Epoch budget {epochs} follows the current train split size of {train_count} samples.",
    }
    quick_search_hints = {
        "width": ranges["width"]["candidates"],
        "depth": ranges["depth"]["candidates"],
        "learning_rate": ranges["learning_rate"]["candidates"],
    }
    return {
        "config": asdict(suggested),
        "ranges": ranges,
        "rationale": rationale,
        "quick_search_hints": quick_search_hints,
        "param_count": int(param_count),
        "signal_per_param": float(signal_per_param),
        "overfit_risk": overfit_risk,
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


def _confidence_summary(train_count: int, signal_per_param: float) -> tuple[str, str]:
    # Confidence reflects how much supervised signal backs the suggested model: it
    # combines the raw sample count with the supervised-values-per-parameter ratio so a
    # large model on little data is never reported as high confidence.
    if train_count >= 1000 and signal_per_param >= 5.0:
        return "high", "Plenty of training samples and supervised signal per model parameter."
    if train_count >= 250 and signal_per_param >= 1.0:
        return "medium", "Enough structure to start from, but optimization and regularization still matter."
    return "low", "Few samples or few supervised values per parameter, so treat these as rough starters."


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
            "The suggested model is large relative to the available training signal, so watch validation loss closely."
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
