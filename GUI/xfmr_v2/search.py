"""Quick hyperparameter search for the v2 XFMR baseline model.

The search strategy is intentionally lightweight:
- start from the heuristic suggestion produced by `xfmr_v2.suggest`
- build a small neighborhood of nearby candidate configurations
- train each candidate for a short run
- rank them with a user-facing tradeoff score

This keeps the search cheap enough to use as a practical refinement step after
the initial scan-only suggestion.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

from .data import CACHE_PATH, DATA_ROOT
from .progress import ProgressCallback, StopChecker, emit_progress, request_stop
from .runner import TrainConfig, make_run_dir, run_baseline_trial
from .suggest import SuggestConfig, suggest_initial_settings


@dataclass
class SearchConfig:
    """Configuration for short hyperparameter search runs."""

    # Data-selection options mirror the training config so the same dataset-loading
    # code can be reused for search trials.
    data_root: str | None = str(DATA_ROOT)
    input_feature_path: str | None = None
    ground_truth_data_dir: str | None = None
    cache_path: str = str(CACHE_PATH)
    output_dir: str = "artifacts/runs/quick_search_v2"
    seed: int = 42
    train_frac: float = 0.8
    val_frac: float = 0.1
    max_samples: int | None = None
    search_max_samples: int | None = None
    model_type: str = "FlatMLP"
    trial_count: int = 6
    epochs_per_trial: int = 60
    patience_per_trial: int | None = None
    objective: str = "balanced"
    variance_threshold: float = 0.95
    show_trial_progress: bool = False


def quick_hyperparameter_search(
    config: SearchConfig,
    show_progress: bool = False,
    progress_callback: ProgressCallback | None = None,
    should_stop: StopChecker | None = None,
) -> dict[str, Any]:
    """Run a handful of short baseline trials and rank them by a visible tradeoff score."""
    # Normalize the objective string once so the rest of the function can work with
    # a small set of canonical names.
    objective = _normalize_objective(config.objective)
    _validate_search_config(config, objective)
    emit_progress(
        progress_callback,
        event="started",
        phase="search",
        message="Quick hyperparameter search started.",
        config=asdict(config),
        objective=objective,
    )

    search_sample_cap = config.search_max_samples if config.search_max_samples is not None else config.max_samples
    if show_progress:
        print("Generating search-centered initial suggestion...")

    # The search does not invent a starting point from scratch. Instead it asks the
    # suggestion module for a reasonable baseline and then explores nearby variants.
    suggestion = suggest_initial_settings(
        SuggestConfig(
            data_root=config.data_root,
            input_feature_path=config.input_feature_path,
            ground_truth_data_dir=config.ground_truth_data_dir,
            cache_path=config.cache_path,
            seed=config.seed,
            train_frac=config.train_frac,
            val_frac=config.val_frac,
            max_samples=search_sample_cap,
            variance_threshold=config.variance_threshold,
        ),
        show_progress=show_progress,
        progress_callback=progress_callback,
        should_stop=should_stop,
    )
    request_stop(should_stop)
    emit_progress(
        progress_callback,
        event="suggestion_ready",
        phase="search",
        message="Initial suggestion is ready for quick search.",
        confidence=suggestion["confidence"],
        confidence_reason=suggestion["confidence_reason"],
    )

    base_trial_config = TrainConfig(**suggestion["suggested_baseline_config"])
    base_trial_config.model_type = config.model_type
    base_trial_config.epochs = config.epochs_per_trial
    base_trial_config.patience = _resolve_patience(config.patience_per_trial, config.epochs_per_trial)
    base_trial_config.max_samples = search_sample_cap

    # `base_full_config` preserves the same candidate settings but without the short-run
    # search-specific epoch cap, so the best candidate can be recommended for a full run.
    base_full_config = TrainConfig(**suggestion["suggested_baseline_config"])
    base_full_config.model_type = config.model_type
    base_full_config.max_samples = config.max_samples

    # Build a compact set of nearby alternatives rather than a huge Cartesian grid.
    candidates = _build_candidate_entries(
        base_trial_config=base_trial_config,
        base_full_config=base_full_config,
        ranges=suggestion["suggested_baseline_ranges"],
        trial_count=config.trial_count,
    )
    if show_progress:
        print(f"Running {len(candidates)} quick-search trial(s)...")
    emit_progress(
        progress_callback,
        event="search_space_ready",
        phase="search",
        message=f"Prepared {len(candidates)} quick-search trial(s).",
        trial_count=len(candidates),
        candidate_labels=[candidate["label"] for candidate in candidates],
    )

    trial_results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        request_stop(should_stop)
        if show_progress:
            print(f"trial {index}/{len(candidates)}: {candidate['label']}")
        emit_progress(
            progress_callback,
            event="trial_started",
            phase="search",
            message=f"Quick-search trial {index}/{len(candidates)} started.",
            trial_index=index,
            trial_count=len(candidates),
            trial_label=candidate["label"],
            search_config=asdict(candidate["trial_config"]),
        )
        trial = run_baseline_trial(
            config=candidate["trial_config"],
            show_progress=config.show_trial_progress,
            save_artifacts=False,
            evaluation_split="val",
            progress_callback=progress_callback,
            should_stop=should_stop,
            event_context={
                "trial_index": index,
                "trial_count": len(candidates),
                "trial_label": candidate["label"],
            },
        )
        if trial.get("status") == "stopped":
            summary = _build_search_summary(
                status="stopped",
                config=config,
                objective=objective,
                suggestion=suggestion,
                patience_per_trial=base_trial_config.patience,
                search_sample_cap=search_sample_cap,
                trial_results=trial_results,
            )
            emit_progress(
                progress_callback,
                event="stopped",
                phase="search",
                message="Quick hyperparameter search stopped.",
                run_dir=summary["run_dir"],
                trial_count_completed=summary["trial_count_completed"],
            )
            return summary
        trial_results.append(
            {
                "trial_index": index,
                "label": candidate["label"],
                "search_config": asdict(candidate["trial_config"]),
                "recommended_full_config": asdict(candidate["full_config"]),
                "best_val_loss": float(trial["best_val_loss"]),
                "average_val_mae": float(trial["average_evaluation_mae"]),
                "val_loss": float(trial["evaluation_loss"]),
                "frequency_val_mae": trial["frequency_mae"],
                "runtime_seconds": float(trial["runtime_seconds"]),
                "epochs_completed": int(trial["epochs_completed"]),
                "best_epoch": int(trial["best_epoch"]),
                "history": trial["history"],
                "device": trial["device"],
            }
        )
        emit_progress(
            progress_callback,
            event="trial_completed",
            phase="search",
            message=f"Quick-search trial {index}/{len(candidates)} completed.",
            trial_index=index,
            trial_count=len(candidates),
            trial_label=candidate["label"],
            best_val_loss=float(trial["best_val_loss"]),
            average_val_mae=float(trial["average_evaluation_mae"]),
            runtime_seconds=float(trial["runtime_seconds"]),
            recommended_full_config=asdict(candidate["full_config"]),
        )

    summary = _build_search_summary(
        status="ok",
        config=config,
        objective=objective,
        suggestion=suggestion,
        patience_per_trial=base_trial_config.patience,
        search_sample_cap=search_sample_cap,
        trial_results=trial_results,
    )
    emit_progress(
        progress_callback,
        event="completed",
        phase="search",
        message="Quick hyperparameter search completed.",
        run_dir=summary["run_dir"],
        trial_count_completed=summary["trial_count_completed"],
        recommended_trial=summary["recommended_trial"],
    )
    return summary


def _build_search_summary(
    status: str,
    config: SearchConfig,
    objective: str,
    suggestion: dict[str, Any],
    patience_per_trial: int,
    search_sample_cap: int | None,
    trial_results: list[dict[str, Any]],
) -> dict[str, Any]:
    # Every search gets its own artifact directory, even if the search stopped early.
    # That way the user can always inspect partial results.
    run_dir = make_run_dir(config.output_dir)
    ranked_trials: list[dict[str, Any]] = []
    compact_trials: list[dict[str, Any]] = []
    recommended_full_config: dict[str, Any] | None = None
    tradeoff_plot_path: str | None = None
    trial_results_path = str((run_dir / "trial_results.json").resolve())
    score_formula = _score_formula_description(objective)

    if trial_results:
        # Rank and score the trials only after all successful trial summaries are collected.
        ranked_trials, scoring = _score_trials(trial_results, objective)
        score_formula = scoring["description"]
        recommended_full_config = ranked_trials[0]["recommended_full_config"]
        tradeoff_plot_file = run_dir / "tradeoff_plot.png"
        _plot_tradeoff(ranked_trials, tradeoff_plot_file, objective)
        tradeoff_plot_path = str(tradeoff_plot_file.resolve())
        compact_trials = [_compact_trial(trial) for trial in ranked_trials]

    summary = {
        "status": status,
        "run_dir": str(run_dir.resolve()),
        "objective": objective,
        "objective_label": _objective_label(objective),
        "score_formula": score_formula,
        "epochs_per_trial": int(config.epochs_per_trial),
        "patience_per_trial": int(patience_per_trial),
        "search_max_samples": search_sample_cap,
        "trial_count_requested": int(config.trial_count),
        "trial_count_completed": int(len(trial_results)),
        "recommended_full_config": recommended_full_config,
        "recommended_trial": compact_trials[0] if compact_trials else None,
        "alternative_trials": compact_trials[1:3],
        "tradeoff_plot_path": tradeoff_plot_path,
        "trial_results_path": trial_results_path,
        "suggestion_confidence": suggestion["confidence"],
        "suggestion_confidence_reason": suggestion["confidence_reason"],
        "search_space": {
            name: values["candidates"]
            for name, values in suggestion["suggested_baseline_ranges"].items()
            if name
            in {
                "width",
                "depth",
                "batch_size",
                "learning_rate",
                "dropout",
                "weight_decay",
            }
        },
    }

    (run_dir / "suggestion.json").write_text(json.dumps(suggestion, indent=2))
    (run_dir / "trial_results.json").write_text(json.dumps(ranked_trials, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return {
        **summary,
        "trial_results": ranked_trials,
    }


def _validate_search_config(config: SearchConfig, objective: str) -> None:
    # Fail fast on invalid fractions and unsupported objectives so the user gets a
    # clear error before any expensive work starts.
    if config.trial_count <= 0:
        raise ValueError("trial_count must be at least 1.")
    if config.epochs_per_trial <= 0:
        raise ValueError("epochs_per_trial must be at least 1.")
    if config.patience_per_trial is not None and config.patience_per_trial <= 0:
        raise ValueError("patience_per_trial must be at least 1 when provided.")
    if not 0.0 < config.train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0.0 <= config.val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")
    if config.train_frac + config.val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must leave room for a test split.")
    if objective not in {"balanced", "best_accuracy", "fastest_acceptable"}:
        raise ValueError(f"Unsupported objective: {config.objective}")


def _normalize_objective(objective: str) -> str:
    # Support a few friendly spellings so the CLI can be forgiving.
    normalized = objective.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "balanced": "balanced",
        "balanced_accuracy_and_time": "balanced",
        "best_accuracy": "best_accuracy",
        "fastest_acceptable": "fastest_acceptable",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported objective: {objective}")
    return aliases[normalized]


def _score_formula_description(objective: str) -> str:
    return {
        "balanced": "score = 0.45 * normalized validation MAE + 0.35 * normalized runtime + 0.20 * normalized best validation loss",
        "best_accuracy": "score = 0.65 * normalized validation MAE + 0.35 * normalized best validation loss",
        "fastest_acceptable": "score = 0.70 * normalized runtime + 0.20 * normalized validation MAE + 0.10 * normalized best validation loss, with an extra penalty for validation MAE worse than 105% of the best trial",
    }[objective]


def _objective_label(objective: str) -> str:
    if objective == "balanced":
        return "Balanced accuracy and time"
    if objective == "best_accuracy":
        return "Best accuracy"
    return "Fastest acceptable"


def _resolve_patience(patience: int | None, epochs_per_trial: int) -> int:
    # Short search runs should still have some room for early stopping, but the
    # patience must never exceed the total epoch budget.
    if patience is not None:
        return min(max(patience, 1), epochs_per_trial)
    return max(5, min(epochs_per_trial, max(epochs_per_trial // 3, 8)))


def _build_candidate_entries(
    base_trial_config: TrainConfig,
    base_full_config: TrainConfig,
    ranges: dict[str, dict[str, Any]],
    trial_count: int,
) -> list[dict[str, Any]]:
    # Helper closures that move one step down or up along the suggested candidate tiers.
    def lower(name: str, current: Any) -> Any:
        return _neighbor_value(ranges.get(name, {}), current, direction=-1)

    def higher(name: str, current: Any) -> Any:
        return _neighbor_value(ranges.get(name, {}), current, direction=1)

    # These proposals are intentionally easy to explain: smaller model, larger model,
    # higher/lower learning rate, lighter/stronger regularization, and so on.
    proposals: list[tuple[str, dict[str, Any]]] = [
        ("base", {}),
        (
            "smaller_model",
            {
                "width": lower("width", base_trial_config.width),
                "depth": lower("depth", base_trial_config.depth),
            },
        ),
        (
            "larger_model",
            {
                "width": higher("width", base_trial_config.width),
                "depth": higher("depth", base_trial_config.depth),
            },
        ),
        ("lower_lr", {"learning_rate": lower("learning_rate", base_trial_config.learning_rate)}),
        ("higher_lr", {"learning_rate": higher("learning_rate", base_trial_config.learning_rate)}),
        (
            "lighter_regularization",
            {
                "dropout": lower("dropout", base_trial_config.dropout),
                "weight_decay": lower("weight_decay", base_trial_config.weight_decay),
            },
        ),
        (
            "stronger_regularization",
            {
                "dropout": higher("dropout", base_trial_config.dropout),
                "weight_decay": higher("weight_decay", base_trial_config.weight_decay),
            },
        ),
        ("smaller_batch", {"batch_size": lower("batch_size", base_trial_config.batch_size)}),
        ("larger_batch", {"batch_size": higher("batch_size", base_trial_config.batch_size)}),
    ]

    entries: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for label, updates in proposals:
        entry = _candidate_entry(label, updates, base_trial_config, base_full_config)
        signature = _config_signature(entry["trial_config"])
        if signature in seen:
            # Skip duplicates when the suggested tier sits at an edge and "lower" or
            # "higher" collapses back to the current value.
            continue
        seen.add(signature)
        entries.append(entry)
        if len(entries) >= trial_count:
            return entries

    fallback_candidates = _fallback_candidate_updates(base_trial_config, ranges)
    for index, updates in enumerate(fallback_candidates, start=1):
        entry = _candidate_entry(f"combo_{index:02d}", updates, base_trial_config, base_full_config)
        signature = _config_signature(entry["trial_config"])
        if signature in seen:
            continue
        seen.add(signature)
        entries.append(entry)
        if len(entries) >= trial_count:
            break
    return entries


def _candidate_entry(
    label: str,
    updates: dict[str, Any],
    base_trial_config: TrainConfig,
    base_full_config: TrainConfig,
) -> dict[str, Any]:
    return {
        "label": label,
        "trial_config": _apply_updates(base_trial_config, updates),
        "full_config": _apply_updates(base_full_config, updates),
    }


def _apply_updates(config: TrainConfig, updates: dict[str, Any]) -> TrainConfig:
    payload = asdict(config)
    payload.update(updates)
    return TrainConfig(**payload)


def _config_signature(config: TrainConfig) -> tuple[Any, ...]:
    # The signature focuses only on the dimensions the search actually varies.
    return (
        config.width,
        config.depth,
        config.batch_size,
        float(config.learning_rate),
        float(config.dropout),
        float(config.weight_decay),
    )


def _neighbor_value(range_info: dict[str, Any], current: Any, direction: int) -> Any:
    # Walk one step left or right inside the ordered candidate list, clamping to the ends.
    candidates = list(range_info.get("candidates", [current]))
    if current not in candidates:
        candidates.append(current)
        candidates = sorted(candidates, key=float)
    index = candidates.index(current)
    index = max(0, min(len(candidates) - 1, index + direction))
    return candidates[index]


def _fallback_candidate_updates(base_config: TrainConfig, ranges: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    # If the hand-written proposals were not enough to fill the requested trial count,
    # generate simple multi-axis combinations ranked from "closest to base" outward.
    axes = {
        "width": list(ranges.get("width", {}).get("candidates", [base_config.width])),
        "depth": list(ranges.get("depth", {}).get("candidates", [base_config.depth])),
        "learning_rate": list(ranges.get("learning_rate", {}).get("candidates", [base_config.learning_rate])),
        "dropout": list(ranges.get("dropout", {}).get("candidates", [base_config.dropout])),
        "batch_size": list(ranges.get("batch_size", {}).get("candidates", [base_config.batch_size])),
        "weight_decay": list(ranges.get("weight_decay", {}).get("candidates", [base_config.weight_decay])),
    }
    base_values = {name: getattr(base_config, name) for name in axes}
    index_maps = {name: {value: idx for idx, value in enumerate(values)} for name, values in axes.items()}
    ranked: list[tuple[int, float, float, dict[str, Any]]] = []
    for combo in product(*axes.values()):
        updates: dict[str, Any] = {}
        changed = 0
        distance = 0.0
        combo_values = dict(zip(axes, combo, strict=True))
        for name, value in combo_values.items():
            if value != base_values[name]:
                updates[name] = value
                changed += 1
                distance += abs(index_maps[name][value] - index_maps[name][base_values[name]])
        if not updates:
            continue
        complexity = float(combo_values["width"] * combo_values["depth"])
        ranked.append((changed, distance, complexity, updates))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [updates for _, _, _, updates in ranked]


def _score_trials(trials: list[dict[str, Any]], objective: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # All objectives are built from the same normalized ingredients so the tradeoff
    # plot remains interpretable across search modes.
    val_losses = [trial["best_val_loss"] for trial in trials]
    val_mae = [trial["average_val_mae"] for trial in trials]
    runtimes = [trial["runtime_seconds"] for trial in trials]
    loss_norm = _min_max_normalize(val_losses)
    mae_norm = _min_max_normalize(val_mae)
    runtime_norm = _min_max_normalize(runtimes)
    best_mae = min(val_mae)
    acceptable_mae = best_mae * 1.05

    description = {
        "balanced": "score = 0.45 * normalized validation MAE + 0.35 * normalized runtime + 0.20 * normalized best validation loss",
        "best_accuracy": "score = 0.65 * normalized validation MAE + 0.35 * normalized best validation loss",
        "fastest_acceptable": "score = 0.70 * normalized runtime + 0.20 * normalized validation MAE + 0.10 * normalized best validation loss, with an extra penalty for validation MAE worse than 105% of the best trial",
    }[objective]

    ranked: list[dict[str, Any]] = []
    for index, trial in enumerate(trials):
        if objective == "balanced":
            score = 0.45 * mae_norm[index] + 0.35 * runtime_norm[index] + 0.20 * loss_norm[index]
        elif objective == "best_accuracy":
            score = 0.65 * mae_norm[index] + 0.35 * loss_norm[index]
        else:
            # "Fastest acceptable" prefers short runtime, but penalizes trials whose
            # validation MAE drifts too far above the best observed accuracy.
            penalty = max(0.0, trial["average_val_mae"] / acceptable_mae - 1.0)
            score = 0.70 * runtime_norm[index] + 0.20 * mae_norm[index] + 0.10 * loss_norm[index] + 2.0 * penalty
        enriched = dict(trial)
        enriched["score"] = float(score)
        enriched["score_breakdown"] = {
            "normalized_best_val_loss": float(loss_norm[index]),
            "normalized_average_val_mae": float(mae_norm[index]),
            "normalized_runtime": float(runtime_norm[index]),
        }
        ranked.append(enriched)

    ranked.sort(key=lambda trial: (trial["score"], trial["average_val_mae"], trial["runtime_seconds"]))
    for rank, trial in enumerate(ranked, start=1):
        trial["rank"] = rank
    return ranked, {"description": description, "acceptable_mae_threshold": acceptable_mae}


def _min_max_normalize(values: list[float]) -> list[float]:
    # Normalize only relative to the current set of trials. When all values are the
    # same, return zeros so no metric artificially dominates the score.
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-12:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def _compact_trial(trial: dict[str, Any]) -> dict[str, Any]:
    # The CLI and summary JSON surface this smaller record instead of the full history.
    return {
        "rank": trial["rank"],
        "label": trial["label"],
        "best_val_loss": trial["best_val_loss"],
        "average_val_mae": trial["average_val_mae"],
        "runtime_seconds": trial["runtime_seconds"],
        "epochs_completed": trial["epochs_completed"],
        "score": trial["score"],
        "recommended_full_config": trial["recommended_full_config"],
    }


def _plot_tradeoff(trials: list[dict[str, Any]], path: Path, objective: str) -> None:
    # Import matplotlib lazily so users who never request plots do not pay the import
    # cost during module load.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    runtimes = [trial["runtime_seconds"] for trial in trials]
    mae = [trial["average_val_mae"] for trial in trials]
    scores = [trial["score"] for trial in trials]
    scatter = ax.scatter(runtimes, mae, c=scores, cmap="viridis_r", s=110, edgecolors="black", linewidths=0.6)
    for trial in trials:
        ax.annotate(
            str(trial["rank"]),
            (trial["runtime_seconds"], trial["average_val_mae"]),
            textcoords="offset points",
            xytext=(6, 4),
        )
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Validation Average MAE")
    ax.set_title(f"Quick Search Tradeoff ({_objective_label(objective)})")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Score (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
