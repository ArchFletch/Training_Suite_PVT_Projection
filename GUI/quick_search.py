"""Command-line entry point for the short hyperparameter search.

This wrapper does not implement search logic itself. Its job is to:
- collect CLI arguments
- build a `SearchConfig`
- call the search module
- print a compact JSON summary that is convenient for humans and scripts
"""

from __future__ import annotations

import argparse
import json

from xfmr_v2.atomic_json import dumps_json
from xfmr_v2.search import SearchConfig, quick_hyperparameter_search


def main() -> None:
    # Instantiate defaults once so the parser stays synchronized with the dataclass
    # that the search code actually consumes.
    defaults = SearchConfig()
    parser = argparse.ArgumentParser(description="Run a quick hyperparameter search for the v2 XFMR baseline.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--input-feature-path", default=None)
    parser.add_argument("--ground-truth-data-dir", default=None)
    parser.add_argument("--cache-path", default=defaults.cache_path)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--train-frac", type=float, default=defaults.train_frac)
    parser.add_argument("--val-frac", type=float, default=defaults.val_frac)
    parser.add_argument("--max-samples", type=int, default=defaults.max_samples)
    parser.add_argument("--search-max-samples", type=int, default=defaults.search_max_samples)
    parser.add_argument("--model-type", default=defaults.model_type)
    parser.add_argument(
        "--projection-columns",
        default=None,
        help="Comma-separated input-feature names of the PVT corner columns fed to the "
        "learned projection (SpectraHydraProj only), e.g. 'Temp_C,VDD,proc_tt'.",
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=defaults.projection_dim,
        help="Corner-embedding width (SpectraHydraProj only).",
    )
    parser.add_argument(
        "--split-design-columns",
        default=None,
        help="Comma-separated names of the columns that IDENTIFY a design (the geometry parameters). Every other column is treated as corner-varying, so the design-level split cannot be silently defeated by a derived corner column the way --split-corner-columns can. Prefer this flag.",
    )
    parser.add_argument(
        "--split-corner-columns",
        default=None,
        help="Comma-separated PVT corner column names enabling the design-level split "
        "for every trial: rows identical in every other input column are one design "
        "and all of its corner rows stay in the same train/val/test fold (prevents "
        "corner-row leakage). Works with every model type.",
    )
    parser.add_argument("--trial-count", type=int, default=defaults.trial_count)
    parser.add_argument("--epochs-per-trial", type=int, default=defaults.epochs_per_trial)
    parser.add_argument(
        "--objective",
        default=defaults.objective,
        choices=["balanced", "best_accuracy", "fastest_acceptable"],
    )
    parser.add_argument("--variance-threshold", type=float, default=defaults.variance_threshold)
    parser.add_argument("--show-trial-progress", action="store_true")
    args = parser.parse_args()

    # Match the training CLI behavior: if no explicit input/output paths were given,
    # use the package's default dataset root.
    data_root = args.data_root
    if data_root is None and args.input_feature_path is None and args.ground_truth_data_dir is None:
        data_root = defaults.data_root

    # The search code returns a rich summary with full trial details. For the CLI we
    # intentionally print only the high-level fields a person is most likely to scan.
    summary = quick_hyperparameter_search(
        SearchConfig(
            data_root=data_root,
            input_feature_path=args.input_feature_path,
            ground_truth_data_dir=args.ground_truth_data_dir,
            cache_path=args.cache_path,
            output_dir=args.output_dir,
            seed=args.seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            max_samples=args.max_samples,
            search_max_samples=args.search_max_samples,
            model_type=args.model_type,
            projection_columns=(
                [name.strip() for name in args.projection_columns.split(",") if name.strip()]
                if args.projection_columns is not None
                else None
            ),
            projection_dim=args.projection_dim,
            split_corner_columns=(
                [name.strip() for name in args.split_corner_columns.split(",") if name.strip()]
                if args.split_corner_columns is not None
                else None
            ),
            split_design_columns=(
                [name.strip() for name in args.split_design_columns.split(",") if name.strip()]
                if args.split_design_columns is not None
                else None
            ),
            trial_count=args.trial_count,
            epochs_per_trial=args.epochs_per_trial,
            objective=args.objective,
            variance_threshold=args.variance_threshold,
            show_trial_progress=args.show_trial_progress,
        ),
        show_progress=True,
    )
    print(dumps_json(_cli_summary(summary)))


def _cli_summary(summary: dict) -> dict:
    # Keep the CLI output focused on the recommendation and the artifact paths.
    # The full trial table still exists on disk in the saved JSON files.
    return {
        "status": summary["status"],
        "run_dir": summary["run_dir"],
        "objective": summary["objective_label"],
        "epochs_per_trial": summary["epochs_per_trial"],
        "trial_count_completed": summary["trial_count_completed"],
        "recommended_trial": summary["recommended_trial"],
        "alternative_trials": summary["alternative_trials"],
        "recommended_full_config": summary["recommended_full_config"],
        "tradeoff_plot_path": summary["tradeoff_plot_path"],
        "trial_results_path": summary["trial_results_path"],
    }


if __name__ == "__main__":
    main()
