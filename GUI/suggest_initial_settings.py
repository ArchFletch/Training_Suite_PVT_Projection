"""Command-line entry point for scan-only hyperparameter suggestions.

This script is useful when we want a first-pass recommendation without paying the
cost of actually training the model. It simply wraps `xfmr_v2.suggest`.
"""

from __future__ import annotations

import argparse
import json

from xfmr_v2.atomic_json import dumps_json
from xfmr_v2.suggest import SuggestConfig, suggest_initial_settings


def main() -> None:
    # Reuse the dataclass defaults so the CLI and Python API stay in sync.
    defaults = SuggestConfig()
    parser = argparse.ArgumentParser(description="Suggest initial baseline and transfer settings without training.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--input-feature-path", default=None)
    parser.add_argument("--ground-truth-data-dir", default=None)
    parser.add_argument("--cache-path", default=defaults.cache_path)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--train-frac", type=float, default=defaults.train_frac)
    parser.add_argument("--val-frac", type=float, default=defaults.val_frac)
    parser.add_argument(
        "--split-design-columns",
        default=None,
        help="Comma-separated names of the columns that IDENTIFY a design (the geometry parameters). Every other column is treated as corner-varying, so the design-level split cannot be silently defeated by a derived corner column the way --split-corner-columns can. Prefer this flag.",
    )
    parser.add_argument(
        "--split-corner-columns",
        default=None,
        help="Comma-separated PVT corner column names enabling the design-level split "
        "for the diagnostics: rows identical in every other input column are one "
        "design and all of its corner rows stay in the same train/val/test fold, so "
        "the reported split sizes match a design-split training run.",
    )
    parser.add_argument("--max-samples", type=int, default=defaults.max_samples)
    parser.add_argument("--variance-threshold", type=float, default=defaults.variance_threshold)
    args = parser.parse_args()

    # Fall back to the package default dataset root only when the caller did not
    # explicitly point to input/output locations.
    data_root = args.data_root
    if data_root is None and args.input_feature_path is None and args.ground_truth_data_dir is None:
        data_root = defaults.data_root

    # Print the full suggestion result because the output is already concise and
    # each field helps explain why the heuristic made its recommendation.
    summary = suggest_initial_settings(
        SuggestConfig(
            data_root=data_root,
            input_feature_path=args.input_feature_path,
            ground_truth_data_dir=args.ground_truth_data_dir,
            cache_path=args.cache_path,
            seed=args.seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
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
            max_samples=args.max_samples,
            variance_threshold=args.variance_threshold,
        )
    )
    print(dumps_json(summary))


if __name__ == "__main__":
    main()
