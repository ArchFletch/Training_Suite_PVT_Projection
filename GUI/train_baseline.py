"""Command-line entry point for training the baseline model.

The heavy lifting happens inside `xfmr_v2.runner.train_baseline`. This file is the
shell-facing wrapper that turns CLI arguments into a `TrainConfig` and prints a
small JSON summary of the finished run.
"""

from __future__ import annotations

import argparse
import json

from xfmr_v2.data import CACHE_PATH, DATA_ROOT
from xfmr_v2.runner import TrainConfig, open_in_vscode, train_baseline


def main() -> None:
    # Read defaults from the dataclass so there is one authoritative place where
    # training defaults are defined.
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train the v2 XFMR baseline.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--input-feature-path", default=None)
    parser.add_argument("--ground-truth-data-dir", default=None)
    parser.add_argument("--cache-path", default=str(CACHE_PATH))
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--max-samples", type=int, default=defaults.max_samples)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--open-plots-in-vscode", action="store_true")
    args = parser.parse_args()

    # Like the other CLI wrappers, default to the package dataset root only when
    # the caller did not explicitly provide data locations.
    data_root = args.data_root
    if data_root is None and args.input_feature_path is None and args.ground_truth_data_dir is None:
        data_root = str(DATA_ROOT)

    # Delegate the actual training run to the runner module.
    summary = train_baseline(
        TrainConfig(
            data_root=data_root,
            input_feature_path=args.input_feature_path,
            ground_truth_data_dir=args.ground_truth_data_dir,
            cache_path=args.cache_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            max_samples=args.max_samples,
            use_amp=not args.disable_amp,
        )
    )
    if args.open_plots_in_vscode:
        # These two plots are the quickest way to judge whether training behaved well:
        # one shows the error curve across frequency and the other shows the average error.
        open_in_vscode(
            [
                summary["test_frequency_mae_plot_path"],
                summary["average_test_mae_plot_path"],
            ]
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
