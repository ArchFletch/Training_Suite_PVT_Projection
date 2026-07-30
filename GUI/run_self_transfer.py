"""Command-line entry point for the self-transfer experiment.

The transfer logic is implemented in `xfmr_v2.runner`. This script only exposes
that workflow as a shell command and optionally opens the generated plots in VS Code.
"""

from __future__ import annotations

import argparse
import json

from xfmr_v2.data import CACHE_PATH
from xfmr_v2.runner import TransferConfig, open_in_vscode, run_self_transfer


def main() -> None:
    # `TransferConfig` is the source of truth for defaults. Pulling them from the
    # dataclass avoids duplicating numbers in two places.
    defaults = TransferConfig.__dataclass_fields__
    parser = argparse.ArgumentParser(description="Run standalone self-transfer learning for the v2 XFMR project.")
    parser.add_argument("--cache-path", default=str(CACHE_PATH))
    parser.add_argument("--output-dir", default=defaults["output_dir"].default)
    parser.add_argument("--model-type", default=defaults["model_type"].default)
    parser.add_argument("--width", type=int, default=defaults["width"].default)
    parser.add_argument("--depth", type=int, default=defaults["depth"].default)
    parser.add_argument("--train-frac", type=float, default=defaults["train_frac"].default)
    parser.add_argument("--val-frac", type=float, default=defaults["val_frac"].default)
    parser.add_argument("--iterations", type=int, default=defaults["iterations"].default)
    parser.add_argument("--num-bands", type=int, default=defaults["num_bands"].default)
    parser.add_argument("--transfer-epochs", type=int, default=defaults["transfer_epochs"].default)
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"].default)
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"].default)
    parser.add_argument("--seed", type=int, default=defaults["seed"].default)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--open-plots-in-vscode", action="store_true")
    args = parser.parse_args()

    # Build the configuration object exactly once, then hand it off to the runner.
    summary = run_self_transfer(
        TransferConfig(
            cache_path=args.cache_path,
            output_dir=args.output_dir,
            model_type=args.model_type,
            width=args.width,
            depth=args.depth,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            iterations=args.iterations,
            num_bands=args.num_bands,
            transfer_epochs=args.transfer_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            use_amp=not args.disable_amp,
        )
    )
    if args.open_plots_in_vscode:
        # Open the most useful diagnostic plots from the finished run. This is a small
        # convenience layer for interactive use and has no effect on the saved results.
        open_in_vscode(
            [
                summary["frequency_mae_plot_path"],
                summary["average_mae_plot_path"],
                summary["band_mae_plot_path"],
                summary["final_average_mae_plot_path"],
            ]
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
