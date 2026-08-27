"""Command-line entry point for training the baseline model.

The heavy lifting happens inside `xfmr_v2.runner.train_baseline`. This file is the
shell-facing wrapper that turns CLI arguments into a `TrainConfig` and prints a
small JSON summary of the finished run.

Configuration precedence (lowest to highest):

1. `TrainConfig` dataclass defaults
2. ``--config-json PATH`` — a full ``TrainConfig`` dict, or a suggest/search summary
   that embeds one under ``recommended_full_config`` / ``suggested_baseline_config``
3. explicit command-line flags

This lets the CLI train any architecture recommended by ``suggest_initial_settings.py``
or ``quick_search.py`` (model_type / width / depth / learning_rate / ...), not just the
dataclass defaults.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields

from xfmr_v2.data import CACHE_PATH, DATA_ROOT
from xfmr_v2.runner import (
    LOSS_FUNCTIONS,
    SCHEDULER_TYPES,
    TrainConfig,
    open_in_vscode,
    train_baseline,
)

# Names the merge step is allowed to copy out of a --config-json payload, so stray
# summary keys (run_dir, history, ...) are ignored rather than crashing TrainConfig.
_TRAINCONFIG_FIELDS = {f.name for f in fields(TrainConfig)}


def _load_config_json(path: str) -> dict:
    """Load a TrainConfig-shaped dict from a JSON file.

    Accepts either a bare ``TrainConfig`` dict or a richer summary produced by
    ``quick_search.py`` / ``suggest_initial_settings.py``, pulling the embedded config
    out of the well-known keys when present.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"--config-json {path!r} must contain a JSON object.")
    for key in ("recommended_full_config", "suggested_baseline_config"):
        embedded = data.get(key)
        if isinstance(embedded, dict):
            return embedded
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the v2 XFMR baseline.")
    parser.add_argument(
        "--config-json",
        default=None,
        help="Path to a JSON file holding a full TrainConfig (or a suggest/search summary "
        "containing recommended_full_config / suggested_baseline_config). Explicit flags "
        "below override values loaded here.",
    )
    # Data + cache + output locations.
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--input-feature-path", default=None)
    parser.add_argument("--ground-truth-data-dir", default=None)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--output-dir", default=None)
    # Architecture — these are the knobs the bare CLI previously could not set.
    parser.add_argument(
        "--model-type",
        default=None,
        help="SpectraNet, SpectraHydra, or SpectraHydraProj (legacy FlatMLP / CTLE_MLP also accepted).",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    # SpectraHydraProj only.
    parser.add_argument(
        "--projection-columns",
        default=None,
        help="Comma-separated input-feature names of the PVT corner columns fed to the "
        "learned projection (SpectraHydraProj only), e.g. 'Temp_C,VDD,proc_tt,proc_ff'.",
    )
    parser.add_argument("--projection-dim", type=int, default=None, help="Corner-embedding width (SpectraHydraProj only).")
    # Optimization / schedule.
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", default=None, choices=list(SCHEDULER_TYPES))
    parser.add_argument("--loss-function", default=None, choices=list(LOSS_FUNCTIONS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    # Split.
    parser.add_argument("--train-frac", type=float, default=None)
    parser.add_argument("--val-frac", type=float, default=None)
    parser.add_argument(
        "--split-design-columns",
        default=None,
        help="Comma-separated names of the columns that IDENTIFY a design (the geometry parameters). Every other column is treated as corner-varying, so the design-level split cannot be silently defeated by a derived corner column the way --split-corner-columns can. Prefer this flag.",
    )
    parser.add_argument(
        "--split-corner-columns",
        default=None,
        help="Comma-separated PVT corner column names enabling the design-level split: "
        "rows identical in every other input column are one design and all of its "
        "corner rows stay in the same train/val/test fold (prevents corner-row "
        "leakage between splits). Works with every model type. Pass '' to disable "
        "a value coming from --config-json.",
    )
    parser.add_argument(
        "--eval-dataset-path",
        default=None,
        help="Optional .npz file (features, targets, feature_names — a dataset "
        "bundle.npz qualifies) the best checkpoint is scored on after training, "
        "e.g. extra held-out designs. Reported as external_* summary fields. "
        "Pass '' to disable a value coming from --config-json.",
    )
    # Runtime.
    parser.add_argument("--device", default=None, help="e.g. cuda:0, cuda:1, cpu. None auto-selects.")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--open-plots-in-vscode", action="store_true")
    return parser


def resolve_train_config(args: argparse.Namespace) -> TrainConfig:
    """Merge dataclass defaults, an optional ``--config-json``, and explicit flags."""
    # Layer 1: dataclass defaults. Null the data-source/cache fields so the packaged
    # Windows default data_root does not leak in and shadow explicit inputs below.
    merged = asdict(TrainConfig())
    for key in ("data_root", "input_feature_path", "ground_truth_data_dir", "cache_path"):
        merged[key] = None

    # Layer 2: values from --config-json (only recognized TrainConfig fields).
    if args.config_json:
        loaded = _load_config_json(args.config_json)
        merged.update({k: v for k, v in loaded.items() if k in _TRAINCONFIG_FIELDS})

    # Layer 3: explicit command-line flags (None == "not provided", so it does not
    # clobber a value supplied by --config-json or a dataclass default).
    cli_overrides = {
        "data_root": args.data_root,
        "input_feature_path": args.input_feature_path,
        "ground_truth_data_dir": args.ground_truth_data_dir,
        "cache_path": args.cache_path,
        "output_dir": args.output_dir,
        "model_type": args.model_type,
        "width": args.width,
        "depth": args.depth,
        "projection_columns": (
            [name.strip() for name in args.projection_columns.split(",") if name.strip()]
            if args.projection_columns is not None
            else None
        ),
        "projection_dim": args.projection_dim,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "loss_function": args.loss_function,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "split_corner_columns": (
            [name.strip() for name in args.split_corner_columns.split(",") if name.strip()]
            if args.split_corner_columns is not None
            else None
        ),
        "split_design_columns": (
            [name.strip() for name in args.split_design_columns.split(",") if name.strip()]
            if args.split_design_columns is not None
            else None
        ),
        "eval_dataset_path": args.eval_dataset_path,
        "device": args.device,
    }
    merged.update({k: v for k, v in cli_overrides.items() if v is not None})
    if args.disable_amp:
        merged["use_amp"] = False

    # Only fall back to the packaged default dataset root when no data source at all
    # was provided (matches the original wrapper behavior).
    if not merged.get("data_root") and not merged.get("input_feature_path") and not merged.get("ground_truth_data_dir"):
        merged["data_root"] = str(DATA_ROOT)
    if not merged.get("cache_path"):
        merged["cache_path"] = str(CACHE_PATH)

    return TrainConfig(**merged)


def main() -> None:
    args = build_parser().parse_args()
    config = resolve_train_config(args)

    # Delegate the actual training run to the runner module.
    summary = train_baseline(config)
    if args.open_plots_in_vscode:
        # These two plots are the quickest way to judge whether training behaved well:
        # one shows the error curve across frequency and the other shows the average error.
        # Absent when the run kept no test fold; then there is nothing to open.
        plot_paths = [
            summary[key]
            for key in ("test_frequency_mae_plot_path", "average_test_mae_plot_path")
            if key in summary
        ]
        if plot_paths:
            open_in_vscode(plot_paths)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
