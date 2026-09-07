"""Command-line entry point for building the local training cache.

This script is intentionally thin:
- it only translates command-line flags into Python values
- it delegates the real work to ``xfmr_v2.data.build_cache_from_dataset``
- it prints the returned summary as JSON so the result is easy to inspect

Keeping the wrapper small makes it easy to trace where the actual cache logic lives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xfmr_v2.atomic_json import json_safe
from xfmr_v2.data import CACHE_PATH, build_cache_from_dataset, load_existing_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the training cache from a dataset folder (format auto-detected: "
        "SPData/Touchstone or Cadence CSV)."
    )
    parser.add_argument("dataset_root", help="Folder containing the raw dataset.")
    parser.add_argument("--cache-path", default=str(CACHE_PATH))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild even when the cache file already exists.")
    args = parser.parse_args()

    def _print_progress(payload: dict) -> None:
        message = payload.get("message")
        if message:
            print(message, flush=True)

    if Path(args.cache_path).is_file() and not args.overwrite:
        summary = load_existing_cache(args.cache_path, progress_callback=_print_progress)
        print(f"Cache already exists at {args.cache_path}; pass --overwrite to rebuild.")
    else:
        summary = build_cache_from_dataset(
            args.dataset_root,
            args.cache_path,
            max_samples=args.max_samples,
            progress_callback=_print_progress,
        )

    # Print a machine-readable summary so shell users, tests, and GUI wrappers can
    # all consume the same output format.
    print(json.dumps(json_safe(summary), indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
