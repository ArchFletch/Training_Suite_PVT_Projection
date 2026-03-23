"""Command-line entry point for building the local training cache.

This script is intentionally thin:
- it only translates command-line flags into Python values
- it delegates the real work to ``xfmr_v2.data.build_cache``
- it prints the returned summary as JSON so the result is easy to inspect

Keeping the wrapper small makes it easy to trace where the actual cache logic lives.
"""

from __future__ import annotations

import argparse
import json

from xfmr_v2.data import CACHE_PATH, DATA_ROOT, build_cache


def main() -> None:
    # Define the CLI surface. These flags mirror the `build_cache` function closely
    # so there is very little hidden behavior between the shell command and the Python API.
    parser = argparse.ArgumentParser(description="Build the v2 XFMR cache.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--input-feature-path", default=None)
    parser.add_argument("--ground-truth-data-dir", default=None)
    parser.add_argument("--cache-path", default=str(CACHE_PATH))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # If the caller did not provide any explicit data location, fall back to the
    # project default path baked into `xfmr_v2.data`.
    data_root = args.data_root
    if data_root is None and args.input_feature_path is None and args.ground_truth_data_dir is None:
        data_root = str(DATA_ROOT)

    # Print a machine-readable summary so shell users, tests, and GUI wrappers can
    # all consume the same output format.
    print(
        json.dumps(
            build_cache(
                data_root=data_root,
                cache_path=args.cache_path,
                overwrite=args.overwrite,
                max_samples=args.max_samples,
                input_feature_path=args.input_feature_path,
                ground_truth_data_dir=args.ground_truth_data_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
