"""Lean XFMR modeling package for the v2 rewrite.

The package is split into a few focused modules:
- `dataset_schema`: parse README metadata that describes the dataset layout
- `data`: read raw files, build caches, and prepare train/validation/test splits
- `model`: define the neural network architecture
- `runner`: train and evaluate the baseline and self-transfer workflows
- `suggest` and `search`: recommend or explore hyperparameters
- `progress`: ship structured progress events to a GUI or other caller

This file is intentionally small because the project does not currently need
package-level exports beyond the module namespace itself.
"""
