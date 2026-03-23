# Tech Stack: MLP Training

## Recommended Stack

- Python 3.12
- NumPy
- PyTorch
- dataclasses for internal configs
- JSON for summaries and saved configs
- `.npz` for caches
- `.pt` for checkpoints
- Matplotlib for saved plots
- pytest for tests

## Why This Fits

- Python + PyTorch gives you full control over custom baseline and transfer loops.
- NumPy + `.npz` is a good fit for local reusable caches.
- dataclasses and JSON keep configs readable and easy to share across CLI and GUI.
- The workflow is still small enough that heavier experiment frameworks would add more complexity than value.

## Keep

- custom PyTorch training loops
- file-based artifacts
- CLI entrypoints per workflow
- lightweight config model

## Avoid for MVP

- PyTorch Lightning
- Hydra
- MLflow
- DVC
- large HPO/search infrastructure

## Likely Later Additions

- Pydantic at external boundaries if config payloads grow
- Optuna only if quick search becomes a real optimization product

## Bottom Line

Stay with **Python + NumPy + PyTorch + dataclasses + JSON + pytest**.

## References

- [PyTorch Docs](https://docs.pytorch.org/docs/stable/index.html)
- [NumPy Docs](https://numpy.org/doc/stable)
