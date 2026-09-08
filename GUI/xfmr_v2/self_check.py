"""End-to-end self-check that runs inside the packaged binary.

A launch-only smoke test does not catch the failure this exists for. The app
starts, checks out a licence, scans a dataset and produces suggestions perfectly
well while being unable to train: the shipped bundle excluded
``torch.testing._internal``, which ``torch/utils/checkpoint.py`` imports at
module scope, so the first ``torch.optim.AdamW`` raised ``ModuleNotFoundError``
one second into every run. Nothing before that step touches it.

Compiling is also not the check. Nuitka reports success for a bundle whose
missing module is only reached at runtime, so the gate has to be the compiled
binary doing the work, not the source tree doing it.

So this trains for real -- two epochs on a small synthetic dataset -- and
exports ONNX, then reports one line per step. Run it from the bundle:

    bin/mlp-training-studio --self-check

Exit status is 0 only if every step passed.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

_SEED = 20260907


def _write_synthetic_dataset(path: Path) -> None:
    """A dataset in the array form the app consumes, small enough to train fast.

    Six designs by four corners, so a design-level split has something to group;
    the numbers are meaningless, only the shapes and dtypes matter here.
    """

    import numpy as np

    rng = np.random.default_rng(_SEED)
    rows: list[list[float]] = []
    for design in range(6):
        for temp, vdd in ((-40.0, 0.9), (25.0, 1.0), (85.0, 1.1), (125.0, 1.2)):
            rows.append([float(design), float(design) * 0.5 + 1.0, temp, vdd])
    features = np.asarray(rows, dtype=np.float32)
    n_freq = 8
    targets = rng.standard_normal((len(features), 2, n_freq)).astype(np.float32)
    np.savez_compressed(
        path,
        features=features,
        targets=targets,
        frequency_hz=np.linspace(1e9, 4e9, n_freq).astype(np.float32),
        input_feature_names=np.asarray(["geom_a", "geom_b", "Temp_C", "VDD"]),
        target_names=np.asarray(["gain", "phase"]),
        channel_names=np.asarray(["gain", "phase"]),
        channel_units=np.asarray(["dB", "deg"]),
        channel_transforms=np.asarray(["", ""]),
        sweep_label=np.asarray("Frequency (GHz)"),
    )


def run_self_check(*, report: Callable[[str], None] = print) -> int:
    """Run the gate. Returns a process exit status."""

    failures: list[str] = []

    def step(name: str, fn: Callable[[], Any]) -> Any:
        try:
            value = fn()
        except BaseException:  # noqa: BLE001 - a gate reports, it does not propagate
            report(f"FAIL  {name}")
            for line in traceback.format_exc().rstrip().splitlines():
                report(f"        {line}")
            failures.append(name)
            return None
        report(f"PASS  {name}")
        return value

    report("mlp-training-studio self-check")
    report(f"  python  {sys.version.split()[0]}")
    report(f"  frozen  {'__compiled__' in globals() or hasattr(sys, 'frozen')}")

    torch = step("import torch", lambda: __import__("torch"))
    if torch is not None:
        report(f"  torch   {torch.__version__}")

    # The exact chain that the shipped bundle could not complete. Checked before
    # the training run so a failure here is unambiguous rather than buried in a
    # trainer traceback.
    def _optimizer() -> Any:
        import torch as t

        model = t.nn.Linear(4, 3)
        opt = t.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model(t.zeros(1, 4)).sum()
        loss.backward()
        opt.step()
        import torch.utils.checkpoint  # noqa: F401  - the module that pulled it in

        return opt

    step("construct an optimizer and take one step", _optimizer)

    with tempfile.TemporaryDirectory(prefix="mlp-self-check-") as tmp:
        root = Path(tmp)
        # The .npz IS the cache: scan_dataset loads an existing one rather than
        # building it, which is what the GUI does once Scan Dataset has run.
        cache = root / "self_check_cache.npz"
        step("write a synthetic dataset", lambda: _write_synthetic_dataset(cache))

        from .gui_backend import (
            build_suggest_result,
            export_model_to_onnx,
            run_training_workflow,
            scan_dataset,
        )
        from .runner import TrainConfig

        scan = step(
            "scan the dataset",
            lambda: scan_dataset(
                cache_path=str(cache), train_frac=0.7, val_frac=0.15, seed=_SEED
            ),
        )
        if scan is not None:
            report(f"  scan    status={scan.get('status')} freq_bins={scan.get('frequency_count')}")
            if scan.get("status") not in (None, "ok"):
                failures.append("scan did not report ok")

        step(
            "suggest initial settings",
            lambda: build_suggest_result(
                cache_path=str(cache),
                seed=_SEED,
                train_frac=0.7,
                val_frac=0.15,
                max_samples=None,
            ),
        )

        config = TrainConfig(
            cache_path=str(cache),
            output_dir=str(root / "runs"),
            epochs=2,
            batch_size=4,
            width=16,
            depth=2,
            train_frac=0.7,
            val_frac=0.15,
            seed=_SEED,
        )

        summary = step(
            "train two epochs",
            lambda: run_training_workflow(baseline_config=config, transfer_config=None),
        )
        baseline: dict[str, Any] = (summary or {}).get("baseline") or {}
        if summary is not None:
            # One record per finished epoch; the summary has no epoch counter.
            epochs_done = len(baseline.get("history") or [])
            report(
                f"  status  {summary.get('status')}  epochs={epochs_done}"
                f"  best_val_loss={baseline.get('best_val_loss')}"
                f"  train/val/test={baseline.get('train_samples')}/"
                f"{baseline.get('val_samples')}/{baseline.get('test_samples')}"
            )
            if summary.get("status") != "ok":
                failures.append("training did not report ok")
            # An "ok" status with zero epochs is the shape a silent no-op takes.
            if epochs_done < 1:
                report("FAIL  at least one epoch completed")
                failures.append("no epoch completed")
            else:
                report("PASS  at least one epoch completed")

        # train_baseline writes into a timestamped directory under output_dir and
        # reports it; the configured output_dir is the parent, not the run.
        run_dir = Path(baseline.get("run_dir") or (root / "runs"))
        checkpoint = run_dir / "best_model.pt"
        step(
            "the run left a checkpoint",
            lambda: checkpoint.stat().st_size or _raise(f"{checkpoint} is empty"),
        )

        onnx_path = root / "self_check.onnx"
        step(
            "export ONNX",
            lambda: export_model_to_onnx(
                checkpoint_path=str(checkpoint), out_path=str(onnx_path)
            ),
        )
        step(
            "the export left a file",
            lambda: onnx_path.stat().st_size or _raise(f"{onnx_path} is empty"),
        )

    if failures:
        report(f"\nSELF-CHECK FAILED: {len(failures)} step(s): {', '.join(failures)}")
        return 1
    report("\nSELF-CHECK PASSED")
    return 0


def _raise(message: str) -> None:
    raise AssertionError(message)
