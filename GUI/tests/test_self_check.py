"""The release gate has to work, and has to fail loudly when a step breaks.

A launch-only smoke test passed on a bundle that could not train. `--self-check`
exists so the gate is the compiled binary doing real work; these tests keep the
gate itself honest, because a gate that cannot fail is worse than no gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xfmr_v2 import self_check

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_entry_point_wires_up_the_flag() -> None:
    """The gate is only reachable if launch_gui.py routes to it, and it has to be
    checked before Qt so it needs no display."""

    source = (REPO_ROOT / "launch_gui.py").read_text(encoding="utf-8")
    assert '"--self-check" in sys.argv[1:]' in source
    assert source.index("--self-check") < source.index("create_application()"), (
        "the self-check must run before the GUI is constructed, or it needs a display"
    )


def test_the_gate_passes_end_to_end() -> None:
    """Scan, suggest, two epochs, checkpoint, ONNX -- the steps the release gate
    lists, run for real. It trains, so it is the slowest test here."""

    lines: list[str] = []
    status = self_check.run_self_check(report=lines.append)
    output = "\n".join(lines)
    assert status == 0, output

    # Named individually: "SELF-CHECK PASSED" alone would also print if the step
    # list silently shrank to nothing.
    for step in (
        "construct an optimizer and take one step",
        "scan the dataset",
        "suggest initial settings",
        "train two epochs",
        "at least one epoch completed",
        "the run left a checkpoint",
        "export ONNX",
        "the export left a file",
    ):
        assert f"PASS  {step}" in output, f"step missing or failed: {step}\n{output}"
    assert "FAIL" not in output


def test_the_gate_reports_failure_and_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gate that cannot fail is decoration. Break the one step that the shipped
    bundle actually broke and confirm the gate says so."""

    import torch

    def exploding_adamw(*args: object, **kwargs: object) -> object:
        raise ModuleNotFoundError("No module named 'torch.testing._internal'")

    monkeypatch.setattr(torch.optim, "AdamW", exploding_adamw)

    lines: list[str] = []
    status = self_check.run_self_check(report=lines.append)
    output = "\n".join(lines)

    assert status == 1, output
    assert "FAIL  construct an optimizer and take one step" in output
    assert "No module named 'torch.testing._internal'" in output
    assert "SELF-CHECK FAILED" in output
