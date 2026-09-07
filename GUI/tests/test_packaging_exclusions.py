"""Every module the bundle excludes must be one the app never imports.

This test exists because getting it wrong is expensive and near-silent.
`--noinclude-unittest-mode=nofollow` looked obviously safe -- unittest is test
scaffolding -- and produced a bundle that aborted during startup with SIGABRT and
not one byte on stdout or stderr. torch imports unittest for real, from
torch/utils/_config_module.py by way of torch/_inductor/config.py. The cost of
finding that out was a forty-minute compile plus a gdb session; the cost of this
test is a few seconds.

It reads the exclusions out of the Nuitka spec rather than restating them, so a
new exclusion is covered the moment someone adds it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = REPO_ROOT / "packaging" / "gui" / "pysidedeploy.spec.in"

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _declared_exclusions() -> list[str]:
    """Module names the spec tells Nuitka not to compile in."""

    text = SPEC.read_text(encoding="utf-8")
    extra_args = next(
        line for line in text.splitlines() if line.strip().startswith("extra_args")
    )

    excluded = re.findall(r"--nofollow-import-to=([\w.]+)", extra_args)
    # The *-mode=nofollow switches drop a whole family rather than one name.
    for family in re.findall(r"--noinclude-(\w+)-mode=nofollow", extra_args):
        if family not in {"qt-translations", "qt"}:
            excluded.append(family)
    return sorted(set(excluded))


def _import_everything_the_app_can_reach() -> set[str]:
    """Load the app's full surface in a CLEAN SUBPROCESS and report sys.modules.

    A subprocess, not this interpreter: pytest itself is in sys.modules here, so
    an in-process check reports `pytest` as imported by the app and fails on its
    own harness. The bundle does not run under a test runner, so measuring in one
    answers the wrong question.

    The ONNX export is part of the surface on purpose. torch routes it through
    dynamo, which pulls in modules a plain `import torch` never touches.
    """

    probe = textwrap.dedent(
        """
        import json, os, sys, warnings
        warnings.filterwarnings("ignore")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot
        import numpy, onnx, onnxruntime, onnxscript, pyqtgraph, torch
        import xfmr_v2.data, xfmr_v2.export_onnx, xfmr_v2.gui_window
        import xfmr_v2.runner, xfmr_v2.search

        torch.onnx.export(torch.nn.Linear(3, 4), (torch.zeros(1, 3),), os.devnull)
        json.dump(sorted(sys.modules), sys.stdout)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
    )
    if result.returncode != 0:
        pytest.fail(f"the import probe itself failed:\n{result.stderr[-2000:]}")
    return set(json.loads(result.stdout))


def test_the_spec_declares_at_least_one_exclusion() -> None:
    """A guard on the guard: if the parse silently returns nothing, the real
    test below passes vacuously and proves nothing."""

    assert _declared_exclusions(), f"parsed no exclusions out of {SPEC.name}"


def test_no_excluded_module_is_imported_at_runtime() -> None:
    loaded = _import_everything_the_app_can_reach()

    unsafe: dict[str, list[str]] = {}
    for name in _declared_exclusions():
        hits = sorted(m for m in loaded if m == name or m.startswith(f"{name}."))
        if hits:
            unsafe[name] = hits[:4]

    assert not unsafe, (
        "these are excluded from the bundle but imported at runtime, so the "
        "packaged build would fail where the source checkout works: "
        + "; ".join(f"{name} (loaded: {', '.join(hits)})" for name, hits in unsafe.items())
    )


@pytest.mark.parametrize(
    "name, reason",
    [
        ("unittest", "torch imports it from torch/utils/_config_module.py"),
        ("torch.testing", "torch imports _creation/_comparison/_utils at import time"),
        ("sympy.polys.polyquinticconst", "sympy.polys.polyroots imports it at module scope"),
    ],
)
def test_known_traps_are_not_excluded(name: str, reason: str) -> None:
    """Three exclusions that look safe and are not. Named individually so a
    future edit that re-adds one fails with the reason attached."""

    assert name not in _declared_exclusions(), f"{name} must stay in the bundle: {reason}"
