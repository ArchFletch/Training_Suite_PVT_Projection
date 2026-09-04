"""Shared pytest fixtures for the dataset and training pipeline tests.

The tests in this repository work with tiny synthetic datasets so they stay fast
and deterministic. This fixture file creates those synthetic raw files in the
same shape that the real data loader expects.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def isolated_app_state(tmp_path_factory, monkeypatch):
    """Point every writable app location at a temp dir for the whole test session.

    In source mode the GUI keeps its session, license config, and default output
    under <repo>/GUI/artifacts, so any test that constructs the window or saves a
    config writes over the developer's real GUI state -- which has happened: a run
    picked up a pytest tmp_path as its output folder and cache. Individual tests
    used to guard this by monkeypatching save_last_session, which only protects the
    tests that remember to. Redirect the paths themselves instead.

    ``app_paths`` itself is left alone so its own tests still see the real policy.
    """
    from xfmr_v2 import app_paths

    root = tmp_path_factory.mktemp("app_state")
    gui_state = root / "gui"
    gui_state.mkdir()
    paths = app_paths.RuntimePaths(
        mode="source",
        config_dir=gui_state,
        state_dir=gui_state,
        cache_dir=root / "cache",
        log_dir=root / "logs",
        default_output_dir=root / "output",
        license_client_path=gui_state / app_paths.LICENSE_CLIENT_FILENAME,
        last_session_path=gui_state / app_paths.LAST_SESSION_FILENAME,
    )
    real_current_runtime_paths = app_paths.current_runtime_paths

    def redirected_runtime_paths(*, platform=None, env=None):
        """Temp-dir locations, but the REAL source/packaged decision.

        The stub used to hard-code mode="source" and take no arguments, so a
        test could not exercise behaviour that depends on the runtime mode, and
        any caller passing ``env`` hit a TypeError.
        """
        from dataclasses import replace

        mode = real_current_runtime_paths(platform=platform, env=env).mode
        return replace(paths, mode=mode)

    for module in ("xfmr_v2.gui_backend", "xfmr_v2.gui_window", "xfmr_v2.licensing.client_config"):
        try:
            __import__(module)
        except ImportError:
            continue  # optional dependency (e.g. PySide6 absent)
        import sys
        monkeypatch.setattr(
            sys.modules[module], "current_runtime_paths", redirected_runtime_paths, raising=False
        )
    return paths


def _write_touchstone_sample(path: Path, sample_index: int) -> None:
    """Write one small two-port Touchstone file with predictable values.

    The numbers are synthetic but intentionally simple, which makes it easy for
    the tests to verify specific channels after parsing.
    """

    lines = [
        "! synthetic touchstone sample\n",
        "# ghz s ri r 50\n",
        "[number of ports] 2\n",
        "[matrix format] full\n",
    ]
    for freq_index, frequency_ghz in enumerate((1.0, 2.0, 3.0, 4.0), start=1):
        # Vary the responses with both sample index and frequency index so the cache
        # loader cannot accidentally pass while reading a constant signal.
        base = 0.05 * sample_index + 0.01 * freq_index
        s11 = complex(base, -0.5 * base)
        s12 = complex(base / 3.0, base / 4.0)
        s21 = complex(base / 4.0, -base / 5.0)
        s22 = complex(-base / 2.0, base)
        row = [
            frequency_ghz,
            s11.real,
            s11.imag,
            s12.real,
            s12.imag,
            s21.real,
            s21.imag,
            s22.real,
            s22.imag,
        ]
        lines.append(" ".join(f"{value:.8f}" for value in row) + "\n")
    lines.append("[end]\n")
    path.write_text("".join(lines), encoding="utf-8")


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> dict[str, Path]:
    """Create a minimal dataset tree that exercises the full cache pipeline."""

    root = tmp_path / "synthetic_dataset"
    input_dir = root / "input"
    output_dir = root / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    # The README schema is what the production loader consumes, so the tests build
    # it the same way instead of hard-coding schema objects in Python.
    schema = {
        "dataset_name": "SyntheticTouchstoneDataset",
        "input_feature": {
            "columns": ["x", "y", "const", "sample_id"],
            "feature_columns": ["x", "y", "const"],
            "sample_id_column": "sample_id",
        },
        "ground_truth": {
            "source": "per_sample",
            "file_extension": ".s2p",
            "ground_truth_parameters": ["S11", "S12"],
            "ground_truth_parts": ["re", "im"],
            "drop_first_frequency": True,
        },
    }
    readme = root / "README.md"
    readme.write_text(
        "# Synthetic Dataset\n\n```json\n" + json.dumps(schema, indent=2) + "\n```\n",
        encoding="utf-8",
    )

    # The input-feature file mirrors the real dataset format: one Python literal row per line.
    rows: list[str] = []
    for sample_id in range(1, 11):
        x = float(sample_id)
        y = float((sample_id % 4) - 1.5)
        const = 7.0
        rows.append(str([x, y, const, sample_id]))
        _write_touchstone_sample(output_dir / f"{sample_id}.s2p", sample_index=sample_id)
    (input_dir / "log.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Pre-build the cache directly so tests have one to load. The production app
    # builds caches via build_cache_from_dataset (folder auto-detection); this
    # fixture writes the cache contents explicitly to pin a small, fixed
    # S11+S12 channel set (auto-detection would also emit S22).
    from xfmr_v2.data import _save_cache
    cache_path = tmp_path / "synthetic_cache.npz"

    # Mirror _write_touchstone_sample exactly, dropping the first frequency point.
    features = np.array(
        [[float(sid), float((sid % 4) - 1.5), 7.0] for sid in range(1, 11)],
        dtype=np.float32,
    )
    targets_list = []
    for sid in range(1, 11):
        channels = [[], [], [], []]  # S11_re, S11_im, S12_re, S12_im
        for freq_index in range(2, 5):  # frequency index 1 is dropped
            base = 0.05 * sid + 0.01 * freq_index
            channels[0].append(base)
            channels[1].append(-0.5 * base)
            channels[2].append(base / 3.0)
            channels[3].append(base / 4.0)
        targets_list.append(np.array(channels, dtype=np.float32))
    targets = np.stack(targets_list, axis=0)
    frequency_hz = np.array([2.0e9, 3.0e9, 4.0e9], dtype=np.float32)

    _save_cache(
        cache_path,
        dataset_root=root,
        dataset_name="SyntheticTouchstoneDataset",
        features=features,
        targets=targets,
        frequency_hz=frequency_hz,
        feature_names=["x", "y", "const"],
        channel_names=["S11_re", "S11_im", "S12_re", "S12_im"],
        target_names=["S11_re", "S11_im", "S12_re", "S12_im"],
        channel_units=["", "", "", ""],
        channel_transforms=["", "", "", ""],
        sweep_label="Frequency (GHz)",
    )

    return {
        "root": root,
        "readme": readme,
        "input_dir": input_dir,
        "input_file": input_dir / "log.txt",
        "output_dir": output_dir,
        "cache_path": cache_path,
    }
