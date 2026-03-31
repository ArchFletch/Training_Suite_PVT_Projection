"""Shared pytest fixtures for the dataset and training pipeline tests.

The tests in this repository work with tiny synthetic datasets so they stay fast
and deterministic. This fixture file creates those synthetic raw files in the
same shape that the real data loader expects.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


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

    # Pre-build the cache so tests can use it directly.
    from xfmr_v2.data import build_cache_from_loader
    cache_path = tmp_path / "synthetic_cache.npz"
    loader_code = '''
import ast, re, numpy as np
from pathlib import Path

_SI = {"f":1e-15,"p":1e-12,"n":1e-9,"u":1e-6,"m":1e-3,"K":1e3,"M":1e6,"G":1e9,"T":1e12}

def load_dataset(dataset_root, max_samples=None):
    root = Path(dataset_root)
    input_file = root / "input" / "log.txt"
    output_dir = root / "output"
    rows = [ast.literal_eval(line.strip()) for line in input_file.read_text().strip().splitlines()]
    if max_samples:
        rows = rows[:max_samples]
    features = np.array([[r[0], r[1], r[2]] for r in rows], dtype=np.float32)
    sample_ids = [int(r[3]) for r in rows]
    targets_list, freq_ref = [], None
    for sid in sample_ids:
        path = output_dir / f"{sid}.s2p"
        lines = [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("!") and not l.startswith("#") and not l.startswith("[")]
        data_rows = [list(map(float, l.split())) for l in lines]
        freq = np.array([r[0] for r in data_rows]) * 1e9
        s11_re = np.array([r[1] for r in data_rows])
        s11_im = np.array([r[2] for r in data_rows])
        s12_re = np.array([r[3] for r in data_rows])
        s12_im = np.array([r[4] for r in data_rows])
        if freq_ref is None:
            freq_ref = freq[1:]
        targets_list.append(np.stack([s11_re[1:], s11_im[1:], s12_re[1:], s12_im[1:]], axis=0))
    targets = np.stack(targets_list, axis=0).astype(np.float32)
    return {
        "features": features,
        "targets": targets,
        "sweep_axis": freq_ref.astype(np.float32),
        "feature_names": ["x", "y", "const"],
        "channel_names": ["S11_re", "S11_im", "S12_re", "S12_im"],
        "channel_units": ["", "", "", ""],
        "sweep_label": "Frequency (GHz)",
        "dataset_name": "SyntheticTouchstoneDataset",
    }
'''
    build_cache_from_loader(loader_code, str(root), str(cache_path))

    return {
        "root": root,
        "readme": readme,
        "input_dir": input_dir,
        "input_file": input_dir / "log.txt",
        "output_dir": output_dir,
        "cache_path": cache_path,
    }
