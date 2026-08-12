"""Integration-style tests for schema parsing, caching, suggestions, and search."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from xfmr_v2 import data, dataset_schema, search, suggest
from xfmr_v2.runner import TrainConfig
from xfmr_v2.search import SearchConfig
from xfmr_v2.suggest import SuggestConfig


def test_parse_dataset_readme_uses_parent_name_and_default_parts(tmp_path: Path) -> None:
    """A README without an explicit dataset name should fall back to the folder name."""

    dataset_root = tmp_path / "readme_defaults"
    dataset_root.mkdir()
    readme = dataset_root / "README.md"
    readme.write_text(
        """# Dataset

```json
{
  "input_feature": {
    "columns": ["x", "sample_id"],
    "feature_columns": ["x"],
    "sample_id_column": "sample_id"
  },
  "ground_truth": {
    "source": "per_sample",
    "file_extension": ".S2P",
    "ground_truth_parameters": ["S11"]
  }
}
```
""",
        encoding="utf-8",
    )

    schema = dataset_schema.parse_dataset_readme(readme)

    assert schema.dataset_name == "readme_defaults"
    assert schema.input_feature.columns == ("x", "sample_id")
    assert schema.input_feature.feature_columns == ("x",)
    assert schema.ground_truth.source == "per_sample"
    assert schema.ground_truth.file_extension == ".s2p"
    assert schema.ground_truth.ground_truth_parts == ("re", "im")
    assert schema.ground_truth.channel_names == ["S11_re", "S11_im"]


def test_parse_dataset_readme_rejects_duplicate_columns(tmp_path: Path) -> None:
    """Schema validation should fail early when input columns are ambiguous."""

    dataset_root = tmp_path / "bad_schema"
    dataset_root.mkdir()
    readme = dataset_root / "README.md"
    readme.write_text(
        """```json
{
  "input_feature": {
    "columns": ["x", "x", "sample_id"],
    "feature_columns": ["x"],
    "sample_id_column": "sample_id"
  },
  "ground_truth": {
    "source": "per_sample",
    "file_extension": ".s2p",
    "ground_truth_parameters": ["S11"]
  }
}
```
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Input-feature columns must be unique"):
        dataset_schema.parse_dataset_readme(readme)


def test_build_cache_from_dataset_autodetect_touchstone(synthetic_dataset: dict[str, Path], tmp_path: Path) -> None:
    """With no README, an SPData/Touchstone layout (log.txt + .sNp dir) is auto-detected."""
    import shutil

    auto_root = tmp_path / "auto_touchstone"
    sp_dir = auto_root / "SPData"
    sp_dir.mkdir(parents=True)
    for f in sorted(synthetic_dataset["output_dir"].glob("*.s2p")):
        shutil.copy(f, sp_dir / f.name)
    # log.txt at the root with a header naming the columns (no README present).
    rows = synthetic_dataset["input_file"].read_text(encoding="utf-8")
    (auto_root / "log.txt").write_text("# [x, y, const, index]\n" + rows, encoding="utf-8")

    summary = data.build_cache_from_dataset(str(auto_root), str(tmp_path / "auto_ts.npz"))

    assert summary["num_samples"] == 10
    assert summary["num_features"] == 3  # x, y, const (index dropped as the id column)
    assert summary["num_channels"] == 6  # 2-port upper-tri S11,S12,S22 x (re, im)
    assert summary["num_frequencies"] == 3  # 4 points minus the dropped first


def test_build_cache_from_dataset_autodetect_cadence(tmp_path: Path) -> None:
    """Cadence CSVs are auto-detected and channels sharing an axis are combined."""

    root = tmp_path / "ctle_auto"
    root.mkdir()

    def write_cadence(path: Path, freqs: list[str], base: float) -> None:
        lines = ["", "freq (Hz)   db(stuff)"]
        for s in range(2):  # two samples (distinct CS)
            lines += [f"CS = {80.98 + s}f", "LD = 191.6p", "M = 40", "RD = 167.2", "RS = 560.4", "MN = 13", "VCM   599m"]
            lines += [f"  {f}   {base + s + 0.1 * i}" for i, f in enumerate(freqs)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_cadence(root / "CTLE_gain.csv", ["1K", "2K", "3K"], -18.0)
    write_cadence(root / "CTLE_phase.csv", ["1K", "2K", "3K"], 90.0)
    write_cadence(root / "CTLE_hb_gain.csv", ["1m", "2m"], 5.0)  # different axis -> excluded

    summary = data.build_cache_from_dataset(str(root), str(tmp_path / "ctle_auto.npz"))

    assert summary["num_channels"] == 2  # gain + phase share the freq axis; hb_gain excluded
    assert summary["num_frequencies"] == 3
    assert summary["num_features"] == 6  # CS_fF, LD_pH, M, RD, RS, MN
    with np.load(tmp_path / "ctle_auto.npz", allow_pickle=False) as cache:
        assert sorted(cache["channel_names"].astype(str).tolist()) == ["gain", "phase"]


def _write_array_npz(path: Path, *, engine_cache: bool, samples: int = 6) -> None:
    """Write a prebuilt array .npz in either the engine-cache or bare-bundle layout."""
    rng = np.random.default_rng(0)
    arrays = {
        "features": rng.standard_normal((samples, 3)).astype(np.float32),
        "targets": rng.standard_normal((samples, 2, 4)).astype(np.float32),
        "frequency_hz": np.linspace(1e9, 4e9, 4).astype(np.float32),
        "channel_names": np.asarray(["gain", "phase"]),
        "channel_units": np.asarray(["dB", "deg"]),
        "channel_transforms": np.asarray(["", ""]),
        "sweep_label": np.asarray("Frequency (GHz)"),
    }
    if engine_cache:
        arrays["input_feature_names"] = np.asarray(["Temp_C", "VDD", "geom"])
        arrays["target_names"] = np.asarray(["gain", "phase"])
    else:
        # The layout offline prep scripts write: feature_names, no target_names.
        arrays["feature_names"] = np.asarray(["Temp_C", "VDD", "geom"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


@pytest.mark.parametrize("engine_cache", [True, False])
def test_build_cache_from_dataset_autodetect_array_npz(tmp_path: Path, engine_cache: bool) -> None:
    """A folder holding a prebuilt .npz is adopted as the dataset.

    Covers both the engine's own cache layout and the barer array-bundle layout
    (feature_names, no target_names) that offline preparation scripts produce.
    """
    root = tmp_path / ("adopt_cache" if engine_cache else "adopt_bundle")
    _write_array_npz(root / ("cache.npz" if engine_cache else "bundle.npz"), engine_cache=engine_cache)

    out = tmp_path / "adopted.npz"
    summary = data.build_cache_from_dataset(str(root), str(out))

    assert summary["num_samples"] == 6
    assert summary["num_features"] == 3
    assert summary["num_channels"] == 2
    assert summary["num_frequencies"] == 4

    # The written cache must be complete enough for the normal training path.
    bundle = data.load_split_bundle(out, batch_size=2, train_frac=0.6, val_frac=0.2)
    assert bundle.input_feature_names == ["Temp_C", "VDD", "geom"]
    assert bundle.channel_names == ["gain", "phase"]
    assert bundle.channel_units == ["dB", "deg"]


def test_array_npz_detection_yields_to_raw_sources(tmp_path: Path, synthetic_dataset: dict[str, Path]) -> None:
    """A raw Touchstone layout still wins when a stale .npz sits in the same folder."""
    import shutil

    root = tmp_path / "both"
    sp_dir = root / "SPData"
    sp_dir.mkdir(parents=True)
    for f in sorted(synthetic_dataset["output_dir"].glob("*.s2p")):
        shutil.copy(f, sp_dir / f.name)
    (root / "log.txt").write_text(
        "# [x, y, const, index]\n" + synthetic_dataset["input_file"].read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_array_npz(root / "cache.npz", engine_cache=True)

    summary = data.build_cache_from_dataset(str(root), str(tmp_path / "both.npz"))

    # Touchstone numbers (10 samples, 6 channels), not the decoy npz's (6 samples, 2).
    assert summary["num_samples"] == 10
    assert summary["num_channels"] == 6


@pytest.mark.parametrize(
    ("target_name", "label"),
    [
        ("cache.npz", "the detected source itself"),
        ("bundle.npz", "a sibling source the detector did not pick"),
        ("cache", "an extension-less path numpy resolves to the source"),
    ],
)
def test_array_npz_never_overwrites_a_source_file(tmp_path: Path, target_name: str, label: str) -> None:
    """A cache path that resolves onto any source .npz is refused, not written.

    ``np.savez_compressed`` appends ``.npz``, and the folder can hold several array
    files, so the guard must cover every candidate and the real write destination --
    otherwise a prepared dataset and its sidecar are destroyed with no warning.
    """
    root = tmp_path / f"guard_{target_name.replace('.', '_')}"
    _write_array_npz(root / "cache.npz", engine_cache=True, samples=6)
    _write_array_npz(root / "bundle.npz", engine_cache=False, samples=11)
    sidecar = root / "cache.json"
    sidecar.write_text(json.dumps({"dataset_name": "original"}), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in root.glob("*.npz")}

    with pytest.raises(ValueError, match="overwrite the source dataset file"):
        data.build_cache_from_dataset(str(root), str(root / target_name))

    assert {p.name: p.read_bytes() for p in root.glob("*.npz")} == before, f"clobbered {label}"
    assert json.loads(sidecar.read_text())["dataset_name"] == "original"


def test_array_npz_used_when_a_stray_non_cadence_csv_is_present(tmp_path: Path) -> None:
    """A metrics/manifest .csv beside a bundle must not shadow the prebuilt arrays.

    The Cadence branch triggers on any *.csv and raises when nothing parses, which
    previously made the array branch unreachable and reported a Cadence error for a
    dataset that has nothing to do with Cadence.
    """
    root = tmp_path / "stray_csv"
    _write_array_npz(root / "bundle.npz", engine_cache=False, samples=7)
    (root / "metrics.csv").write_text("run,rmse\n1,0.5\n", encoding="utf-8")

    summary = data.build_cache_from_dataset(str(root), str(tmp_path / "stray_out.npz"))

    assert summary["num_samples"] == 7


def test_array_npz_reports_corruption_rather_than_unsupported_format(tmp_path: Path) -> None:
    """A truncated .npz (interrupted copy) must not be reported as a bad folder layout."""
    root = tmp_path / "truncated"
    source = root / "cache.npz"
    _write_array_npz(source, engine_cache=True, samples=40)
    raw = source.read_bytes()
    source.write_bytes(raw[: len(raw) // 3])

    with pytest.raises(FileNotFoundError, match="incomplete"):
        data.build_cache_from_dataset(str(root), str(tmp_path / "trunc_out.npz"))


def test_array_npz_tolerates_scalar_label_arrays(tmp_path: Path) -> None:
    """0-d and 1-element string arrays are read as single labels, not character lists."""
    root = tmp_path / "scalar_labels"
    root.mkdir(parents=True)
    np.savez_compressed(
        root / "bundle.npz",
        features=np.ones((4, 1), dtype=np.float32),
        targets=np.ones((4, 1, 3), dtype=np.float32),
        frequency_hz=np.ones(3, dtype=np.float32),
        feature_names=np.asarray("vdiff"),
        channel_names=np.asarray("gain"),
        channel_units=np.asarray("dB"),
        channel_transforms=np.asarray(""),
        sweep_label=np.asarray(["VDIFF (mV)"]),  # 1-element, not 0-d
    )
    out = tmp_path / "scalar_out.npz"
    data.build_cache_from_dataset(str(root), str(out))

    bundle = data.load_split_bundle(out, batch_size=2, train_frac=0.5, val_frac=0.25)
    assert bundle.channel_names == ["gain"]
    assert bundle.channel_units == ["dB"]
    # Not "['VDIFF (mV)']", which would also corrupt the Hz->GHz 'freq' heuristic.
    assert bundle.sweep_label == "VDIFF (mV)"


def test_array_npz_rejects_mismatched_shapes(tmp_path: Path) -> None:
    root = tmp_path / "bad"
    root.mkdir(parents=True)
    np.savez_compressed(
        root / "cache.npz",
        features=np.zeros((5, 3), dtype=np.float32),
        targets=np.zeros((5, 2, 4), dtype=np.float32),
        frequency_hz=np.zeros(7, dtype=np.float32),  # 7 != 4 sweep points
    )
    with pytest.raises(ValueError, match="sweep points"):
        data.build_cache_from_dataset(str(root), str(tmp_path / "bad_out.npz"))


def test_load_cache_and_split_bundle(synthetic_dataset: dict[str, Path]) -> None:
    """The cache (pre-built by fixture) should load correctly."""

    cache_path = synthetic_dataset["cache_path"]

    # Cache was already built by the fixture.
    summary = data.load_existing_cache(cache_path)
    assert summary["status"] == "existing"
    assert summary["num_samples"] == 10
    assert summary["num_channels"] == 4
    assert summary["num_frequencies"] == 3

    with np.load(cache_path, allow_pickle=False) as arrays:
        assert arrays["features"].shape == (10, 3)
        assert arrays["targets"].shape == (10, 4, 3)
        assert arrays["input_feature_names"].astype(str).tolist() == ["x", "y", "const"]
        assert arrays["channel_names"].astype(str).tolist() == ["S11_re", "S11_im", "S12_re", "S12_im"]

    bundle = data.load_split_bundle(cache_path=cache_path, batch_size=4, seed=3, train_frac=0.6, val_frac=0.2)

    assert bundle.input_feature_names == ["x", "y", "const"]
    assert bundle.active_names == ["x", "y"]
    assert bundle.dropped_names == ["const"]
    train_x, train_y = next(iter(bundle.train_loader))
    assert train_x.shape[1] == 2
    assert tuple(train_y.shape[1:]) == (4, 3)


def _pvt_grid_features(num_designs: int) -> tuple[np.ndarray, list[str]]:
    """Build a corner-major PVT sweep: every design appears once per corner.

    Corner-major ordering means a design's rows are spread across the whole
    array — the worst case for any split that hopes contiguity keeps designs
    together, which is exactly what the design-level split must not rely on.
    """
    corners = [(temp, vdd) for temp in (-40.0, 27.0, 125.0) for vdd in (0.9, 1.0)]
    rows = [
        [0.5 * design, 1.25 * design + 3.0, temp, vdd]
        for temp, vdd in corners
        for design in range(num_designs)
    ]
    return np.asarray(rows, dtype=np.float32), ["geom_a", "geom_b", "Temp_C", "VDD"]


def _design_keys(features: np.ndarray, rows: np.ndarray) -> set[tuple[float, ...]]:
    return {tuple(map(float, features[row, :2])) for row in rows}


def test_design_split_keeps_all_corner_rows_of_a_design_together() -> None:
    """No design may appear in two folds — that is the leakage the split removes."""
    features, names = _pvt_grid_features(num_designs=20)

    split = data.design_split_indices(features, names, ["Temp_C", "VDD"], 0.8, 0.1, seed=42)

    # The folds partition all 120 rows.
    combined = np.sort(np.concatenate([split["train"], split["val"], split["test"]]))
    assert np.array_equal(combined, np.arange(len(features)))
    # 20 designs x 6 corners split 0.8/0.1 -> 16/2/2 designs = 96/12/12 rows.
    assert {name: len(rows) for name, rows in split.items()} == {"train": 96, "val": 12, "test": 12}
    # Design membership is disjoint across folds.
    train_keys = _design_keys(features, split["train"])
    val_keys = _design_keys(features, split["val"])
    test_keys = _design_keys(features, split["test"])
    assert len(train_keys) == 16 and len(val_keys) == 2 and len(test_keys) == 2
    assert not (train_keys & val_keys) and not (train_keys & test_keys) and not (val_keys & test_keys)

    # Deterministic under the same seed, different under another seed.
    repeat = data.design_split_indices(features, names, ["Temp_C", "VDD"], 0.8, 0.1, seed=42)
    assert all(np.array_equal(split[name], repeat[name]) for name in split)
    reshuffled = data.design_split_indices(features, names, ["Temp_C", "VDD"], 0.8, 0.1, seed=7)
    assert _design_keys(features, reshuffled["train"]) != train_keys


def test_design_split_validates_corner_columns() -> None:
    features, names = _pvt_grid_features(num_designs=4)

    with pytest.raises(ValueError, match="at least one corner column"):
        data.design_split_indices(features, names, [" "], 0.8, 0.1, seed=1)
    with pytest.raises(ValueError, match="not input features"):
        data.design_split_indices(features, names, ["Temp_C", "vdd_typo"], 0.8, 0.1, seed=1)
    with pytest.raises(ValueError, match="identify a design"):
        data.design_split_indices(features, names, names, 0.8, 0.1, seed=1)


def test_design_split_requires_three_designs(tmp_path: Path) -> None:
    """Fewer than 3 designs must fail loudly up front.

    split_indices would hand 1-2 designs an empty val or test fold, and an empty
    fold does not fail loudly: val_loss reads 0.0 every epoch (freezing the best
    checkpoint at epoch 1) and the run only crashes at final evaluation, after
    the whole epoch budget is spent.
    """
    features, names = _pvt_grid_features(num_designs=2)
    with pytest.raises(ValueError, match="at least 3 designs"):
        data.design_split_indices(features, names, ["Temp_C", "VDD"], 0.8, 0.1, seed=1)

    # The bundle path hits the same guard — including via max_samples, which
    # truncates rows BEFORE designs are grouped. On a design-major cache (all
    # corner rows of a design contiguous) a small smoke-run cap leaves 1 design.
    corners = [(temp, vdd) for temp in (-40.0, 27.0, 125.0) for vdd in (0.9, 1.0)]
    design_major = np.asarray(
        [
            [0.5 * design, 1.25 * design + 3.0, temp, vdd]
            for design in range(12)
            for temp, vdd in corners
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(9)
    cache_path = tmp_path / "design_major.npz"
    np.savez_compressed(
        cache_path,
        features=design_major,
        targets=rng.standard_normal((len(design_major), 2, 4)).astype(np.float32),
        frequency_hz=np.linspace(1e9, 4e9, 4).astype(np.float32),
        input_feature_names=np.asarray(names),
        target_names=np.asarray(["gain", "phase"]),
        channel_names=np.asarray(["gain", "phase"]),
        channel_units=np.asarray(["dB", "deg"]),
        channel_transforms=np.asarray(["", ""]),
        sweep_label=np.asarray("Frequency (GHz)"),
    )
    with pytest.raises(ValueError, match="at least 3 designs"):
        data.load_split_bundle(
            cache_path,
            batch_size=8,
            max_samples=6,  # exactly one design's corner rows
            split_corner_columns=["Temp_C", "VDD"],
        )


def _write_pvt_cache_npz(path: Path, num_designs: int = 12) -> tuple[np.ndarray, list[str]]:
    """Write an engine cache for a corner-major PVT grid and return its features."""
    features, names = _pvt_grid_features(num_designs=num_designs)
    rng = np.random.default_rng(5)
    np.savez_compressed(
        path,
        features=features,
        targets=rng.standard_normal((len(features), 2, 4)).astype(np.float32),
        frequency_hz=np.linspace(1e9, 4e9, 4).astype(np.float32),
        input_feature_names=np.asarray(names),
        target_names=np.asarray(["gain", "phase"]),
        channel_names=np.asarray(["gain", "phase"]),
        channel_units=np.asarray(["dB", "deg"]),
        channel_transforms=np.asarray(["", ""]),
        sweep_label=np.asarray("Frequency (GHz)"),
    )
    return features, names


def test_load_split_bundle_design_level_split(tmp_path: Path) -> None:
    """The bundle path: grouped split, per-fold design counts, and train-only stats."""
    cache_path = tmp_path / "pvt_cache.npz"
    features, _ = _write_pvt_cache_npz(cache_path, num_designs=12)

    bundle = data.load_split_bundle(
        cache_path,
        batch_size=8,
        seed=3,
        train_frac=0.8,
        val_frac=0.1,
        split_corner_columns=["Temp_C", "VDD"],
    )

    # 12 designs split 0.8/0.1 -> 9/1/2 designs = 54/6/12 of the 72 rows.
    assert bundle.design_counts == {"train": 9, "val": 1, "test": 2}
    assert {name: len(rows) for name, rows in bundle.split_indices.items()} == {"train": 54, "val": 6, "test": 12}
    for name, expected in (("train_loader", 54), ("val_loader", 6), ("test_loader", 12)):
        assert len(getattr(bundle, name).dataset) == expected
    train_keys = _design_keys(features, bundle.split_indices["train"])
    holdout_keys = _design_keys(features, bundle.split_indices["val"]) | _design_keys(
        features, bundle.split_indices["test"]
    )
    assert not (train_keys & holdout_keys)

    # The default row-level split reports no design counts.
    row_level = data.load_split_bundle(cache_path, batch_size=8, seed=3)
    assert row_level.design_counts is None


def test_suggest_uses_design_level_split(tmp_path: Path) -> None:
    """Suggestion diagnostics must describe the split the recommended run trains on.

    Without this, a design-split search persists a suggestion.json whose split
    sizes (and train-count-driven epoch/capacity heuristics) come from the
    row-level split its own trials never use.
    """
    cache_path = tmp_path / "pvt_suggest_cache.npz"
    _write_pvt_cache_npz(cache_path, num_designs=12)

    summary = suggest.suggest_initial_settings(
        SuggestConfig(
            data_root=None,
            cache_path=str(cache_path),
            split_corner_columns=["Temp_C", "VDD"],
        )
    )

    # 12 designs x 6 corners split 0.8/0.1 -> 9 train designs = 54 rows; the
    # row-level split would report int(72 * 0.8) = 57.
    assert summary["diagnostics"]["num_samples_train"] == 54
    # The embedded config keeps the setting, so TrainConfig(**config) round-trips
    # it into search trials and CLI --config-json runs.
    assert summary["suggested_baseline_config"]["split_corner_columns"] == ["Temp_C", "VDD"]


def test_suggest_initial_settings_reports_schema_driven_diagnostics(
    synthetic_dataset: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The suggestion pass should report diagnostics derived from the cached schema-aware data."""

    monkeypatch.setattr(
        suggest,
        "_detect_gpu",
        lambda: {"available": False, "name": None, "memory_gb": None},
    )

    result = suggest.suggest_initial_settings(
        SuggestConfig(
            input_feature_path=str(synthetic_dataset["input_dir"]),
            ground_truth_data_dir=str(synthetic_dataset["output_dir"]),
            cache_path=str(synthetic_dataset["cache_path"]),
            seed=5,
            train_frac=0.6,
            val_frac=0.2,
            variance_threshold=0.9,
        )
    )

    assert result["status"] == "ok"
    assert result["active_input_feature_names"] == ["x", "y"]
    assert result["dropped_input_feature_names"] == ["const"]
    assert result["diagnostics"]["num_samples_total"] == 10
    assert result["diagnostics"]["frequency_point_count"] == 3
    assert result["diagnostics"]["gpu_available"] is False
    assert result["suggested_baseline_config"]["data_root"] is None
    assert result["suggested_baseline_config"]["input_feature_path"] == str(synthetic_dataset["input_dir"])
    assert result["suggested_baseline_config"]["ground_truth_data_dir"] == str(synthetic_dataset["output_dir"])
    assert result["suggested_baseline_config"]["use_amp"] is False
    assert result["quick_search_hints"]["width"]


def test_normalize_objective_accepts_aliases() -> None:
    """Friendly objective spellings should normalize to canonical search names."""

    assert search._normalize_objective("best accuracy") == "best_accuracy"
    assert search._normalize_objective("fastest-acceptable") == "fastest_acceptable"


def test_quick_hyperparameter_search_ranks_trials_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A monkeypatched search run should still rank candidates and save artifacts."""

    base_config = TrainConfig(
        data_root=None,
        input_feature_path="synthetic_inputs",
        ground_truth_data_dir="synthetic_outputs",
        cache_path="synthetic_cache.npz",
        output_dir="artifacts/runs/baseline_v2",
        seed=11,
        batch_size=16,
        epochs=200,
        learning_rate=1e-4,
        weight_decay=1e-4,
        train_frac=0.8,
        val_frac=0.1,
        width=192,
        depth=4,
        use_amp=False,
        max_samples=24,
    )
    suggestion = {
        "status": "ok",
        "confidence": "medium",
        "confidence_reason": "synthetic search suggestion",
        "suggested_baseline_config": asdict(base_config),
        "suggested_baseline_ranges": {
            "width": {"selected": 192, "candidates": [128, 192, 256]},
            "depth": {"selected": 4, "candidates": [3, 4, 5]},
            "batch_size": {"selected": 16, "candidates": [8, 16, 32]},
            "learning_rate": {"selected": 1e-4, "candidates": [7e-5, 1e-4, 2e-4]},
            "weight_decay": {"selected": 1e-4, "candidates": [5e-5, 1e-4, 3e-4]},
        },
    }

    def fake_suggest_initial_settings(*args, **kwargs) -> dict[str, object]:
        return suggestion

    def fake_run_baseline_trial(config: TrainConfig, **kwargs) -> dict[str, object]:
        # Manufacture a simple accuracy/runtime tradeoff so ranking logic can be tested.
        mae = 0.020 + abs(config.width - 128) / 10000.0 + abs(config.depth - 3) / 5000.0
        mae += abs(config.learning_rate - 1e-4) * 100.0
        runtime = 10.0 + config.width / 32.0 + config.depth
        return {
            "status": "ok",
            "best_val_loss": mae + 0.004,
            "average_evaluation_mae": mae,
            "evaluation_loss": mae + 0.007,
            "frequency_mae": [mae, mae + 0.001, mae + 0.002],
            "runtime_seconds": runtime,
            "epochs_completed": config.epochs,
            "best_epoch": min(config.epochs, 4),
            "history": [{"epoch": 1, "train_loss": 0.2, "val_loss": 0.1}],
            "device": "cpu",
        }

    monkeypatch.setattr(search, "suggest_initial_settings", fake_suggest_initial_settings)
    monkeypatch.setattr(search, "run_baseline_trial", fake_run_baseline_trial)

    summary = search.quick_hyperparameter_search(
        SearchConfig(
            data_root=None,
            output_dir=str(tmp_path / "search_runs"),
            trial_count=4,
            epochs_per_trial=5,
            objective="balanced",
        )
    )

    assert summary["status"] == "ok"
    assert summary["objective"] == "balanced"
    assert summary["trial_count_completed"] == 4
    assert summary["recommended_trial"]["label"] == "smaller_model"
    assert Path(summary["run_dir"]).is_dir()
    assert Path(summary["tradeoff_plot_path"]).is_file()
    assert Path(summary["trial_results_path"]).is_file()
    assert (Path(summary["run_dir"]) / "summary.json").is_file()
    assert (Path(summary["run_dir"]) / "suggestion.json").is_file()

    saved_summary = json.loads((Path(summary["run_dir"]) / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["recommended_trial"]["label"] == "smaller_model"
    assert saved_summary["objective_label"] == "Balanced accuracy and time"


def _pvt_grid_with_derived(num_designs: int = 12) -> tuple[np.ndarray, list[str]]:
    """PVT grid carrying two derived columns, as real datasets do.

    ``emb0`` is a function of the corner alone (a frozen corner embedding);
    ``phys`` depends on the design AND the temperature (a physics anchor). Both
    change between a design's corner rows, so both must be treated as corner
    columns or they silently splinter designs.
    """
    base, names = _pvt_grid_features(num_designs)
    temp, vdd = base[:, 2], base[:, 3]
    emb0 = 0.25 * temp + 3.0 * vdd
    phys = (base[:, 0] + 1.0) * (1.0 + 0.001 * temp)
    return np.column_stack([base, emb0, phys]).astype(np.float32), names + ["emb0", "phys"]


def test_design_split_rejects_a_grouping_that_is_really_row_level() -> None:
    """One design per row is exactly the row-level split, so it must not be
    accepted under a design-level label — that is the silent failure where the
    fold counts look leak-free and every design still crosses folds."""
    features, names = _pvt_grid_features(num_designs=20)
    # Make every row unique on the non-corner columns, as an unlisted
    # corner-varying column does.
    features = np.column_stack([features, np.arange(len(features), dtype=np.float32)])
    names = names + ["row_tag"]

    with pytest.raises(ValueError, match="one design per row"):
        data.design_split_indices(features, names, ["Temp_C", "VDD"], 0.8, 0.1, seed=42)


def test_corner_derived_columns_are_folded_into_the_corner_list() -> None:
    """A column determined by the corner values cannot identify a design, so it is
    treated as a corner column instead of splintering every design."""
    features, names = _pvt_grid_with_derived(num_designs=12)

    with pytest.warns(UserWarning, match="emb0"):
        group_ids = data.design_group_ids(features, names, ["Temp_C", "VDD"])
    # 'phys' still varies with temperature, so designs remain split per temperature
    # (3 temps) rather than collapsing to 12 — the partial case the warning covers.
    assert int(group_ids.max()) + 1 == 36


def test_design_split_warns_when_designs_do_not_reach_every_corner() -> None:
    features, names = _pvt_grid_with_derived(num_designs=12)

    with pytest.warns(UserWarning, match="no design appears at more than"):
        data.design_split_indices(features, names, ["Temp_C", "VDD"], 0.8, 0.1, seed=42)


def test_design_columns_specify_the_split_directly() -> None:
    """Naming the design-identifying columns is immune to derived corner columns:
    everything unnamed is corner-varying, so nothing can splinter a design."""
    features, names = _pvt_grid_with_derived(num_designs=12)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no auto-fold, no crossing warning
        group_ids = data.design_group_ids(features, names, design_columns=["geom_a", "geom_b"])
    assert int(group_ids.max()) + 1 == 12

    split = data.design_split_indices(
        features, names, None, 0.8, 0.1, seed=42, design_columns=["geom_a", "geom_b"]
    )
    keys = {name: _design_keys(features, rows) for name, rows in split.items()}
    assert not (keys["train"] & keys["test"]) and not (keys["train"] & keys["val"])
    assert sum(len(rows) for rows in split.values()) == len(features)


def test_design_split_requires_exactly_one_column_list() -> None:
    features, names = _pvt_grid_features(num_designs=6)

    with pytest.raises(ValueError, match="exactly one of corner_columns"):
        data.design_group_ids(features, names)
    with pytest.raises(ValueError, match="exactly one of corner_columns"):
        data.design_group_ids(features, names, ["Temp_C"], design_columns=["geom_a"])
    with pytest.raises(ValueError, match="not input features"):
        data.design_group_ids(features, names, design_columns=["nope"])


def test_load_split_bundle_accepts_design_columns(tmp_path: Path) -> None:
    cache_path = tmp_path / "pvt_design_cols.npz"
    _write_pvt_cache_npz(cache_path, num_designs=12)

    bundle = data.load_split_bundle(
        cache_path, batch_size=8, seed=3, split_design_columns=["geom_a", "geom_b"]
    )
    assert bundle.design_counts == {"train": 9, "val": 1, "test": 2}
