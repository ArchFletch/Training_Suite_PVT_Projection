"""Focused transfer-runner tests.

This file keeps the transfer coverage narrow and fast by patching out the
expensive training internals and checking only the progress bookkeeping that
the GUI and summaries depend on.
"""

from __future__ import annotations

from time import perf_counter

import torch
from torch.utils.data import DataLoader, TensorDataset

from xfmr_v2 import runner


def test_train_band_reports_elapsed_and_eta(monkeypatch) -> None:
    """Transfer progress events should expose elapsed and ETA timing fields."""

    losses = iter([0.40, 0.30])
    monkeypatch.setattr(runner, "_run_epoch", lambda *args, **kwargs: next(losses))

    model_kwargs = {
        "input_feature_dim": 1,
        "ground_truth_channels": 1,
        "width": 8,
        "depth": 2,
    }
    init_model = runner.SpectraNet(num_frequencies=3, **model_kwargs)
    init_state = runner.clone_state(init_model.state_dict())
    loader = DataLoader(
        # A tiny deterministic tensor dataset is enough because the patched epoch
        # runner ignores the actual values and only needs the loader structure.
        TensorDataset(torch.zeros((2, 1), dtype=torch.float32), torch.zeros((2, 1, 3), dtype=torch.float32)),
        batch_size=1,
        shuffle=False,
    )

    events: list[dict[str, object]] = []
    runner._train_band(
        model_kwargs=model_kwargs,
        init_state=init_state,
        loader=loader,
        device=torch.device("cpu"),
        amp=False,
        epochs=2,
        lr=1e-3,
        weight_decay=0.0,
        show_progress=False,
        progress_callback=events.append,
        event_context={"band_run_index": 2, "total_band_runs": 5},
        run_start_time=perf_counter() - 0.5,
        model_type="SpectraNet",
        num_frequencies=3,
    )

    epoch_events = [event for event in events if event["event"] == "band_epoch_end"]

    assert len(epoch_events) == 2
    assert all("elapsed_seconds" in event["data"] for event in epoch_events)
    assert all("eta_seconds" in event["data"] for event in epoch_events)
    assert all(event["data"]["elapsed_seconds"] >= 0.0 for event in epoch_events)
    assert all(event["data"]["eta_seconds"] >= 0.0 for event in epoch_events)


def test_legacy_model_type_names_still_build() -> None:
    """Runs saved under the pre-rename names must still load after the rename.

    ``canonical_model_type`` maps the legacy strings, and ``build_model`` applies
    it, so an old checkpoint's ``model_type`` instantiates the renamed class.
    """
    from xfmr_v2.model import SpectraHydra, SpectraNet

    assert runner.canonical_model_type("FlatMLP") == "SpectraNet"
    assert runner.canonical_model_type("CTLE_MLP") == "SpectraHydra"
    # Current names pass through unchanged.
    assert runner.canonical_model_type("SpectraNet") == "SpectraNet"
    assert runner.canonical_model_type("SpectraHydra") == "SpectraHydra"

    kwargs = {"input_feature_dim": 1, "ground_truth_channels": 1, "width": 8, "depth": 2}
    assert isinstance(runner.build_model("FlatMLP", num_frequencies=3, **kwargs), SpectraNet)
    assert isinstance(runner.build_model("CTLE_MLP", num_frequencies=3, **kwargs), SpectraHydra)


def test_train_baseline_stop_returns_stopped_summary() -> None:
    """Stopping a baseline run must return a 'stopped' summary, not raise KeyError.

    Regression: a cancelled run yields a partial result without 'run_dir' (and the
    other completed-run keys); train_baseline used to index those unconditionally and
    crash with KeyError('run_dir'). The stop check fires before any data is touched.
    """
    summary = runner.train_baseline(
        runner.TrainConfig(),
        show_progress=False,
        should_stop=lambda: True,
    )
    assert summary["status"] == "stopped"
    # The completed-run keys must be absent rather than raising.
    assert "run_dir" not in summary
    assert "test_loss" not in summary


def test_frequency_mse_matches_per_channel_loop() -> None:
    """The vectorized frequency_mse must equal the per-channel-loop form it replaced.

    It is the loss the suggest heuristic picks for SpectraHydra, so a silent change
    here would shift every reported loss and every best-checkpoint decision.
    """
    torch.manual_seed(0)
    for channels in (1, 2, 3, 6):
        pred = torch.randn(7, channels, 11, dtype=torch.float64)
        target = torch.randn(7, channels, 11, dtype=torch.float64)
        if channels == 1:
            expected = torch.mean((pred - target) ** 2)
        else:
            expected = sum(
                torch.mean((pred[:, c, :] - target[:, c, :]) ** 2) for c in range(channels)
            )
        assert torch.allclose(runner.frequency_mse(pred, target), expected, rtol=0, atol=1e-12)


def test_eval_loss_is_batch_size_independent() -> None:
    """Evaluation batching must not move the metric.

    load_split_bundle now evaluates with a larger batch than it trains with, which is
    only safe because the loss is accumulated weighted by sample count.
    """
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    x = torch.randn(10, 3)
    y = torch.randn(10, 2, 5)
    model = runner.build_model(
        "SpectraNet", num_frequencies=5, input_feature_dim=3, ground_truth_channels=2,
        width=8, depth=2,
    )
    dataset = TensorDataset(x, y)
    losses = [
        runner._eval_loss(
            model, DataLoader(dataset, batch_size=bs, shuffle=False),
            torch.device("cpu"), amp=False, loss_fn=runner.frequency_mse,
        )
        for bs in (1, 3, 10)
    ]
    assert max(losses) - min(losses) < 1e-6, losses


def test_limit_cpu_threads_only_affects_gpu_runs() -> None:
    """The thread cap must leave CPU training's wide thread pool alone."""
    original = torch.get_num_threads()
    try:
        torch.set_num_threads(original)
        runner.limit_cpu_threads_for_gpu(torch.device("cpu"))
        assert torch.get_num_threads() == original, "CPU training must keep its threads"

        torch.set_num_threads(max(original, 16))
        runner.limit_cpu_threads_for_gpu(torch.device("cuda"))
        assert torch.get_num_threads() == runner._GPU_TRAINING_CPU_THREADS
        # Idempotent, and never raises the count back up.
        torch.set_num_threads(1)
        runner.limit_cpu_threads_for_gpu(torch.device("cuda"))
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(original)


def _write_pvt_cache(path, num_designs: int = 12) -> None:
    """Write a tiny engine cache of `num_designs` designs x 6 PVT corners."""
    import numpy as np

    corners = [(temp, vdd) for temp in (-40.0, 27.0, 125.0) for vdd in (0.9, 1.0)]
    features = np.asarray(
        [
            [0.5 * design, 1.25 * design + 3.0, temp, vdd]
            for temp, vdd in corners
            for design in range(num_designs)
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(11)
    np.savez_compressed(
        path,
        features=features,
        targets=rng.standard_normal((len(features), 2, 4)).astype(np.float32),
        frequency_hz=np.linspace(1e9, 4e9, 4).astype(np.float32),
        input_feature_names=np.asarray(["geom_a", "geom_b", "Temp_C", "VDD"]),
        target_names=np.asarray(["gain", "phase"]),
        channel_names=np.asarray(["gain", "phase"]),
        channel_units=np.asarray(["dB", "deg"]),
        channel_transforms=np.asarray(["", ""]),
        sweep_label=np.asarray("Frequency (GHz)"),
    )


def test_baseline_design_level_split_reports_and_persists(tmp_path) -> None:
    """split_corner_columns flows config -> load_split_bundle -> progress + checkpoint."""
    from pathlib import Path

    cache_path = tmp_path / "pvt_cache.npz"
    _write_pvt_cache(cache_path)
    events: list[dict[str, object]] = []
    summary = runner.train_baseline(
        runner.TrainConfig(
            data_root=str(tmp_path),
            cache_path=str(cache_path),
            output_dir=str(tmp_path / "runs/design_split_baseline"),
            split_corner_columns=["Temp_C", "VDD"],
            width=8,
            depth=2,
            epochs=1,
            batch_size=16,
            use_amp=False,
            device="cpu",
        ),
        show_progress=False,
        progress_callback=events.append,
    )
    assert summary.get("status") != "stopped"

    # 12 designs split 0.8/0.1 -> 9/1/2 designs, reported on the data_ready event.
    data_ready = next(event for event in events if event["event"] == "data_ready")
    assert data_ready["data"]["design_counts"] == {"train": 9, "val": 1, "test": 2}
    assert data_ready["data"]["train_samples"] == 54

    # The checkpoint's config keeps the split setting for reproducibility.
    checkpoint = torch.load(
        Path(summary["run_dir"]) / "best_model.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["config"]["split_corner_columns"] == ["Temp_C", "VDD"]


def _write_external_eval(path, num_rows: int = 8) -> None:
    """External eval npz: the model's columns present by NAME — shuffled order,
    plus a column the model never saw — so only name-matching can align it."""
    import numpy as np

    rng = np.random.default_rng(7)
    columns = {
        "extra_junk": rng.standard_normal(num_rows),
        "VDD": rng.choice([0.9, 1.0], size=num_rows),
        "geom_a": rng.uniform(0.0, 6.0, size=num_rows),
        "Temp_C": rng.choice([-40.0, 27.0, 125.0], size=num_rows),
        "geom_b": rng.uniform(3.0, 18.0, size=num_rows),
    }
    np.savez_compressed(
        path,
        features=np.stack(list(columns.values()), axis=1).astype(np.float32),
        targets=rng.standard_normal((num_rows, 2, 4)).astype(np.float32),
        feature_names=np.asarray(list(columns)),
    )


def test_baseline_external_eval_scores_best_checkpoint(tmp_path) -> None:
    """eval_dataset_path: the best checkpoint is scored on the external file and
    the reported per-channel MAE equals a by-hand score of the saved checkpoint,
    which pins down the name-based column alignment and the normalization."""
    from pathlib import Path

    import numpy as np

    cache_path = tmp_path / "pvt_cache.npz"
    _write_pvt_cache(cache_path)
    eval_path = tmp_path / "holdout.npz"
    _write_external_eval(eval_path)

    events: list[dict[str, object]] = []
    summary = runner.train_baseline(
        runner.TrainConfig(
            data_root=str(tmp_path),
            cache_path=str(cache_path),
            output_dir=str(tmp_path / "runs/external_eval"),
            eval_dataset_path=str(eval_path),
            width=8,
            depth=2,
            epochs=1,
            batch_size=16,
            use_amp=False,
            device="cpu",
        ),
        show_progress=False,
        progress_callback=events.append,
    )
    assert summary.get("status") != "stopped"
    assert summary["external_eval_rows"] == 8
    assert [label.split(":")[0] for label in summary["external_channel_mae_with_units"]] == ["gain", "phase"]

    # The file is validated before any training time is spent.
    order = [event["event"] for event in events]
    assert order.index("external_eval_ready") < order.index("model_ready")
    assert "external_evaluation_completed" in order

    checkpoint = torch.load(
        Path(summary["run_dir"]) / "best_model.pt", map_location="cpu", weights_only=False
    )
    with np.load(eval_path) as data:
        names = data["feature_names"].astype(str).tolist()
        column_index = [names.index(name) for name in checkpoint["active_input_feature_names"]]
        x = (data["features"][:, column_index] - checkpoint["input_feature_mean"]) / checkpoint["input_feature_std"]
        targets = np.asarray(data["targets"], dtype=np.float32)
    model = runner.build_model(
        "SpectraNet", num_frequencies=4, input_feature_dim=x.shape[1],
        ground_truth_channels=2, width=8, depth=2,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x.astype(np.float32))).numpy()
    pred = pred * checkpoint["target_std"] + checkpoint["target_mean"]
    expected_per_channel = np.abs(pred - targets).mean(axis=(0, 2))
    assert np.allclose(summary["external_per_channel_mae"], expected_per_channel, atol=1e-5)


def test_baseline_external_eval_fails_before_training_on_a_bad_file(tmp_path) -> None:
    """A mismatched or missing eval file must cost zero epochs — the run fails at
    data time, not after the full training budget."""
    import numpy as np
    import pytest

    cache_path = tmp_path / "pvt_cache.npz"
    _write_pvt_cache(cache_path)
    eval_path = tmp_path / "bad_holdout.npz"
    rng = np.random.default_rng(3)
    np.savez_compressed(
        eval_path,
        features=rng.standard_normal((5, 2)).astype(np.float32),
        targets=rng.standard_normal((5, 2, 4)).astype(np.float32),
        feature_names=np.asarray(["geom_a", "Temp_C"]),  # geom_b and VDD missing
    )

    base = dict(
        data_root=str(tmp_path),
        cache_path=str(cache_path),
        width=8,
        depth=2,
        epochs=1,
        batch_size=16,
        use_amp=False,
        device="cpu",
    )
    events: list[dict[str, object]] = []
    with pytest.raises(ValueError, match="lacks input columns"):
        runner.train_baseline(
            runner.TrainConfig(
                output_dir=str(tmp_path / "runs/external_eval_bad"),
                eval_dataset_path=str(eval_path),
                **base,
            ),
            show_progress=False,
            progress_callback=events.append,
        )
    assert not any(event["event"] == "epoch_end" for event in events)

    with pytest.raises(ValueError, match="not found"):
        runner.train_baseline(
            runner.TrainConfig(
                output_dir=str(tmp_path / "runs/external_eval_missing"),
                eval_dataset_path=str(tmp_path / "does_not_exist.npz"),
                **base,
            ),
            show_progress=False,
        )


def _tiny_bundle(tmp_path):
    from xfmr_v2 import data

    cache_path = tmp_path / "pvt_cache.npz"
    _write_pvt_cache(cache_path)
    return data.load_split_bundle(
        cache_path=cache_path, batch_size=16, seed=0, train_frac=0.8, val_frac=0.1
    )


def test_external_eval_loader_cross_checks_file_metadata(tmp_path) -> None:
    """Shape-compatible files whose own metadata contradicts the training dataset
    must be rejected: a value-space mismatch (log10 vs raw) exponentiates raw
    values into garbage, swapped channels score each against the other's
    statistics, and a different grid at the same point count compares the wrong
    frequencies — all silently, if allowed through."""
    import numpy as np
    import pytest

    bundle = _tiny_bundle(tmp_path)
    rng = np.random.default_rng(5)
    base = dict(
        features=rng.standard_normal((6, 4)).astype(np.float32),
        targets=rng.standard_normal((6, 2, 4)).astype(np.float32),
        feature_names=np.asarray(["geom_a", "geom_b", "Temp_C", "VDD"]),
    )

    def write(name, **extra):
        path = tmp_path / name
        np.savez_compressed(path, **{**base, **extra})
        return str(path)

    with pytest.raises(ValueError, match="channel transforms"):
        runner._load_external_eval(
            write("transforms.npz", channel_transforms=np.asarray(["log10", ""])), bundle
        )
    with pytest.raises(ValueError, match="declares channels"):
        runner._load_external_eval(
            write("channels.npz", channel_names=np.asarray(["phase", "gain"])), bundle
        )
    with pytest.raises(ValueError, match="frequency grid"):
        runner._load_external_eval(
            write("grid.npz", frequency_hz=np.linspace(2e9, 9e9, 4).astype(np.float32)), bundle
        )
    # Matching metadata sails through.
    x, y, notes = runner._load_external_eval(
        write(
            "ok.npz",
            channel_names=np.asarray(["gain", "phase"]),
            channel_transforms=np.asarray(["", ""]),
            frequency_hz=np.linspace(1e9, 4e9, 4).astype(np.float32),
        ),
        bundle,
    )
    assert x.shape == (6, 4) and y.shape == (6, 2, 4) and notes == []


def test_external_eval_loader_rejects_malformed_and_nonfinite_files(tmp_path) -> None:
    """NaN rows, ambiguous duplicate column names, and a names/width mismatch must
    all fail as ValueError before training — not surface as 'nan' MAE after the
    full epoch budget or escape as a raw numpy IndexError."""
    import numpy as np
    import pytest

    bundle = _tiny_bundle(tmp_path)
    rng = np.random.default_rng(5)
    targets = rng.standard_normal((6, 2, 4)).astype(np.float32)

    nan_features = rng.standard_normal((6, 4)).astype(np.float32)
    nan_features[2, 1] = np.nan
    nan_path = tmp_path / "nan.npz"
    np.savez_compressed(
        nan_path, features=nan_features, targets=targets,
        feature_names=np.asarray(["geom_a", "geom_b", "Temp_C", "VDD"]),
    )
    with pytest.raises(ValueError, match="NaN or infinite"):
        runner._load_external_eval(str(nan_path), bundle)

    dup_path = tmp_path / "dup.npz"
    np.savez_compressed(
        dup_path, features=rng.standard_normal((6, 5)).astype(np.float32), targets=targets,
        feature_names=np.asarray(["geom_a", "geom_b", "Temp_C", "VDD", "Temp_C"]),
    )
    with pytest.raises(ValueError, match="more than once"):
        runner._load_external_eval(str(dup_path), bundle)

    short_path = tmp_path / "short.npz"
    np.savez_compressed(
        short_path, features=rng.standard_normal((6, 3)).astype(np.float32), targets=targets,
        feature_names=np.asarray(["geom_a", "geom_b", "Temp_C", "VDD"]),
    )
    with pytest.raises(ValueError, match="malformed"):
        runner._load_external_eval(str(short_path), bundle)


def test_external_eval_loader_accepts_cache_layout_and_notes_dropped_columns(tmp_path) -> None:
    """The engine's own cache.npz layout (input_feature_names) qualifies, and a
    column that was constant in training but varies externally produces a note —
    the model is blind to it, and the reader of the score should know."""
    import numpy as np
    from xfmr_v2 import data

    base_path = tmp_path / "base.npz"
    _write_pvt_cache(base_path)
    with np.load(base_path) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    arrays["features"] = np.concatenate(
        [arrays["features"], np.ones((len(arrays["features"]), 1), np.float32)], axis=1
    )
    arrays["input_feature_names"] = np.asarray(
        [*arrays["input_feature_names"].astype(str).tolist(), "bias"]
    )
    cache_path = tmp_path / "pvt_cache_bias.npz"
    np.savez_compressed(cache_path, **arrays)
    bundle = data.load_split_bundle(
        cache_path=cache_path, batch_size=16, seed=0, train_frac=0.8, val_frac=0.1
    )
    assert "bias" in bundle.dropped_names

    rng = np.random.default_rng(9)
    external_path = tmp_path / "external.npz"
    np.savez_compressed(
        external_path,
        features=rng.standard_normal((6, 5)).astype(np.float32),
        targets=rng.standard_normal((6, 2, 4)).astype(np.float32),
        # The cache layout's key, not "feature_names" — both must be accepted.
        input_feature_names=np.asarray(["geom_a", "geom_b", "Temp_C", "VDD", "bias"]),
    )
    x, _y, notes = runner._load_external_eval(str(external_path), bundle)
    assert x.shape == (6, 4)  # only the active columns feed the model
    assert any("bias" in note for note in notes)


def test_stop_during_external_eval_keeps_the_trained_run(tmp_path, monkeypatch) -> None:
    """A Stop click while the optional external scoring runs must not discard the
    fully-trained checkpoint and its artifacts — only the external numbers."""
    from pathlib import Path

    cache_path = tmp_path / "pvt_cache.npz"
    _write_pvt_cache(cache_path)
    eval_path = tmp_path / "holdout.npz"
    _write_external_eval(eval_path)

    calls = {"count": 0}
    real_eval_metrics = runner._eval_metrics

    def cancel_on_external(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:  # 1st call = internal test fold, 2nd = external set
            raise runner.RunCancelled()
        return real_eval_metrics(*args, **kwargs)

    monkeypatch.setattr(runner, "_eval_metrics", cancel_on_external)
    events: list[dict[str, object]] = []
    summary = runner.train_baseline(
        runner.TrainConfig(
            data_root=str(tmp_path),
            cache_path=str(cache_path),
            output_dir=str(tmp_path / "runs/external_eval_stopped"),
            eval_dataset_path=str(eval_path),
            width=8, depth=2, epochs=1, batch_size=16, use_amp=False, device="cpu",
        ),
        show_progress=False,
        progress_callback=events.append,
    )
    assert summary.get("status") != "stopped"
    assert "average_external_mae" not in summary
    assert any(event["event"] == "external_evaluation_skipped" for event in events)
    assert (Path(summary["run_dir"]) / "best_model.pt").exists()


def test_transfer_design_level_split_runs_and_validates(tmp_path) -> None:
    """Self-transfer honors split_corner_columns and fails loudly on unknown names."""
    import pytest

    cache_path = tmp_path / "pvt_cache.npz"
    _write_pvt_cache(cache_path)
    base = dict(
        cache_path=str(cache_path),
        width=8,
        depth=2,
        num_bands=2,
        iterations=1,
        transfer_epochs=1,
        batch_size=16,
        use_amp=False,
        device="cpu",
    )

    events: list[dict[str, object]] = []
    summary = runner.run_self_transfer(
        runner.TransferConfig(
            output_dir=str(tmp_path / "runs/design_split_transfer"),
            split_corner_columns=["Temp_C", "VDD"],
            **base,
        ),
        show_progress=False,
        progress_callback=events.append,
    )
    assert summary["status"] == "ok"
    # Pin the fold sizes so a regression back to the row-level split (57/8 rows
    # for these 72 rows) cannot pass: 12 designs split 0.8/0.1 -> 9/1/2 designs
    # = 54 train / 12 test rows.
    data_ready = next(event for event in events if event["event"] == "data_ready")
    assert data_ready["data"]["train_samples"] == 54
    assert data_ready["data"]["test_samples"] == 12

    with pytest.raises(ValueError, match="not input features"):
        runner.run_self_transfer(
            runner.TransferConfig(
                output_dir=str(tmp_path / "runs/design_split_transfer_bad"),
                split_corner_columns=["Temp_K"],
                **base,
            ),
            show_progress=False,
        )


def test_design_columns_split_flows_through_baseline_and_transfer(tmp_path) -> None:
    """split_design_columns must reach load_split_bundle and the transfer split, not
    just the data layer: 12 designs -> 9/1/2 designs = 54 train rows either way."""
    cache_path = tmp_path / "pvt_cache.npz"
    _write_pvt_cache(cache_path)
    design_columns = ["geom_a", "geom_b"]

    events: list[dict[str, object]] = []
    summary = runner.train_baseline(
        runner.TrainConfig(
            data_root=str(tmp_path),
            cache_path=str(cache_path),
            output_dir=str(tmp_path / "runs/design_cols_baseline"),
            split_design_columns=design_columns,
            width=8, depth=2, epochs=1, batch_size=16, use_amp=False, device="cpu",
        ),
        show_progress=False,
        progress_callback=events.append,
    )
    assert summary.get("status") != "stopped"
    data_ready = next(event for event in events if event["event"] == "data_ready")
    assert data_ready["data"]["design_counts"] == {"train": 9, "val": 1, "test": 2}
    assert data_ready["data"]["train_samples"] == 54

    transfer_events: list[dict[str, object]] = []
    transfer = runner.run_self_transfer(
        runner.TransferConfig(
            cache_path=str(cache_path),
            output_dir=str(tmp_path / "runs/design_cols_transfer"),
            split_design_columns=design_columns,
            width=8, depth=2, num_bands=2, iterations=1, transfer_epochs=1,
            batch_size=16, use_amp=False, device="cpu",
        ),
        show_progress=False,
        progress_callback=transfer_events.append,
    )
    assert transfer["status"] == "ok"
    transfer_ready = next(e for e in transfer_events if e["event"] == "data_ready")
    assert transfer_ready["data"]["train_samples"] == 54
    assert transfer_ready["data"]["test_samples"] == 12
