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
    init_model = runner.FlatMLPNet(num_frequencies=3, **model_kwargs)
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
        freq_slice=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32).numpy(),
        device=torch.device("cpu"),
        amp=False,
        epochs=2,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        show_progress=False,
        progress_callback=events.append,
        event_context={"band_run_index": 2, "total_band_runs": 5},
        run_start_time=perf_counter() - 0.5,
        model_type="FlatMLP",
        num_frequencies=3,
    )

    epoch_events = [event for event in events if event["event"] == "band_epoch_end"]

    assert len(epoch_events) == 2
    assert all("elapsed_seconds" in event["data"] for event in epoch_events)
    assert all("eta_seconds" in event["data"] for event in epoch_events)
    assert all(event["data"]["elapsed_seconds"] >= 0.0 for event in epoch_events)
    assert all(event["data"]["eta_seconds"] >= 0.0 for event in epoch_events)
