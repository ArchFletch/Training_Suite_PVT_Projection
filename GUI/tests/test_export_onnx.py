"""Tests for the ONNX exporter (xfmr_v2.export_onnx).

These build tiny checkpoints (no training needed), export them, and verify the
ONNX graph reproduces the torch pipeline (normalize -> net -> denormalize) so the
end-to-end numerics and the metadata/reshape contract are pinned down.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from xfmr_v2.export_onnx import export_checkpoint_to_onnx
from xfmr_v2.runner import build_model

# The exporter needs `onnx`; the numerical check needs `onnxruntime`.
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def _make_checkpoint(
    run_dir: Path,
    *,
    model_type: str,
    n_features: int = 4,
    channels: int = 2,
    freqs: int = 8,
    width: int = 16,
    depth: int = 3,
    transforms: list[str] | None = None,
    with_cache: bool = False,
) -> tuple[Path, dict]:
    """Write a minimal but valid checkpoint and return (path, reference info)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(
        model_type,
        num_frequencies=freqs,
        input_feature_dim=n_features,
        ground_truth_channels=channels,
        width=width,
        depth=depth,
    )
    model.eval()
    rng = np.random.default_rng(0)
    input_mean = rng.standard_normal(n_features).astype(np.float32)
    input_std = (np.abs(rng.standard_normal(n_features)) + 0.5).astype(np.float32)
    target_mean = rng.standard_normal((channels, freqs)).astype(np.float32)
    target_std = (np.abs(rng.standard_normal((channels, freqs))) + 0.5).astype(np.float32)

    cache_path = ""
    if with_cache:
        cache_path = str(run_dir / "cache.npz")
        units = ["dB", "deg", "V", "A"][:channels]
        np.savez_compressed(
            cache_path,
            frequency_hz=np.linspace(1e9, 5e9, freqs).astype(np.float32),
            channel_units=np.asarray(units),
            channel_transforms=np.asarray(transforms or [""] * channels),
            sweep_label=np.asarray("Frequency (GHz)"),
        )

    checkpoint = {
        "model_state": model.state_dict(),
        "config": {"model_type": model_type, "width": width, "depth": depth, "cache_path": cache_path},
        "active_input_feature_names": [f"f{i}" for i in range(n_features)],
        "target_channel_names": [f"ch{i}" for i in range(channels)],
        "input_feature_mean": input_mean,
        "input_feature_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }
    ckpt_path = run_dir / "best_model.pt"
    torch.save(checkpoint, ckpt_path)
    return ckpt_path, {
        "model": model,
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "channels": channels,
        "freqs": freqs,
        "n_features": n_features,
        "transforms": transforms or [""] * channels,
    }


def _reference(ref: dict, x: np.ndarray) -> np.ndarray:
    """Recompute the baked pipeline in numpy/torch: physical-unit, flat (B, C*F)."""
    xn = (x - ref["input_mean"]) / ref["input_std"]
    with torch.no_grad():
        yn = ref["model"](torch.from_numpy(xn.astype(np.float32)), torch.zeros(1)).numpy()
    y = yn * ref["target_std"] + ref["target_mean"]
    for c, t in enumerate(ref["transforms"]):
        if t == "log10":
            y[:, c, :] = np.power(10.0, y[:, c, :])
    return y.reshape(x.shape[0], ref["channels"] * ref["freqs"])


@pytest.mark.parametrize("model_type", ["CTLE_MLP", "FlatMLP"])
def test_export_matches_torch_pipeline(tmp_path: Path, model_type: str) -> None:
    ckpt, ref = _make_checkpoint(tmp_path / model_type, model_type=model_type, with_cache=True)
    out = export_checkpoint_to_onnx(ckpt)

    assert out.exists()
    assert out.with_suffix(".meta.json").exists()

    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, ref["n_features"])).astype(np.float32)
    onnx_out = ort.InferenceSession(str(out)).run(["prediction"], {"input_features": x})[0]
    expected = _reference(ref, x)

    assert onnx_out.shape == (5, ref["channels"] * ref["freqs"])
    assert np.max(np.abs(onnx_out - expected)) < 1e-3


def test_log10_channel_is_inverted_in_graph(tmp_path: Path) -> None:
    transforms = ["log10", ""]
    ckpt, ref = _make_checkpoint(
        tmp_path / "log10", model_type="CTLE_MLP", channels=2, transforms=transforms, with_cache=True
    )
    out = export_checkpoint_to_onnx(ckpt)

    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, ref["n_features"])).astype(np.float32)
    onnx_out = ort.InferenceSession(str(out)).run(["prediction"], {"input_features": x})[0]
    expected = _reference(ref, x)
    assert np.max(np.abs(onnx_out - expected)) < 1e-3

    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["channel_transforms"] == transforms


def test_meta_sidecar_contract_and_reshape(tmp_path: Path) -> None:
    ckpt, ref = _make_checkpoint(tmp_path / "meta", model_type="CTLE_MLP", channels=2, freqs=8, with_cache=True)
    out = export_checkpoint_to_onnx(ckpt)
    meta = json.loads(out.with_suffix(".meta.json").read_text())

    assert meta["model_type"] == "CTLE_MLP"
    assert meta["num_channels"] == ref["channels"]
    assert meta["num_frequencies"] == ref["freqs"]
    assert meta["normalization_baked_in"] is True
    assert meta["input_feature_names"] == [f"f{i}" for i in range(ref["n_features"])]
    assert meta["channel_names"] == [f"ch{i}" for i in range(ref["channels"])]
    assert len(meta["frequency_hz"]) == ref["freqs"]

    # The documented reshape (numpy row-major (C, F)) must recover the (C, F) curve.
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, ref["n_features"])).astype(np.float32)
    flat = ort.InferenceSession(str(out)).run(["prediction"], {"input_features": x})[0][0]
    as_cf = flat.reshape(meta["num_channels"], meta["num_frequencies"])
    expected_cf = _reference(ref, x)[0].reshape(ref["channels"], ref["freqs"])
    assert np.allclose(as_cf, expected_cf, atol=1e-3)


def test_missing_cache_still_exports_with_empty_axis_meta(tmp_path: Path) -> None:
    ckpt, _ = _make_checkpoint(tmp_path / "nocache", model_type="CTLE_MLP", with_cache=False)
    out = export_checkpoint_to_onnx(ckpt)
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    # Export succeeds; axis labels are empty placeholders when no cache is present.
    assert meta["frequency_hz"] == []
    assert meta["num_channels"] == 2
