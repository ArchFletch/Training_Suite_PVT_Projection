"""Export a trained checkpoint to ONNX for use in MATLAB (or any ONNX runtime).

A training checkpoint (``best_model.pt``) stores the raw network weights *plus* the
normalization statistics that live outside the ``nn.Module`` (input mean/std, per
channel/frequency target mean/std).  The network alone therefore predicts in
*normalized* space and cannot be used standalone.

This exporter wraps the network so the exported graph is end-to-end:

    raw design parameters  ->  [normalize] -> network -> [denormalize] -> physical curves

so MATLAB only has to call ``predict`` -- no pre/post math on the MATLAB side.  A
sidecar ``<out>.meta.json`` records the input feature order, channel names/units, and
the frequency axis so the MATLAB demo can label and reshape the output.

Run it from the ``GUI`` directory (where ``xfmr_v2`` is importable)::

    python -m xfmr_v2.export_onnx path/to/best_model.pt
    python -m xfmr_v2.export_onnx path/to/best_model.pt -o ctle.onnx --check

The forward signature of every model is ``(input_features, frequency)`` but
``frequency`` is unused (the models predict all frequency points at once), so the
exported graph has a single input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .runner import build_model


class _ExportWrapper(nn.Module):
    """Wrap a trained net so the ONNX graph maps raw inputs to physical outputs.

    The output is flattened to ``(batch, channels * num_frequencies)`` (channel-major:
    all frequencies of channel 0, then channel 1, ...).  A 2-D output imports far more
    cleanly into MATLAB than a 3-D one; the MATLAB demo reshapes it back to
    ``(channels, num_frequencies)`` using the dimensions recorded in the meta sidecar.
    """

    def __init__(
        self,
        model: nn.Module,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        log10_channel_mask: list[bool],
        bake_normalization: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.bake_normalization = bake_normalization
        self.num_channels, self.num_frequencies = target_mean.shape

        def _buf(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(np.asarray(arr, dtype=np.float32))

        # Input stats broadcast over (batch, features); target stats over (batch, C, F).
        self.register_buffer("input_mean", _buf(input_mean))
        self.register_buffer("input_std", _buf(input_std))
        self.register_buffer("target_mean", _buf(target_mean).unsqueeze(0))
        self.register_buffer("target_std", _buf(target_std).unsqueeze(0))
        # log10 channels are stored as log10(value); the inverse is 10**x.  Encoded as a
        # (1, C, 1) float mask so the graph stays branch-free and export-friendly.
        mask = torch.tensor([1.0 if m else 0.0 for m in log10_channel_mask], dtype=torch.float32)
        self.register_buffer("log10_mask", mask.view(1, -1, 1))
        # `frequency` is required by the forward signature but ignored by the models.
        self.register_buffer("_unused_frequency", torch.zeros(1, dtype=torch.float32))

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        x = input_features
        if self.bake_normalization:
            x = (x - self.input_mean) / self.input_std
        y = self.model(x, self._unused_frequency)  # (batch, C, F), normalized space
        if self.bake_normalization:
            y = y * self.target_std + self.target_mean
            # Undo log10 only on the channels that were stored in log space.
            y = self.log10_mask * torch.pow(10.0, y) + (1.0 - self.log10_mask) * y
        return y.reshape(y.shape[0], self.num_channels * self.num_frequencies)


def _get(checkpoint: dict, *names: str) -> Any:
    """Return the first present key among ``names`` (tolerates legacy field names)."""
    for name in names:
        if name in checkpoint:
            return checkpoint[name]
    raise KeyError(f"None of {names} found in checkpoint (have: {sorted(checkpoint)}).")


def _load_axis_metadata(cache_path: str | None, num_channels: int) -> dict[str, Any]:
    """Read the frequency axis, units, and transforms from the training cache.

    The cache is the source of truth for these labels.  If it has moved or been
    deleted, export still succeeds with empty/placeholder metadata.
    """
    meta: dict[str, Any] = {"channel_units": [], "channel_transforms": [], "frequency_hz": [], "sweep_label": ""}
    if not cache_path or not Path(cache_path).exists():
        return meta
    with np.load(cache_path, allow_pickle=False) as data:
        if "frequency_hz" in data:
            meta["frequency_hz"] = data["frequency_hz"].astype(float).tolist()
        if "channel_units" in data:
            meta["channel_units"] = data["channel_units"].astype(str).tolist()
        if "channel_transforms" in data:
            meta["channel_transforms"] = data["channel_transforms"].astype(str).tolist()
        if "sweep_label" in data:
            meta["sweep_label"] = str(data["sweep_label"])
    return meta


def export_checkpoint_to_onnx(
    checkpoint_path: str | Path,
    out_path: str | Path | None = None,
    *,
    bake_normalization: bool = True,
    opset: int = 17,
    check: bool = False,
) -> Path:
    """Export ``checkpoint_path`` to ONNX and write a ``<out>.meta.json`` sidecar.

    Returns the path to the written ``.onnx`` file.
    """
    checkpoint_path = Path(checkpoint_path)
    out_path = Path(out_path) if out_path else checkpoint_path.with_suffix(".onnx")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    active_names = list(_get(checkpoint, "active_input_feature_names", "active_feature_names"))
    channel_names = list(checkpoint["target_channel_names"])
    input_mean = np.asarray(_get(checkpoint, "input_feature_mean", "input_mean"), dtype=np.float32)
    input_std = np.asarray(_get(checkpoint, "input_feature_std", "input_std"), dtype=np.float32)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    num_channels, num_frequencies = target_mean.shape

    axis_meta = _load_axis_metadata(config.get("cache_path"), num_channels)
    transforms = axis_meta["channel_transforms"] or [""] * num_channels
    log10_mask = [t == "log10" for t in transforms]

    model = build_model(
        config.get("model_type", "FlatMLP"),
        num_frequencies=num_frequencies,
        input_feature_dim=len(active_names),
        ground_truth_channels=num_channels,
        width=int(config["width"]),
        depth=int(config["depth"]),
    )
    model.load_state_dict(_get(checkpoint, "model_state", "model_state_dict"))
    model.eval()

    wrapper = _ExportWrapper(
        model, input_mean, input_std, target_mean, target_std, log10_mask, bake_normalization
    ).eval()

    example = torch.zeros(1, len(active_names), dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (example,),
        str(out_path),
        input_names=["input_features"],
        output_names=["prediction"],
        # Let MATLAB feed any number of samples at once.
        dynamic_axes={"input_features": {0: "batch"}, "prediction": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )

    sidecar = {
        "model_type": config.get("model_type", "FlatMLP"),
        "input_feature_names": active_names,
        "channel_names": channel_names,
        "channel_units": axis_meta["channel_units"],
        "channel_transforms": transforms,
        "num_channels": int(num_channels),
        "num_frequencies": int(num_frequencies),
        "frequency_hz": axis_meta["frequency_hz"],
        "sweep_label": axis_meta["sweep_label"],
        "normalization_baked_in": bool(bake_normalization),
        # When normalization is NOT baked in, MATLAB needs these to do it manually.
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "output_layout": "row-major (channel, frequency); reshape to [num_frequencies, num_channels].' in MATLAB",
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(sidecar, indent=2))

    if check:
        _verify(wrapper, str(out_path), len(active_names))

    return out_path


def _verify(wrapper: nn.Module, onnx_path: str, input_dim: int, num_samples: int = 8) -> None:
    """Compare ONNX runtime output against the torch wrapper on random inputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[check] onnxruntime not installed; skipping numerical verification.")
        return
    # Deterministic-but-varied inputs without depending on global RNG state.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((num_samples, input_dim)).astype(np.float32)
    with torch.no_grad():
        torch_out = wrapper(torch.from_numpy(x)).numpy()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(["prediction"], {"input_features": x})[0]
    max_abs = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"[check] max |torch - onnx| = {max_abs:.3e} over {num_samples} samples", end=" ")
    print("-> OK" if max_abs < 1e-3 else "-> WARNING: large mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Path to best_model.pt")
    parser.add_argument("-o", "--out", help="Output .onnx path (default: alongside the checkpoint)")
    parser.add_argument(
        "--no-norm",
        action="store_true",
        help="Export the bare network (normalized space); do normalization in MATLAB using the meta JSON.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--check", action="store_true", help="Verify ONNX output matches torch (needs onnxruntime)")
    args = parser.parse_args()

    out = export_checkpoint_to_onnx(
        args.checkpoint,
        args.out,
        bake_normalization=not args.no_norm,
        opset=args.opset,
        check=args.check,
    )
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
