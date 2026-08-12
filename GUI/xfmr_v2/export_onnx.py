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

Every model takes a single ``input_features`` argument and predicts all frequency
points at once, so the exported graph has a single input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .runner import build_model, resolve_projection_kwargs


def _apply_log10_inverse(y: torch.Tensor, log10_mask: torch.Tensor, has_log10: bool) -> torch.Tensor:
    """Undo a log10 transform on the masked channels of ``y`` (shape (batch, C, F)).

    ``torch.where`` is used rather than ``mask * 10**y + (1-mask) * y`` because the
    latter computes ``10**y`` for every channel, and a non-log10 channel with a large
    denormalized value (e.g. phase in degrees) overflows to +inf -> ``0 * inf`` = NaN.
    ``where`` is a select, so inf in the unselected lanes is simply discarded.  When no
    channel is log10 (e.g. CTLE), the pow op is skipped entirely.
    """
    if not has_log10:
        return y
    return torch.where(log10_mask, torch.pow(10.0, y), y)


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
        # (1, C, 1) bool mask used with torch.where (see forward for why a mask-multiply
        # would be wrong).  The pow op is only emitted when some channel needs it.
        self._has_log10 = any(log10_channel_mask)
        mask = torch.tensor([m for m in log10_channel_mask], dtype=torch.bool)
        self.register_buffer("log10_mask", mask.view(1, -1, 1))

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        x = input_features
        if self.bake_normalization:
            x = (x - self.input_mean) / self.input_std
        y = self.model(x)  # (batch, C, F), normalized space
        if self.bake_normalization:
            y = y * self.target_std + self.target_mean
            y = _apply_log10_inverse(y, self.log10_mask, self._has_log10)
        return y.reshape(y.shape[0], self.num_channels * self.num_frequencies)


class _TransferExportWrapper(nn.Module):
    """Stitch per-band transfer submodels into one end-to-end graph.

    Self-transfer learning trains one submodel per contiguous frequency band, each
    predicting its slice in the baseline's normalized target space.  This wrapper runs
    every band, concatenates the slices in band order, then denormalizes — producing
    the same ``(batch, channels * num_frequencies)`` output as :class:`_ExportWrapper`
    so the same MATLAB loader works unchanged.  Bands are concatenated in list order,
    which matches the ascending, contiguous index layout that training produces.
    """

    def __init__(
        self,
        submodels: list[nn.Module],
        input_mean: np.ndarray,
        input_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        log10_channel_mask: list[bool],
        bake_normalization: bool,
    ) -> None:
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        self.bake_normalization = bake_normalization
        self.num_channels, self.num_frequencies = target_mean.shape

        def _buf(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(np.asarray(arr, dtype=np.float32))

        self.register_buffer("input_mean", _buf(input_mean))
        self.register_buffer("input_std", _buf(input_std))
        self.register_buffer("target_mean", _buf(target_mean).unsqueeze(0))
        self.register_buffer("target_std", _buf(target_std).unsqueeze(0))
        self._has_log10 = any(log10_channel_mask)
        mask = torch.tensor([m for m in log10_channel_mask], dtype=torch.bool)
        self.register_buffer("log10_mask", mask.view(1, -1, 1))

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        x = input_features
        if self.bake_normalization:
            x = (x - self.input_mean) / self.input_std
        # Each submodel returns (batch, C, band_width); concatenate along frequency.
        y = torch.cat([sub(x) for sub in self.submodels], dim=2)
        if self.bake_normalization:
            y = y * self.target_std + self.target_mean
            y = _apply_log10_inverse(y, self.log10_mask, self._has_log10)
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

    # SpectraHydraProj checkpoints resolve their corner columns exactly as training
    # did: by name against the checkpoint's own active feature list. Other model
    # types get an empty dict. Old checkpoints lack the projection keys entirely,
    # which the .get defaults tolerate.
    projection_kwargs = resolve_projection_kwargs(
        config.get("model_type", "SpectraNet"),
        config.get("projection_columns"),
        int(config.get("projection_dim", 16) or 16),
        active_names,
        list(checkpoint.get("dropped_input_feature_names", [])),
    )
    model = build_model(
        config.get("model_type", "SpectraNet"),
        num_frequencies=num_frequencies,
        input_feature_dim=len(active_names),
        ground_truth_channels=num_channels,
        width=int(config["width"]),
        depth=int(config["depth"]),
        **projection_kwargs,
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
        "model_type": config.get("model_type", "SpectraNet"),
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
    if projection_kwargs:
        # Informative only: the graph input stays the raw feature vector; the
        # corner projection happens inside the exported network.
        sidecar["projection_columns"] = [active_names[i] for i in projection_kwargs["corner_indices"]]
        sidecar["projection_corner_indices"] = list(projection_kwargs["corner_indices"])
        sidecar["projection_dim"] = int(projection_kwargs["projection_dim"])
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(sidecar, indent=2))

    if check:
        _verify(wrapper, str(out_path), len(active_names))

    return out_path


def export_transfer_to_onnx(
    transfer_run_dir: str | Path,
    out_path: str | Path | None = None,
    *,
    baseline_checkpoint: str | Path | None = None,
    bake_normalization: bool = True,
    opset: int = 17,
    check: bool = False,
) -> Path:
    """Export a self-transfer (per-band) model to a single stitched ONNX graph.

    ``transfer_run_dir`` must contain ``final_submodels.pt``. Standalone transfer runs
    embed their own normalization stats, model type, channel names, and frequency axis,
    so no baseline is needed. Older runs that were seeded from a baseline resolve those
    from the baseline (via the transfer ``summary.json``'s ``base_run_dir``, or an
    explicit ``baseline_checkpoint``). Writes ``<out>.onnx`` + ``<out>.meta.json``; the
    output contract matches the baseline exporter, so the same MATLAB loader works.
    """
    transfer_run_dir = Path(transfer_run_dir)
    bundle = torch.load(transfer_run_dir / "final_submodels.pt", map_location="cpu", weights_only=False)
    states = bundle["states"]
    model_kwargs = dict(bundle["model_kwargs"])
    bands = [np.asarray(b, dtype=np.int64) for b in bundle["bands"]]

    if "target_mean" in bundle:
        # Standalone transfer run: the bundle's embedded normalization + metadata are
        # authoritative — they are what the submodels were trained with. An explicitly
        # passed baseline_checkpoint is redundant here and is ignored: mixing a foreign
        # baseline's stats or model_type with this bundle's states/model_kwargs would
        # export a wrong (or unbuildable) graph.
        model_type = bundle.get("model_type", "SpectraNet")
        active_names = list(bundle["active_input_feature_names"])
        channel_names = list(bundle["target_channel_names"])
        input_mean = np.asarray(bundle["input_feature_mean"], dtype=np.float32)
        input_std = np.asarray(bundle["input_feature_std"], dtype=np.float32)
        target_mean_full = np.asarray(bundle["target_mean"], dtype=np.float32)
        target_std_full = np.asarray(bundle["target_std"], dtype=np.float32)
        transforms = list(bundle.get("channel_transforms") or [""] * len(channel_names))
        units = list(bundle.get("channel_units") or [""] * len(channel_names))
        freq_full = np.asarray(bundle["frequency_hz"], dtype=float).tolist() if "frequency_hz" in bundle else []
        sweep_label = "Frequency (GHz)"
    else:
        # Legacy: the baseline run supplies model_type + normalization.
        if baseline_checkpoint is None:
            summary = json.loads((transfer_run_dir / "summary.json").read_text())
            base_run_dir = summary.get("base_run_dir")
            if not base_run_dir:
                raise ValueError(
                    "Transfer run has no embedded normalization and summary.json has no "
                    "'base_run_dir'; pass baseline_checkpoint explicitly."
                )
            baseline_checkpoint = Path(base_run_dir) / "best_model.pt"
        baseline_checkpoint = Path(baseline_checkpoint)
        base = torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
        config = base["config"]
        # The bundle's submodels were built from its own model_kwargs, so a bundle
        # model_type (when recorded) beats the baseline's — they must agree with
        # each other, not with a possibly retrained baseline.
        model_type = bundle.get("model_type") or config.get("model_type", "SpectraNet")
        active_names = list(_get(base, "active_input_feature_names", "active_feature_names"))
        channel_names = list(base["target_channel_names"])
        input_mean = np.asarray(_get(base, "input_feature_mean", "input_mean"), dtype=np.float32)
        input_std = np.asarray(_get(base, "input_feature_std", "input_std"), dtype=np.float32)
        target_mean_full = np.asarray(base["target_mean"], dtype=np.float32)
        target_std_full = np.asarray(base["target_std"], dtype=np.float32)
        axis_meta = _load_axis_metadata(config.get("cache_path"), target_mean_full.shape[0])
        transforms = axis_meta["channel_transforms"] or [""] * len(channel_names)
        units = axis_meta["channel_units"]
        freq_full = axis_meta["frequency_hz"]
        sweep_label = axis_meta["sweep_label"]

    # The transfer model covers the (possibly trimmed) union of band indices, in band order.
    eff_idx = np.concatenate(bands)
    target_mean = target_mean_full[:, eff_idx]
    target_std = target_std_full[:, eff_idx]
    num_channels = target_mean.shape[0]
    log10_mask = [t == "log10" for t in transforms]
    freq_eff = [float(freq_full[int(i)]) for i in eff_idx] if freq_full else []

    submodels: list[nn.Module] = []
    for band, state in zip(bands, states, strict=True):
        sub = build_model(model_type, num_frequencies=int(len(band)), **model_kwargs)
        sub.load_state_dict(state)
        sub.eval()
        submodels.append(sub)

    wrapper = _TransferExportWrapper(
        submodels, input_mean, input_std, target_mean, target_std, log10_mask, bake_normalization
    ).eval()

    out_path = Path(out_path) if out_path else (transfer_run_dir / "transfer_model.onnx")
    example = torch.zeros(1, len(active_names), dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (example,),
        str(out_path),
        input_names=["input_features"],
        output_names=["prediction"],
        dynamic_axes={"input_features": {0: "batch"}, "prediction": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )

    sidecar = {
        "model_type": f"{model_type} (self-transfer, {len(bands)} bands)",
        "input_feature_names": active_names,
        "channel_names": channel_names,
        "channel_units": units,
        "channel_transforms": transforms,
        "num_channels": int(num_channels),
        "num_frequencies": int(len(eff_idx)),
        "frequency_hz": freq_eff,
        "sweep_label": sweep_label,
        "normalization_baked_in": bool(bake_normalization),
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "transfer_bands": [[int(i) for i in band] for band in bands],
        "baseline_checkpoint": str(baseline_checkpoint) if baseline_checkpoint else None,
        "output_layout": "row-major (channel, frequency); reshape to [num_frequencies, num_channels].' in MATLAB",
    }
    if "corner_indices" in model_kwargs:
        corner_indices = [int(i) for i in model_kwargs["corner_indices"]]
        sidecar["projection_columns"] = [active_names[i] for i in corner_indices]
        sidecar["projection_corner_indices"] = corner_indices
        sidecar["projection_dim"] = int(model_kwargs.get("projection_dim", 16))
    out_path.with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2))

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
    parser.add_argument(
        "checkpoint",
        help="Path to best_model.pt (baseline), or a self-transfer run directory with --transfer.",
    )
    parser.add_argument("-o", "--out", help="Output .onnx path (default: alongside the checkpoint)")
    parser.add_argument(
        "--transfer",
        action="store_true",
        help="Treat the path as a self-transfer run dir and stitch its per-band submodels into one ONNX.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        help="(transfer only) baseline best_model.pt supplying normalization; default: resolved from summary.json.",
    )
    parser.add_argument(
        "--no-norm",
        action="store_true",
        help="Export the bare network (normalized space); do normalization in MATLAB using the meta JSON.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--check", action="store_true", help="Verify ONNX output matches torch (needs onnxruntime)")
    args = parser.parse_args()

    if args.transfer:
        out = export_transfer_to_onnx(
            args.checkpoint,
            args.out,
            baseline_checkpoint=args.baseline_checkpoint,
            bake_normalization=not args.no_norm,
            opset=args.opset,
            check=args.check,
        )
    else:
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
