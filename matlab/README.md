# Using a trained surrogate model in MATLAB

A two-step workflow: export the trained checkpoint to ONNX (Python), then load and
run it in MATLAB. Normalization is baked into the exported graph, so MATLAB feeds
raw design parameters and gets physical-unit curves back — no scaling math needed.

## 1. Export the checkpoint to ONNX (Python)

Run from the `GUI` directory (where `xfmr_v2` is importable):

```bash
python -m xfmr_v2.export_onnx path/to/best_model.pt
# optional: -o out.onnx   --check (verify vs torch)   --no-norm (export bare net)
```

This writes two files next to the checkpoint:

- `best_model.onnx` — the network, with input/output normalization folded in.
- `best_model.meta.json` — input feature order + units, channel names/units,
  frequency axis, and the output reshape convention.

One-time Python deps: `pip install onnx onnxruntime` (`onnxruntime` only needed for
`--check`).

## 2. Predict in MATLAB

Requires the **Deep Learning Toolbox** and the **Deep Learning Toolbox Converter for
ONNX Model Format** support package.

```matlab
% X rows are design points, columns in meta.input_feature_names order/units.
% For CTLE: [CS_fF, LD_pH, M, RD, RS, MN]
[pred, meta] = predict_ctle("best_model.onnx", "best_model.meta.json", [180 95 4 1200 800 3]);

% pred is nSamples x nChannels x nFrequencies, in physical units.
```

See `demo_predict_ctle.m` for a runnable example that plots each channel against the
frequency axis.

## Notes

- **Input units matter.** Supply features in the same physical units the model was
  trained on (the export bakes in the training-time normalization). The required
  order/units are listed in the `.meta.json`.
- **Output layout.** The graph emits each sample flattened channel-major
  (`[ch1_f1..ch1_fF, ch2_f1..]`); `predict_ctle.m` reshapes it back to
  `(channel, frequency)` for you.
- **Older MATLAB releases.** `predict_ctle.m` uses `importNetworkFromONNX` (R2023b+)
  and falls back to `importONNXNetwork` on earlier releases. If you hit an opset or
  layer-import error, re-export with a lower `--opset` or ask for the self-contained
  `.mat` exporter (no toolbox required) as a fallback.
