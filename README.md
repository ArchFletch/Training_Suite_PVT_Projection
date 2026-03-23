# XFMR_2508_1x1_DiffXY Dataset

This README serves two purposes:

- dataset documentation for `XFMR_2508_1x1_DiffXY`
- an exemplar template for future datasets prepared for this training pipeline

## Expected Layout

```text
XFMR_2508_1x1_DiffXY/
  README.md
  log.txt
  SPData/
    0.s4p
    1.s4p
    2.s4p
    ...
```

## Input-Feature and Ground-Truth Conventions

- `log.txt` contains one Python-style list per line.
- The order of values in each row must exactly match `input_feature.columns`.
- `input_feature.sample_id_column` identifies the value used to match one input-feature row to one ground-truth file.
- The ground-truth filename must be `<sample_id><file_extension>`, for example `17.s4p`.
- In the JSON schema, the top-level `ground_truth` section describes the ground-truth training targets.
- The current code supports Touchstone ground-truth files through `ground_truth.format: "touchstone"`.
- Supported `ground_truth.ground_truth_parts` values are `re`, `im`, `mag`, `db`, and `angle_deg`.
- Set `drop_first_frequency` to `false` when the first point, including DC or 0 Hz, should be kept in training.

## Dataset-Specific Notes

- `rax`, `ray`, `rbx`, and `rby` are independent geometry parameters in this dataset.
- `gapa`, `gapb`, and `batch` are present in `log.txt` but are not used as model input-feature columns.
- `index` is the row-to-file mapping key.
- The current training target uses 12 channels, formed from 6 S-parameters with real and imaginary parts.

## Machine-Readable Schema

The training code reads the fenced JSON block below directly. Future datasets can copy this structure and change the values.

```json
{
  "dataset_name": "XFMR_2508_1x1_DiffXY",
  "input_feature": {
    "columns": [
      "rax",
      "ray",
      "rbx",
      "rby",
      "na",
      "nb",
      "wida",
      "widb",
      "gapa",
      "gapb",
      "opena",
      "openb",
      "outa",
      "outb",
      "exta",
      "extb",
      "dist",
      "ratio",
      "outbound",
      "index",
      "batch"
    ],
    "feature_columns": [
      "rax",
      "ray",
      "rbx",
      "rby",
      "na",
      "nb",
      "wida",
      "widb",
      "opena",
      "openb",
      "outa",
      "outb",
      "exta",
      "extb",
      "dist",
      "ratio",
      "outbound"
    ],
    "sample_id_column": "index"
  },
  "ground_truth": {
    "format": "touchstone",
    "file_extension": ".s4p",
    "ground_truth_parameters": [
      "S11",
      "S12",
      "S13",
      "S14",
      "S33",
      "S34"
    ],
    "ground_truth_parts": [
      "re",
      "im"
    ],
    "drop_first_frequency": false
  }
}
```

## Field Guide for Future Datasets

- `dataset_name`: any descriptive dataset name
- `input_feature.columns`: every column in the raw input-feature row, in exact file order
- `input_feature.feature_columns`: the subset used as model input-feature columns
- `input_feature.sample_id_column`: column that maps a row to its ground-truth filename
- `ground_truth.format`: currently must be `touchstone`
- `ground_truth.file_extension`: file suffix such as `.s2p`, `.s4p`, or another `.sNp`
- `ground_truth.ground_truth_parameters`: S-parameter names to extract, such as `S11` or `S34`
- `ground_truth.ground_truth_parts`: channel decomposition used for training targets
- `ground_truth.drop_first_frequency`: set to `true` only if the first frequency point is invalid and should be skipped
