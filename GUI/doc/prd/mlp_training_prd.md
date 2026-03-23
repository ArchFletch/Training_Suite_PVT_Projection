# PRD: MLP Training Workflow

## Product

Core modeling workflow for `Surrogate Model Traning Suite`.

## Goal

Turn a README-described dataset into reproducible surrogate-model runs with:

- cache preparation
- scan-only recommendations
- quick baseline refinement
- baseline training
- optional self-transfer learning
- saved artifacts and summaries

## Users

- Modeling engineer: prepares data, trains, inspects outputs
- Research engineer: tunes settings and compares runs
- Application engineer: reproduces and diagnoses workflows

## In Scope

- dataset README schema parsing
- raw-data loading and cache creation
- deterministic train/validation/test splits
- scan-only suggestion generation
- bounded quick search from the CLI
- baseline model training and evaluation
- self-transfer learning from a compatible baseline run
- checkpoints, plots, summaries, and output folders

## Out of Scope

- distributed training
- general AutoML
- experiment-management platform
- alternate model families beyond the current spectral MLP
- cloud execution

## Primary Workflow

1. User selects dataset inputs and output location.
2. Product validates the README schema and builds or reuses a cache.
3. Product computes scan-only baseline and transfer recommendations.
4. User optionally runs quick search to refine the baseline.
5. User runs baseline training.
6. User optionally runs self-transfer learning from a compatible baseline.
7. Product writes artifacts, summaries, checkpoints, and plots to disk.

## Product Requirements

### Dataset and Cache

- Dataset layout must be described by a README JSON schema.
- The workflow must reject missing or invalid schema fields early.
- Cache creation must be reusable across repeated runs.
- Cache mismatches must be detected and handled explicitly.

### Recommendations

- The product must provide scan-only recommendations without full training.
- The recommendation output must include:
  - baseline config
  - transfer config
  - candidate ranges
  - warnings or confidence indicators

### Quick Search

- Quick search must stay intentionally small and bounded.
- It must evaluate multiple short baseline trials.
- It must recommend a full baseline config and save the trial results.

### Baseline Training

- The product must support configurable architecture and optimization settings.
- Each run must save:
  - best checkpoint
  - summary JSON
  - training history
  - evaluation metrics
  - plots

### Self-Transfer Learning

- Transfer must start from a compatible baseline run.
- Compatibility must be validated before transfer begins.
- Transfer outputs must be stored separately from baseline outputs.

### Observability

- Long-running workflows must emit structured progress events.
- Saved artifacts must be readable by both CLI users and the GUI.

## Non-Functional Requirements

- Reproducible outputs from the same data, seed, and settings
- CPU fallback when CUDA is unavailable
- Stable machine-readable artifacts
- Clear failure messages for dataset, cache, and compatibility problems

## Success Criteria

- Users can prepare a valid cache from a dataset.
- Users can generate scan-only recommendations.
- Users can run quick search from the CLI.
- Users can complete baseline training and inspect saved outputs.
- Users can complete self-transfer from a compatible baseline run.
- The saved outputs are stable enough for GUI consumption.

## Key Risks

- weak schema validation leads to confusing data errors later
- recommendation confidence may be over-trusted if not surfaced clearly
- quick search may be mistaken for full HPO if its scope is not documented
- output-folder sprawl can become hard to navigate if conventions drift

## Source References

- project-root `README.md`
- project-root `xfmr_v2/dataset_schema.py`
- project-root `xfmr_v2/data.py`
- project-root `xfmr_v2/suggest.py`
- project-root `xfmr_v2/search.py`
- project-root `xfmr_v2/model.py`
- project-root `xfmr_v2/runner.py`
