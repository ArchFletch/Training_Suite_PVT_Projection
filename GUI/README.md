# Surrogate Model Training Suite

`MLP_modeling_v2` is the working repository for the current `Surrogate Model Training Suite` effort. The codebase now spans four connected areas:

- a source-based MLP training workflow
- a PySide6 desktop GUI for scan, suggestion, and baseline training workflows
- an on-prem floating license server plus internal vendor-license tooling
- packaging and release-validation assets for Windows and Linux delivery

The root README is intended to be the project front door for collaborators. Detailed architecture, product, packaging, and rehearsal notes live under [`doc/`](doc/).

## Current Status

- The source checkout supports CLI training workflows, the desktop GUI, the license-server API and admin CLI, vendor-side license issuance tooling, and automated tests.
- The desktop GUI is a thin orchestration layer over the training core. Quick search remains CLI-only in the current product boundary.
- The on-prem licensing stack is implemented in source form and covered by tests.
- Windows GUI packaging and the Windows license-server service bundle are documented and scaffolded, but the clean-install delivery slice is still an active implementation and validation area rather than a fully closed release lane.

## Environment Assumptions

- The documented baseline stack is Python 3.12 plus the libraries described under [`doc/tech_stack/`](doc/tech_stack/).
- This repository does not yet expose one finalized root-level environment manifest that cleanly separates GUI, training, and server packaging inputs.
- In practice, source-checkout workflows assume you already have the relevant local dependencies installed for the part of the system you are working on.

## Product Components

### 1. MLP Training Workflow

The training core turns a README-described dataset into reproducible modeling runs with:

- cache preparation
- scan-only recommendations
- bounded quick search
- baseline training
- optional self-transfer learning
- saved checkpoints, JSON summaries, and plots

Four architectures are available: `SpectraNet` (flat dense net), `SpectraHydra`
(shared encoder with per-channel heads), `SpectraHydraProj` (SpectraHydra plus a
learned projection of PVT corner columns — pick the corner/condition feature columns
such as temperature, supply, and process one-hots, and the model concatenates their
trained embedding onto the inputs; the projection also trains live inside every band
submodel during self-transfer), and `SpectraTrunk` (a frequency-trunk model from the
M:N transformer study: a context MLP compresses the inputs to a frequency-flat
context, and one weight-shared residual trunk is evaluated per frequency point with a
33-dim Fourier embedding of the frequency coordinate, so the spectrum is predicted as
a smooth function of frequency rather than one output slot per point. Width sets the
trunk width, Depth the number of residual blocks; the study's reference recipe was
width 512, depth 4, batch 32, AdamW ~1e-3 with the cosine scheduler).

The scan-only recommender detects PVT corner structure on its own and will recommend
`SpectraHydraProj` with the corner columns already selected. It looks for the shape a
corner campaign has — a few low-cardinality columns that vary while the design knobs
stay fixed, with each design appearing once per corner — rather than for particular
column names, so it also works on datasets whose corner columns are not called
`Temp_C` or `VDD`. A low-cardinality *design* knob is not mistaken for a corner,
because a corner column must vary within a design and a design knob does not. Columns
identified from structure alone but not recognizable by name are flagged in the
warnings for review. Data-starved datasets (the conservative capacity tier) keep the
smaller `SpectraNet`: there is no `SpectraNet` + projection model, and such a dataset
cannot support a corner embedding.

For PVT-style datasets where the same design is re-simulated across corners, the optional
design-level split (command line only, any model type) keeps every corner row of a design
in the same train/validation/test fold. The default row-level split would place a design
at one corner in train and the same design at another corner in test, which leaks design
information and makes test error look better than it is. (For randomized campaigns where
each design appears at exactly one corner — the datasets the GUI targets — the row-level
split is already design-disjoint, so the GUI does not offer this option.)

Specify it by naming the columns that **identify a design** — the geometry parameters —
via `train_baseline.py --split-design-columns` or `--config-json`. Everything else
is then treated as corner-varying. The older `--split-corner-columns` form (name the
corner columns instead) still works, but it must cover *every* column that moves with the
corner, derived ones included: physics anchors, a frozen corner embedding (`e0`…`e15`),
and so on. Miss one and each design fractures into per-corner designs, which is a
row-level split wearing a design-level label. The engine now folds provably
corner-determined columns in automatically, refuses a grouping that gives every row its
own design, and warns when no design reaches every corner in the data — but naming the
design columns avoids the whole class of mistake, because an omission there merges
designs (coarser, still leak-free) instead of splintering them.

Primary code lives in [`xfmr_v2/`](xfmr_v2/), with thin root-level entrypoints for local development.

### 2. Desktop GUI

The GUI provides a technical desktop workflow for:

- environment detection
- dataset scan and preview
- suggested baseline settings
- editable baseline training controls
- live metrics, plots, and run logs
- on-prem license checkout, heartbeat, and release

Launch it from the source checkout with:

```powershell
python launch_gui.py
```

### 3. On-Prem Floating License Server

The repository includes a customer-deployed floating license server with:

- FastAPI client endpoints for checkout, heartbeat, release, and status
- a CLI-first admin workflow for customer IT
- signed JSON license import and verification
- SQLite-backed seat and audit persistence

The customer-facing server code lives in [`license_server/`](license_server/). Internal-only license issuance tooling lives in [`license_vendor/`](license_vendor/).

### 4. Packaging And Deployment

The repo also tracks packaging assets for:

- a packaged Windows desktop app
- a Windows license-server service bundle
- Linux GUI and server bundle support
- clean-install rehearsal outside the source checkout

Tracked packaging assets live in [`packaging/`](packaging/).

## Repository Map

- [`xfmr_v2/`](xfmr_v2/): training core, GUI backend, GUI window, path policy, and desktop-side licensing code
- [`license_server/`](license_server/): on-prem floating license-server package
- [`license_vendor/`](license_vendor/): internal vendor tooling for issuing signed eval and paid licenses
- [`packaging/`](packaging/): GUI and server packaging scripts, templates, and bundle assets
- [`tests/`](tests/): source-based regression coverage for training, GUI, licensing, vendor tooling, and path policy
- [`doc/`](doc/): architecture, PRDs, packaging plans, customer notes, operations notes, and install rehearsals

## Common Source-Checkout Workflows

### Training CLI

The root entrypoints keep the common training workflows easy to run from a source checkout:

```powershell
python prepare_cache.py <path-to-dataset-folder>
python suggest_initial_settings.py
python quick_search.py
python train_baseline.py
python run_self_transfer.py --base-run-dir <path-to-compatible-baseline-run>
```

A dataset is either one `.npz` file or a folder to auto-detect. Naming the `.npz`
directly is unambiguous, so it is the standard input format and the only one the GUI
offers; the folder forms remain available from the command line.

Pointing at a **folder** auto-detects the format from its contents: SPData/Touchstone
(`log.txt` + a folder of `.sNp` files), Cadence CSV (`*.csv`), or a prebuilt `.npz`
already holding `features` / `targets` / `frequency_hz` arrays — either this engine's
own cache layout or a barer bundle from an offline preparation script. The
prebuilt-array case is checked last, so a raw source always wins when a folder holds
both, and a folder holding several bundles has to pick one.

Pointing at an **`.npz` file** skips all of that and adopts that file, so no raw
source can shadow it and no preference order applies. The cache is never written over
the file being read.

The other entrypoints build the cache on demand when it is missing — point them at the
dataset with `--data-root <path-to-.npz-or-folder>` (or the explicit
`--input-feature-path` / `--ground-truth-data-dir` paths, which are used to locate the
dataset folder).

### Desktop GUI

Start the GUI from the source checkout with:

```powershell
python launch_gui.py
```

The GUI can scan datasets, generate recommended settings, run baseline training, and manage seat checkout against an on-prem license server.

Frequency-domain self-transfer learning is not exposed in the GUI, and neither is quick
search. The engine still implements self-transfer in full, including the per-band
corner projection; drive it from the command line with `run_self_transfer.py`. The GUI's
suggestion table lists baseline settings only, because listing transfer settings would
offer values with no control to apply them to. Loading a session that set up transfer
logs a note saying so, since the run now trains one model across the whole frequency
range rather than one per band.

Start Baseline Training is disabled while a task is
running and whenever no license seat is checked out. A disabled button carries the
reason as its tooltip, because a disabled button cannot be clicked and so cannot raise
the dialog that would otherwise explain itself. The remaining prerequisites (a dataset
file, a completed scan, valid split fractions, corner columns for the projection model)
are checked on click and reported in a dialog. Changing the Dataset File clears the
previous scan, so a run cannot start against a dataset that was never scanned.

The Baseline tab shows only the settings the selected model actually reads. The PVT
Corner Columns and Corner Projection Width rows appear for `SpectraHydraProj` and are
hidden for every other model, rather than being shown greyed out. Which model types
carry the corner projection is defined once in the engine
(`runner.PROJECTION_MODEL_TYPES`), so the form cannot drift from what the model builder
accepts.

Data loading is one field: **Dataset File**, a `.npz` holding `features` / `targets` /
`frequency_hz`. The folder picker and the Advanced disclosure (explicit input-feature
file, ground-truth folder, and cache file) have been removed — one file names the whole
dataset, so the two path overrides had nothing left to override, and the cache is
derived from the Output Folder and Run Name rather than chosen. All three remain
available from the command line via `train_baseline.py --input-feature-path`,
`--ground-truth-data-dir` and `--cache-path`, and raw folder layouts via `--data-root`.
Loading a session or config that set any of them logs a note saying so instead of
silently ignoring it.

### License-Server Admin CLI

Common source-based admin commands:

```powershell
python -m license_server.cli --help
python -m license_server.cli init
python -m license_server.cli export-request license_request.json
python -m license_server.cli show-status
python -m license_server.cli show-audit
```

To run the HTTP service in a source environment:

```powershell
python -m uvicorn license_server.main:app --host 0.0.0.0 --port 27850
```

### Internal Vendor Tooling

Internal vendor tooling stays separate from the customer server:

```powershell
python -m license_vendor --help
```

Representative eval-license issuance flow:

```powershell
python -m license_vendor init-key --key-id 2026-01 --output vendor_signing_key.json
python -m license_vendor issue-eval --request-file license_request.json --signing-key-file vendor_signing_key.json --company-name "Acme Design House" --seat-count 2 --output eval_license.json --issuance-log issuance_log.jsonl
```

## Dataset Contract

Custom datasets should include a `README.md` or `README.txt` in the dataset root with one fenced `json` block that defines:

- the full input-feature column order in `log.txt`
- the subset of input-feature columns used as model inputs
- the sample ID column that maps each row to one ground-truth file
- the ground-truth file format and extension
- the S-parameter channels and ground-truth parts used as training targets

Use top-level `input_feature` and `ground_truth` sections in that schema block.

## Testing And Validation

Fast regression coverage is source-based:

```powershell
pytest
```

Additional source-oriented rehearsal assets include:

- [`scripts/smoke_test_license_server.ps1`](scripts/smoke_test_license_server.ps1)
- GUI, license-server, and vendor-focused test suites under [`tests/`](tests/)

Customer-facing release readiness for packaged artifacts is documented separately from the source-based smoke flow.

## Runtime Path Policy

The documented runtime boundary is:

- source checkout: developer-friendly repo-relative outputs under `artifacts/`
- packaged GUI: per-user state under `%APPDATA%`, `%LOCALAPPDATA%`, and `Documents`
- packaged license server: install root under `C:\Program Files\MLP License Server\` and runtime data under `%PROGRAMDATA%\MLP License Server\`

Packaged installs must not rely on repo-relative paths or write state into install directories.

## Documentation Map

### Architecture

- [`doc/architecture/overview.md`](doc/architecture/overview.md)
- [`doc/architecture/mlp_training_architecture.md`](doc/architecture/mlp_training_architecture.md)
- [`doc/architecture/gui_architecture.md`](doc/architecture/gui_architecture.md)
- [`doc/architecture/license_server_architecture.md`](doc/architecture/license_server_architecture.md)
- [`doc/architecture/deployment_architecture.md`](doc/architecture/deployment_architecture.md)

### Product Requirements

- [`doc/prd/mlp_training_prd.md`](doc/prd/mlp_training_prd.md)
- [`doc/prd/gui_prd.md`](doc/prd/gui_prd.md)
- [`doc/prd/license_server_prd.md`](doc/prd/license_server_prd.md)
- [`doc/prd/deployment_prd.md`](doc/prd/deployment_prd.md)

### Implementation And Packaging

- [`doc/implementation/gui_packaging_plan.md`](doc/implementation/gui_packaging_plan.md)
- [`doc/implementation/windows_packaging_and_clean_install.md`](doc/implementation/windows_packaging_and_clean_install.md)
- [`doc/implementation/license_server_implementation_checklist.md`](doc/implementation/license_server_implementation_checklist.md)
- [`doc/implementation/license_server_repository_scaffold.md`](doc/implementation/license_server_repository_scaffold.md)

### Customer, Operations, And Rehearsal Notes

- [`doc/customer/gui_install_run.md`](doc/customer/gui_install_run.md)
- [`doc/operations/license_server_customer_install.md`](doc/operations/license_server_customer_install.md)
- [`doc/integration/windows_packaged_install_rehearsal.md`](doc/integration/windows_packaged_install_rehearsal.md)
- [`doc/integration/license_server_mvp_rehearsal.md`](doc/integration/license_server_mvp_rehearsal.md)

### Tech Stack References

- [`doc/tech_stack/tech_stack_mlp_training.md`](doc/tech_stack/tech_stack_mlp_training.md)
- [`doc/tech_stack/tech_stack_gui.md`](doc/tech_stack/tech_stack_gui.md)
- [`doc/tech_stack/tech_stack_license_server.md`](doc/tech_stack/tech_stack_license_server.md)
- [`doc/tech_stack/tech_stack_deployment.md`](doc/tech_stack/tech_stack_deployment.md)

## Local And Generated Files

The repository intentionally treats many runtime outputs as local state. Common generated paths include:

- `artifacts/`
- `runtime/`
- `build/`
- `dist/`
- deployment and install-smoke outputs outside the repo

Before committing, double-check that large checkpoints, logs, screenshots, and temporary validation files are not being added unintentionally.
