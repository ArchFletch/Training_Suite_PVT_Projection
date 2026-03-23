# Surrogate Model Training Suite

`MLP_modeling_v2` is the working repository for the current `Surrogate Model Training Suite` effort. The codebase now spans four connected areas:

- a source-based MLP training workflow
- a PySide6 desktop GUI for scan, suggestion, baseline, and self-transfer workflows
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

Primary code lives in [`xfmr_v2/`](xfmr_v2/), with thin root-level entrypoints for local development.

### 2. Desktop GUI

The GUI provides a technical desktop workflow for:

- environment detection
- dataset scan and preview
- suggested baseline and transfer settings
- editable baseline and self-transfer controls
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
python prepare_cache.py
python suggest_initial_settings.py
python quick_search.py
python train_baseline.py
python run_self_transfer.py --base-run-dir <path-to-compatible-baseline-run>
```

The data entrypoints also support separate input-feature-table and ground-truth paths:

```powershell
python prepare_cache.py --input-feature-path <path-to-log.txt> --ground-truth-data-dir <path-to-SPData>
python suggest_initial_settings.py --input-feature-path <path-to-log.txt> --ground-truth-data-dir <path-to-SPData>
python quick_search.py --input-feature-path <path-to-log.txt> --ground-truth-data-dir <path-to-SPData>
python train_baseline.py --input-feature-path <path-to-log.txt> --ground-truth-data-dir <path-to-SPData>
```

If neither `--input-feature-path` nor `--ground-truth-data-dir` is provided, the scripts continue to support the legacy `--data-root` layout.

### Desktop GUI

Start the GUI from the source checkout with:

```powershell
python launch_gui.py
```

The GUI can scan datasets, generate recommended settings, run baseline training, run self-transfer learning, and manage seat checkout against an on-prem license server. Quick search is currently not exposed in the GUI.

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
