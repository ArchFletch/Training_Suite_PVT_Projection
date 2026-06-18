# PRD: Desktop GUI

## Product

Desktop GUI for `Surrogate Model Training Suite`.

## Goal

Provide a desktop interface that lets technical users run the modeling
workflow without editing source code, while still exposing the controls,
validation, licensing state, and scientific outputs they care about.

## Users

- Modeling engineer: selects dataset paths, connects to the on-prem license server, and launches training
- Research engineer: reviews scan-only suggestions and tunes baseline and transfer settings
- Support engineer: reloads configs, checks environment readiness, and diagnoses transfer or licensing issues

## In Scope

- manual environment detection and readiness reporting
- dataset, output-folder, and cache-path selection
- auto-managed cache path with manual override
- dataset scan, schema validation, and preview
- scan-only suggestion flow
- editable baseline settings
- editable self-transfer settings
- separate baseline and self-transfer launch actions
- load/save config and last-session restore
- on-prem license server URL persistence
- connection testing, startup checkout gating, heartbeat state, and shutdown release
- live metrics, plots, progress, run log, and seat-state feedback

## Out Of Scope

- quick search execution in the current GUI
- code editing
- automatic driver or CUDA installation
- payment, procurement, or vendor-license issuance workflows
- customer license-server administration
- multi-user experiment management

## Main Window

- window title: `Surrogate Model Training Suite`
- minimum size: `1280 x 800`
- preferred working size: about `1440 x 900`
- layout: configuration on the left, monitoring on the right

### Top Bar

- app title and subtitle
- current run-name summary
- `Load Config`
- `Save Config`

### Left Pane

Ordered sections:

1. `License Server`
2. `Data Sources`
3. `Dataset Preview`
4. `Suggest Initial Settings`
5. `Training Settings` (includes the `Training Device` selector)
6. `Run Controls`

Available compute devices (CUDA GPUs and CPU) are detected automatically when
the app starts and offered in the `Training Device` dropdown inside
`Training Settings`; the selected device is used for baseline and transfer runs.

The left pane must remain vertically scrollable.

### Right Pane

Ordered sections:

1. `Current Metrics`
2. `Monitor Tabs`
3. `Run Log`

## Primary Workflows

### Workflow A: Licensed Baseline Training

1. User launches `Surrogate Model Training Suite`.
2. If a server URL is configured, the GUI tests connectivity and attempts startup checkout.
3. The GUI auto-detects available compute devices; the user picks the `Training Device` (e.g. `cuda:0`, `cuda:1`, `cpu`).
4. User selects input-feature file, ground-truth folder, and output folder.
5. User clicks `Scan Data` and reviews the preview.
6. User runs `Suggest Initial Settings` and optionally applies them.
7. User reviews or edits baseline parameters.
8. User clicks `Start Baseline Training`.
9. GUI shows live metrics, seat state, baseline plots, and run-log updates.

### Workflow B: Licensed Self-Transfer Training

1. User completes dataset scan and holds a valid seat.
2. User either trains a baseline in the current session or selects an existing compatible baseline run.
3. User reviews transfer parameters such as band count, iterations, and epochs per stage.
4. User clicks `Start Self-Transfer Learning`.
5. GUI shows transfer progress in the metrics area and plots MAE over frequency by transfer iteration.

## Product Requirements

### System And Environment

- Environment detection must be manual, not automatic on startup.
- The GUI must report detected GPU, CUDA readiness, PyTorch CUDA state, and backend device.
- Training device choice must remain visible to the user.
- The GUI must not silently attempt risky system-level installation steps.

### Licensing And Access

- The GUI must allow the user or customer IT to provide an on-prem server URL.
- The GUI must expose a connection test before work begins.
- If a server URL is configured, the GUI must attempt startup checkout before enabling new training work.
- Failed seat checkout must block new runs with a clear, user-facing message.
- The GUI must maintain background heartbeats while a seat is active.
- Temporary heartbeat loss must enter a warning or grace state before the GUI requires reconnection.
- GUI shutdown should attempt a best-effort release.

### Data Sources And Scan

- The GUI must allow separate selection of:
  - input-feature file
  - ground-truth data folder
  - model output folder
- The cache path must auto-follow the current run configuration unless manually overridden.
- `Scan Data` must validate selected paths, auto-detect the dataset README, validate schema metadata, and populate a preview table.
- Scan progress must be visible through status text, progress bar movement, and run-log messages.
- If an auto-managed cache already exists for mismatched metadata, the GUI should rebuild it automatically.

### Suggest Initial Settings

- The GUI must support a scan-only recommendation flow with no training run required.
- It must display:
  - confidence level
  - diagnostics
  - suggested baseline settings
  - suggested transfer settings
  - warnings and rationale
- Users must be able to apply suggested settings into the editable forms without locking those forms.

### Training Settings

- Baseline settings must expose optimization, split, and architecture parameters.
- Transfer settings must expose:
  - base model source
  - compatibility state
  - number of bands
  - iterations
  - epochs per stage
  - optimization controls
- Existing baseline runs must be compatibility-checked before transfer can start.
- Transfer notes must explain band trimming when the frequency count does not divide evenly by the selected number of bands.

### Run Controls

- Baseline and self-transfer must be separate actions.
- Long-running work must run outside the UI thread.
- `Stop` must request a graceful stop rather than hard-killing the process.
- Controls that would conflict with an active task must lock while that task is running.

### Monitoring

- The GUI must show:
  - current phase
  - primary and secondary progress
  - best metric
  - train and validation loss where relevant
  - average MAE
  - elapsed time
  - ETA
  - run log
  - current seat-state feedback
- Baseline monitoring must show:
  - training loss vs epoch
  - MAE over frequency
- Transfer monitoring must show:
  - MAE over frequency by transfer iteration
- The metrics strip must remain phase-aware so scan, suggestion, baseline, and transfer can reuse the same summary area.

### Persistence

- The GUI must save and load JSON config files.
- The GUI should restore the last session when possible.
- Saved config and state should capture:
  - data paths
  - run name
  - cache override state
  - baseline settings
  - transfer settings
  - license server URL

## Validation Rules

The GUI should block a new run when:

- the input-feature file is missing
- the ground-truth folder is missing
- the output folder is missing
- dataset scan has not been completed
- training split fractions are invalid
- self-transfer is requested without a usable baseline model
- a selected baseline run is incompatible with the current cache or dataset metadata
- transfer band count exceeds the detected frequency count
- a server URL is configured but no healthy seat is currently checked out

## Error Handling

Error messaging should be explicit, actionable, and as plain-language as possible.

Representative cases:

- the input-feature file could not be parsed
- no valid ground-truth files were found
- the dataset README or schema metadata could not be validated
- a cache file conflicts with the current data sources
- self-transfer requires a current-session baseline or a compatible existing baseline run
- the selected number of transfer bands is not valid for the scanned dataset
- the license server is unreachable
- all floating seats are currently in use

## Non-Functional Requirements

- UI must stay responsive during scans and training
- backend logic must stay outside widget code
- saved state must remain readable and supportable
- CPU fallback must be visible when CUDA is unavailable
- progress feedback must be good enough that long-running tasks do not look frozen
- packaged Windows runs must keep writable state outside the install directory

## Success Criteria

- Users can complete baseline training from the GUI without touching code.
- Users can scan a dataset and apply recommended settings.
- Users can run self-transfer from the current session or an existing compatible baseline.
- Users can connect to an on-prem license server and start work when a seat is granted.
- The GUI remains responsive while work is running.
- Saved configs are good enough to replay a workflow later.

## Current Product Boundary

Quick search is currently a CLI-only workflow. If it returns to the GUI later,
that should be treated as a separate feature addition rather than assumed
desktop-GUI scope.

## Key Risks

- too many advanced controls can make the screen feel heavy
- stale session state can confuse users if config restoration is not predictable
- transfer failures are frustrating if compatibility messages are vague
- poorly explained licensing failures can block work even when the training core is healthy

## Source References

- project-root `launch_gui.py`
- project-root `xfmr_v2/gui_window.py`
- project-root `xfmr_v2/gui_backend.py`
- project-root `xfmr_v2/gui_workers.py`
- project-root `xfmr_v2/gui_theme.py`
- project-root `xfmr_v2/licensing/`
