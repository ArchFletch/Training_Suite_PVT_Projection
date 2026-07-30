# Windows Packaging And Clean-Install

## Goal

Define the documentation-level implementation contract for the next Windows
delivery slice:

- packaged GUI for design engineers
- packaged license-server service bundle for customer IT
- clean-install rehearsal outside the source checkout

## Deliverables

### GUI

- standalone Windows GUI build from `launch_gui.py`
- per-user Windows installer for `Surrogate Model Training Suite`
- silent-install and silent-uninstall support for release validation

### License Server

- Windows service bundle rooted at `C:\Program Files\MLP License Server\`
- bundled Python runtime or dedicated bundled virtual environment
- bundled WinSW wrapper and rendered service XML
- bundled install script and admin wrapper commands

### Validation

- packaged-install rehearsal guide
- timestamped rehearsal outputs outside the repo
- release checklist that proves the same behavior already covered by the source smoke flow

## Chosen Windows GUI Flow

1. Render the `pyside6-deploy` spec from the tracked template.
2. Build a standalone directory with `pyside6-deploy` + Nuitka.
3. Smoke the standalone directory before installer wrapping.
4. Wrap the standalone directory in an Inno Setup per-user installer.
5. Validate normal install, silent install, launch, runtime writes, and uninstall.

Current tracked assets:

- `packaging/windows/build_gui.ps1`
- `packaging/windows/build_installer.ps1`
- `packaging/windows/SurrogateModelTrainingSuite.iss`

## Chosen Windows Server Flow

1. Stage the `license_server` package, runtime dependencies, and service assets into a self-contained bundle.
2. Bundle the Python runtime or a dedicated packaged virtual environment with the service files.
3. Bundle WinSW directly so customer IT does not fetch it separately.
4. Install the service into `C:\Program Files\MLP License Server\`.
5. Keep runtime data, config, database, identity, and logs under `%PROGRAMDATA%\MLP License Server\`.
6. Expose customer-admin CLI actions through wrapper commands in the install root.

## Build Inputs

### GUI Inputs

- `launch_gui.py`
- `xfmr_v2/app_paths.py`
- `xfmr_v2/licensing/client_config.py`
- `packaging/gui/pysidedeploy.spec.in`
- `packaging/windows/*.ps1`
- `packaging/windows/SurrogateModelTrainingSuite.iss`

### Server Inputs

- `license_server/`
- `packaging/license_server/windows/`
- `packaging/license_server/shared/`
- bundled runtime dependencies for FastAPI, Uvicorn, Typer, Pydantic, SQLAlchemy, and `cryptography`

## Dependency Manifest Direction

- keep GUI packaging dependencies isolated from license-server runtime dependencies
- do not rely on whichever packages happen to be installed in the repo `.venv`
- treat the server runtime as a reproducible bundled input rather than a host prerequisite

## Runtime Defaults

### GUI

- install root: `%LOCALAPPDATA%\Programs\Surrogate Model Training Suite\`
- config path: `%APPDATA%\Surrogate Model Training Suite\license_client.json`
- state path: `%LOCALAPPDATA%\Surrogate Model Training Suite\`
- output root: `%USERPROFILE%\Documents\Surrogate Model Training Suite\Runs\`

### Server

- install root: `C:\Program Files\MLP License Server\`
- runtime root: `%PROGRAMDATA%\MLP License Server\`
- example URL: `http://mlp-license-01:27850`

## Clean-Install Strategy

### Local Clean Staging

- perform the rehearsal outside the repo
- keep outputs under `MLP_modeling_v2_runtime\install-smoke\<timestamp>\`
- install the server bundle and GUI installer into clean locations
- keep the vendor issuance step internal, even if it still runs from a source checkout

### Happy Path To Mirror

The installed-artifact rehearsal must prove the same happy path already covered
by `scripts/smoke_test_license_server.ps1`:

1. initialize server identity
2. export `license_request.json`
3. issue an evaluation `license.json`
4. import the signed license
5. confirm `status`
6. launch the service on `27850`
7. configure the GUI with the server URL
8. confirm checkout
9. hold long enough to observe heartbeat behavior
10. close the GUI and confirm release

## Acceptance Criteria

### GUI Packaging

- installer launches the packaged app from the installed EXE, not the source tree
- packaged app writes session and licensing state to the OS-specific packaged locations
- uninstall removes binaries and shortcuts but keeps user state and outputs

### Server Packaging

- service installs without a separate host Python dependency
- bundled admin commands complete `init`, `export-request`, `import-license`, and `show-status`
- service starts cleanly and serves the client API on the chosen host and port

### Clean Install

- the same design-engineer and IT-admin flow works from clean install roots
- no packaged runtime step depends on repo-relative paths
- rehearsed outputs are collected outside the repo for inspection

## Out Of Scope For This Slice

- Linux package redesign
- vendor-tool packaging for customer delivery
- code signing
- MSI conversion or enterprise software distribution tooling beyond silent installer support

## Source References

- `gui_packaging_plan.md`
- `license_server_implementation_checklist.md`
- `../architecture/deployment_architecture.md`
- `../integration/windows_packaged_install_rehearsal.md`
