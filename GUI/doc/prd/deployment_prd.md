# PRD: Windows Deployment

## Product

Windows delivery workflow for `Surrogate Model Training Suite` and its on-prem
license server.

## Goal

Deliver an EDA-style Windows install experience where:

- design engineers install the desktop app like a normal per-user tool
- customer IT installs the license server as a self-contained Windows service bundle
- release validation proves the same happy path from clean installs outside the source checkout

## Users

- Design engineer: installs and launches the packaged desktop app
- Customer IT admin: installs and operates the Windows license server
- Release engineer: builds artifacts and validates clean-install readiness
- Support engineer: reproduces installs and diagnoses packaging issues

## In Scope

- per-user Windows GUI installer
- Windows server service bundle
- bundled server runtime and bundled WinSW wrapper
- silent GUI install and uninstall
- explicit install roots and writable-state roots
- clean-install validation outside the repo

## Out Of Scope

- Linux deployment redesign in this slice
- vendor-tool packaging for customer delivery
- code signing
- enterprise SCCM or Intune packaging beyond silent installer support
- MSI conversion

## Primary Workflows

### Design Engineer Workflow

1. Run the Windows GUI installer.
2. Launch `Surrogate Model Training Suite` from the Start menu or desktop shortcut.
3. Use a saved or pre-seeded server URL.
4. Start work only after seat checkout succeeds.

### Customer IT Workflow

1. Stage the Windows license-server service bundle.
2. Install and start the Windows service.
3. Initialize the server and export a license request.
4. Import the signed license.
5. Share `http://mlp-license-01:27850` or the final configured URL with engineers.

### Release Validation Workflow

1. Build the Windows GUI installer and Windows server bundle.
2. Install both into clean non-repo locations.
3. Run the packaged happy path through checkout, heartbeat, and release.
4. Capture logs and outputs under an external install-smoke workspace.

## Product Requirements

### Design Engineer Experience

- GUI install must not require admin rights.
- GUI install must behave like a normal Windows desktop tool with shortcuts and uninstall support.
- The packaged app must not require a source checkout to run.
- The packaged app must keep writable state outside the install directory.

### Customer IT Experience

- Server install must be possible without separately installing Python.
- The Windows server bundle must include the service wrapper, runtime, and install guidance.
- Customer IT must be able to run init, export, import, and status commands after install.
- The default example server URL must use port `27850`.

### Clean-Install Readiness

- Clean-install validation must run outside the repo.
- Clean-install validation must mirror the existing source smoke happy path.
- Release notes and customer-facing docs must agree on paths, names, and ports.

### Uninstall Expectations

- GUI uninstall removes binaries and shortcuts only.
- GUI uninstall does not delete configs, caches, logs, or run outputs.
- Server removal guidance must preserve or explicitly call out runtime-data handling in `%PROGRAMDATA%`.

## Non-Functional Requirements

- stable path policy across packaged runs
- consistent product naming across installer, docs, and GUI
- reproducible build inputs
- customer-facing instructions that do not depend on source-tree knowledge

## Success Criteria

- A design engineer can install and launch `Surrogate Model Training Suite` from a normal Windows installer.
- Customer IT can install the Windows service bundle and reach a running server at the configured LAN URL.
- A clean-install rehearsal proves checkout, heartbeat, and release without relying on source folders.

## Key Risks

- inconsistent path guidance creates support burden
- packaging docs drift from actual runtime behavior
- host Python dependencies accidentally leak into the server install path
- silent-install behavior is not tested before customer rollout

## Source References

- `gui_prd.md`
- `license_server_prd.md`
- `../architecture/deployment_architecture.md`
- `../implementation/windows_packaging_and_clean_install.md`
