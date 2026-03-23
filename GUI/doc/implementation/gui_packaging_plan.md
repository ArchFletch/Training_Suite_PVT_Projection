# GUI Packaging Plan

## Goal

Ship the existing PySide6 desktop GUI as a distributable Windows app without
changing training behavior, while keeping all writable state out of the install
directory and tracked repo artifacts.

This doc stays GUI-specific. The cross-cutting Windows delivery slice is
tracked in `windows_packaging_and_clean_install.md`.

## Chosen Packaging Flow

### Windows

1. Build a standalone GUI bundle with `pyside6-deploy` + Nuitka from `launch_gui.py`.
2. Verify the standalone directory launches and can open datasets, run scans, and save outputs.
3. Wrap that standalone directory in an Inno Setup installer for per-user install.
4. Install into `%LOCALAPPDATA%\Programs\Surrogate Model Traning Suite\`.
5. Validate normal install, silent install, launch, and uninstall behavior.

Tracked assets:

- `packaging/windows/build_gui.ps1`
- `packaging/windows/build_installer.ps1`
- `packaging/windows/SurrogateModelTrainingSuite.iss`

Why this fits:

- keeps the GUI packaging aligned with the existing Qt-for-Python direction in `doc/tech_stack/tech_stack_gui.md`
- gives Windows users a normal installer and shortcuts
- keeps runtime state per-user instead of in the install tree

### Linux

1. Build a standalone GUI bundle with `pyside6-deploy` + Nuitka from `launch_gui.py`.
2. Stage a relocatable tar.gz bundle that contains:
   - the standalone app files under `app/`
   - a launcher under `bin/mlp-training-studio`
   - a rendered `.desktop` entry under `share/applications/`
3. Customer IT extracts the bundle to `/opt/mlp-training-studio` or a user-owned directory.

Tracked assets:

- `packaging/linux/build_gui.sh`
- `packaging/linux/build_bundle.sh`
- `packaging/linux/mlp-training-studio.desktop.in`

## Shared Build Inputs

- GUI entrypoint: `launch_gui.py`
- runtime path policy: `xfmr_v2/app_paths.py`
- licensing config seam: `xfmr_v2/licensing/client_config.py`
- `pyside6-deploy` template: `packaging/gui/pysidedeploy.spec.in`

## Writable-State Policy

The runtime path policy is centralized in `xfmr_v2/app_paths.py`.

### Packaged Windows App

| Concern | Location |
| --- | --- |
| installed binaries | `%LOCALAPPDATA%\Programs\Surrogate Model Traning Suite\` |
| license server settings | `%APPDATA%\Surrogate Model Traning Suite\license_client.json` |
| last restored GUI session | `%LOCALAPPDATA%\Surrogate Model Traning Suite\last_session.json` |
| reserved file-log directory | `%LOCALAPPDATA%\Surrogate Model Traning Suite\logs\` |
| default run outputs | `%USERPROFILE%\Documents\Surrogate Model Traning Suite\Runs\` |
| default auto-managed cache | `%USERPROFILE%\Documents\Surrogate Model Traning Suite\Runs\cache\<run_name>.npz` |

### Packaged Linux App

| Concern | Location |
| --- | --- |
| extracted app bundle | `/opt/mlp-training-studio/` or a user-owned install root |
| license server settings | `~/.config/mlp-training-studio/license_client.json` |
| last restored GUI session | `~/.local/state/mlp-training-studio/last_session.json` |
| reserved file-log directory | `~/.local/state/mlp-training-studio/logs/` |
| default run outputs | `~/Documents/Surrogate Model Traning Suite/runs/` |
| default auto-managed cache | `~/Documents/Surrogate Model Traning Suite/runs/cache/<run_name>.npz` |

### Source Checkout

The repo keeps its existing developer-friendly behavior:

| Concern | Location |
| --- | --- |
| last GUI session | `artifacts/gui/last_session.json` |
| desktop license client config | `artifacts/gui/license_client.json` |
| default outputs | `artifacts/output/` |
| default cache | `artifacts/output/cache/<run_name>.npz` |

## Licensing Coordination

The packaged desktop app and the GUI licensing layer share one per-user file:
`license_client.json`.

Expected shape:

```json
{
  "server_url": "http://mlp-license-01:27850"
}
```

The packaged GUI should keep using the same path helper in
`xfmr_v2/licensing/client_config.py` instead of inventing new OS-specific locations.

## Installed-Artifact Validation

Windows GUI packaging is not complete until all of the following pass:

- standalone build launches from its packaged directory
- installer launches the installed EXE instead of the source checkout
- packaged runtime writes go to `%APPDATA%`, `%LOCALAPPDATA%`, and `Documents`
- a pre-seeded `license_client.json` is read from the packaged config path
- uninstall removes binaries and shortcuts but keeps user configs and outputs

The cross-cutting clean-install flow is documented in:

- `windows_packaging_and_clean_install.md`
- `../integration/windows_packaged_install_rehearsal.md`

## Non-Goals For This GUI-Specific Doc

- Windows server service-bundle implementation details
- vendor-tool packaging
- Linux distro-specific package-manager integration
- code signing

## Related Docs

- `windows_packaging_and_clean_install.md`
- `../customer/gui_install_run.md`
- `../architecture/deployment_architecture.md`
