# GUI Packaging

This folder contains the proposed desktop packaging flow for the PySide6 GUI.

## Build Strategy

- build a standalone GUI bundle with `pyside6-deploy` and Nuitka
- turn the Windows standalone directory into an Inno Setup installer
- turn the Linux standalone directory into a tar.gz bundle with a launcher script

## Scripts

- `windows/build_gui.ps1`: render a local `pysidedeploy.spec` and run the standalone GUI build
- `windows/build_installer.ps1`: wrap a built standalone directory in an Inno Setup installer
- `linux/build_gui.sh`: render a local `pysidedeploy.spec` and run the standalone GUI build
- `linux/build_bundle.sh`: stage a tar.gz Linux bundle from a built standalone directory
- `render_pyside6_spec.py`: shared helper that renders the spec template with absolute paths

## Supporting Files

- `gui/pysidedeploy.spec.in`: template consumed by the build scripts
- `windows/SurrogateModelTrainingSuite.iss`: Inno Setup installer definition
- `linux/mlp-training-studio.desktop.in`: desktop-entry template for Linux installs

## Prerequisites

- Python environment with the GUI dependencies already installed
- `pyside6-deploy` available on `PATH`, or `PySide6.scripts.deploy` importable from the chosen Python
- Windows installer builds additionally need Inno Setup 6

## Runtime State Policy

Packaged installs never write runtime state into the install directory or tracked repo files.
The shared path policy lives in `xfmr_v2/app_paths.py`, and the license client config
helper lives in `xfmr_v2/licensing/client_config.py`.
