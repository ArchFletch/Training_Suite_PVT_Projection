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
- `assemble_delivery.sh`: collect the built customer artifacts into one delivery tree
  grouped by recipient
- `render_pyside6_spec.py`: shared helper that renders the spec template with absolute paths

## Supporting Files

- `gui/pysidedeploy.spec.in`: template consumed by the build scripts
- `windows/SurrogateModelTrainingSuite.iss`: Inno Setup installer definition
- `linux/mlp-training-studio.desktop.in`: desktop-entry template for Linux installs

## Prerequisites

- Python environment with the GUI dependencies already installed
- `pyside6-deploy` available on `PATH`, or `PySide6.scripts.deploy` importable from the chosen Python
- Windows installer builds additionally need Inno Setup 6
- `assemble_delivery.sh` needs only `bash`, `install`, `find` and `sha256sum` (or `shasum`)

## Customer Delivery

The GUI and license server builds each write into their own `artifacts/packaging/`
subdirectory. `assemble_delivery.sh` collects whatever is there into
`artifacts/delivery/`, grouped by who receives it rather than by platform:

- `engineers-gui/`: the Windows installer and the Linux bundle, for the design engineers
- `server-admin/`: both license server bundles, the Windows zip and the Linux tarball, for
  customer IT, who installs the one matching the server OS
- `checksums.txt`: SHA256 of every collected file, one line per file, checkable with
  `sha256sum -c checksums.txt` from the delivery root. It is written only when at least one
  artifact was collected: a zero-byte manifest is not an empty tree, it is a file that makes
  `sha256sum -c` fail with "no properly formatted SHA256 checksum lines found"
- `README.md`: generated; names each folder's recipient, marks each artifact PRESENT,
  CARRIED OVER or MISSING, and describes the license handshake

A recipient directory is created only when something is actually put in it, so an
assembly that collected nothing leaves no empty folders and no checksum file to mislead
anyone.

```bash
packaging/assemble_delivery.sh [--output-dir PATH] [--clean]
```

No single machine builds all four artifacts: the Windows installer needs a Windows host
with Inno Setup, the Windows license server bundle needs Windows, PowerShell, WinSW and
network access to PyPI (its build runs `ensurepip` and `pip install -r
runtime_requirements.txt` into the runtime it bundles), and both GUI builds need a Python
with PySide6 and Nuitka on the OS they target. The Linux license server bundle is the
exception: it only stages files and tars them, so it builds on any host with `bash` and
`tar`, Linux or not, with no Python or Qt toolchain and no network access.

One `--output-dir` is therefore normally filled in by runs on several machines. A missing
artifact is reported with the exact command and toolchain that produce it and does not
fail the run. An artifact already in the tree that this machine cannot rebuild is kept,
re-checksummed from what is on disk and reported as CARRIED OVER: the script removes a
destination only at the moment it has a replacement in hand, so a run on a partly equipped
host cannot destroy what another host contributed. `--clean` discards the whole tree, on
purpose.

Files sitting in `engineers-gui/` or `server-admin/` that a run did not place, such as an
installer left from a previous release, are not deleted, because deleting files the script
did not write is the same mistake as deleting an artifact it cannot rebuild. They are
reported instead: in the console output, in a comment block at the top of `checksums.txt`
(which `sha256sum -c` skips, so verification still works), and in a "Files this tree does
not account for" section of the generated `README.md`. The run then exits 3, so nothing
automated can treat the tree as shippable while an unexplained file is in it.

Exit status: `0` assembled and fully accounted for, `1` hard failure, `2` usage error,
`3` assembled but holding files the run did not place.

The script refuses to place `*signing_key*`, `*.pem`, `*.jsonl` or `license.json` in the
delivery tree and fails if it finds a file with one of those names there, the same guard
`license_server/linux/build_bundle.sh` applies to its staging. Being the one step that
assembles a complete shipment, it is also the one place that check is worth having.

That guard reads file names. It cannot see inside
`SurrogateModelTrainingSuite-Windows.exe`, either tarball or `MLP License Server.zip`, and
the script's closing banner states exactly that rather than declaring the tree clean. The
Windows server zip is the one to open by hand before a release goes out:
`license_server/windows/build_bundle.ps1` copies the build machine's entire
`license_server/` directory into the bundle and has no leak guard of its own, so a
`vendor_public_key.pem`, `license.json` or `issuance_log.jsonl` left in that directory
during development would ship inside the archive. The Linux bundle needs no such review,
`license_server/linux/build_bundle.sh` checks its staging before writing the tarball.

`license.json` and `vendor_public_key.pem` are never assembled here. They are issued per
server, only after the customer admin installs the `server-admin/` bundle for their OS and returns
`license_request.json`, and are sent out of band. Every delivery is therefore two
shipments. See `license_server/README.md` and
`doc/operations/license_server_customer_install.md`.

## Runtime State Policy

Packaged installs never write runtime state into the install directory or tracked repo files.
The shared path policy lives in `xfmr_v2/app_paths.py`, and the license client config
helper lives in `xfmr_v2/licensing/client_config.py`.
