# License Server Packaging Assets

These assets cover the MVP operational packaging layer for the on-prem floating license server.
They are intentionally limited to install-time docs, service templates, sample configuration, and
bootstrap scripts.

## What Is Here

- `windows/build_bundle.ps1`: stages the Windows service bundle with a dedicated venv, WinSW,
  wrapper commands, and service-install assets.
- `windows/install_windows_service.ps1`: prepares the Windows service layout and writes a WinSW
  service definition.
- `windows/mlp-license-server-service.xml.template`: WinSW XML template used by the PowerShell
  installer.
- `linux/build_bundle.sh`: stages the Linux customer tarball with the application package, the
  service-install assets, and the sample config.
- `linux/install_linux_service.sh`: creates the Linux service account, installs sample files, and
  renders the `systemd` unit.
- `linux/mlp-license-server.service.template`: `systemd` unit template for the server.
- `linux/mlp-license-server.env.sample`: sample environment file that carries the final service
  command string.
- `linux/INSTALL.md.template`: customer-admin instructions rendered into the staged Linux bundle.
- `shared/config.windows.toml.sample`: sample Windows config layout.
- `shared/config.linux.toml.sample`: sample Linux config layout.

## Customer Bundles

Both bundles are the complete customer delivery, so neither carries `license_vendor/`, tests, or
internal reports. Customers never receive a source checkout.

- Windows: `windows/build_bundle.ps1 -WinSWExePath <winsw.exe>` stages
  `artifacts\packaging\license_server\windows\MLP License Server\` and zips it. `-SkipArchive`
  stages without the zip.
- Linux: `linux/build_bundle.sh` stages
  `artifacts/packaging/license_server/linux/mlp-license-server/` and writes
  `mlp-license-server-linux.tar.gz` beside it. `--skip-archive` stages without the tarball.

The Linux bundle keeps `linux/` and `shared/` side by side because
`install_linux_service.sh` resolves its templates relative to its own directory, so the customer
runs it straight out of the extracted tarball.

## Current Runtime Commands

These files avoid hard-coding the install-time service command. The service bootstrap scripts
accept a full command line, but the currently implemented defaults are:

- service HTTP app: `python -m uvicorn license_server.main:app --host 0.0.0.0 --port 27850`
- customer admin CLI: `python -m license_server.cli`

## Default OS Layout

- Windows install root: `C:\Program Files\MLP License Server\`
- Windows config/data root: `%PROGRAMDATA%\MLP License Server\`
- Linux install root: `/opt/mlp-license-server/`
- Linux config root: `/etc/mlp-license-server/`
- Linux data root: `/var/lib/mlp-license-server/`

See `doc/operations/license_server_customer_install.md` for the customer-admin install path.
