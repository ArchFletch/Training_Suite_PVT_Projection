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
- `linux/install_linux_service.sh`: creates the Linux service account, installs sample files, and
  renders the `systemd` unit.
- `linux/mlp-license-server.service.template`: `systemd` unit template for the server.
- `linux/mlp-license-server.env.sample`: sample environment file that carries the final service
  command string.
- `shared/config.windows.toml.sample`: sample Windows config layout.
- `shared/config.linux.toml.sample`: sample Linux config layout.

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
