# License Server Customer Install

These steps describe the intended customer IT install path for the on-prem
floating license server.

For Windows, the target delivery is a self-contained service bundle rather than
a source checkout plus manual Python staging.

## Current Runtime Defaults

- example LAN URL: `http://mlp-license-01:27850`
- customer admin model: CLI-first
- runtime data root: `%PROGRAMDATA%\MLP License Server\`

## Default Windows Layout

- install root: `C:\Program Files\MLP License Server\`
- bundled runtime: install-root bundled Python runtime or dedicated bundled virtual environment
- service wrapper files: `C:\Program Files\MLP License Server\mlp-license-server.exe` and
  `C:\Program Files\MLP License Server\mlp-license-server.xml`
- install script: `C:\Program Files\MLP License Server\install-service.ps1`
- admin wrapper command: `C:\Program Files\MLP License Server\mlp-license-server-admin.cmd`
- config file: `%PROGRAMDATA%\MLP License Server\config.toml`
- identity file: `%PROGRAMDATA%\MLP License Server\server_identity.json`
- imported license: `%PROGRAMDATA%\MLP License Server\active_license.json`
- database: `%PROGRAMDATA%\MLP License Server\license.db`
- logs: `%PROGRAMDATA%\MLP License Server\logs\`

## Windows Install

1. Extract or stage the Windows service bundle under `C:\Program Files\MLP License Server\`.
2. Confirm the bundle includes the packaged runtime, WinSW wrapper, config sample, and admin wrapper command.
3. Run the bundled install script to register and start the Windows service.

Example:

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Program Files\MLP License Server\install-service.ps1" `
  -InstallService `
  -StartService
```

4. Confirm the service is running with `Get-Service mlp-license-server` or `sc.exe query mlp-license-server`.
5. Open the chosen TCP port on the local firewall for LAN clients.
6. Initialize the server, export a request, import the signed license, and confirm status with the bundled admin wrapper.

Representative CLI flow:

```powershell
C:\Program Files\MLP License Server\mlp-license-server-admin.cmd init --runtime-root "%PROGRAMDATA%\MLP License Server"
C:\Program Files\MLP License Server\mlp-license-server-admin.cmd export-request "%PROGRAMDATA%\MLP License Server\license_request.json" --runtime-root "%PROGRAMDATA%\MLP License Server"
C:\Program Files\MLP License Server\mlp-license-server-admin.cmd import-license C:\temp\license.json --runtime-root "%PROGRAMDATA%\MLP License Server" --vendor-public-key C:\temp\vendor_public_key.pem
C:\Program Files\MLP License Server\mlp-license-server-admin.cmd show-status --runtime-root "%PROGRAMDATA%\MLP License Server"
```

7. Share the final LAN URL with engineers. The default documented example is:
   `http://mlp-license-01:27850`

## Linux Install

Linux remains a service-bundle workflow rooted outside the repo.

### Default Linux Layout

- install root: `/opt/mlp-license-server/`
- systemd unit: `/etc/systemd/system/mlp-license-server.service`
- config file: `/etc/mlp-license-server/config.toml`
- service env file: `/etc/mlp-license-server/service.env`
- identity file: `/var/lib/mlp-license-server/server_identity.json`
- imported license: `/var/lib/mlp-license-server/active_license.json`
- database: `/var/lib/mlp-license-server/license.db`
- logs: `/var/log/mlp-license-server/`

### Linux Install Flow

1. Stage the application files under `/opt/mlp-license-server/`.
2. Stage a Python runtime or dedicated virtual environment inside that install root.
3. Run the Linux helper as `root` and pass the final service command explicitly.

Example:

```bash
sudo ./packaging/license_server/linux/install_linux_service.sh \
  --install-root /opt/mlp-license-server \
  --service-command "/opt/mlp-license-server/.venv/bin/python -m uvicorn license_server.main:app --host 0.0.0.0 --port 27850" \
  --enable \
  --start
```

4. Confirm the service with `systemctl status mlp-license-server` and inspect logs with
   `journalctl -u mlp-license-server -n 100`.
5. Open the chosen TCP port in the host firewall for LAN clients.
6. Use the customer admin CLI to initialize the server, export a request, and import the signed license.

Representative CLI flow:

```bash
/opt/mlp-license-server/.venv/bin/python -m license_server.cli init --runtime-root /var/lib/mlp-license-server
/opt/mlp-license-server/.venv/bin/python -m license_server.cli export-request /var/lib/mlp-license-server/license_request.json --runtime-root /var/lib/mlp-license-server
/opt/mlp-license-server/.venv/bin/python -m license_server.cli import-license /tmp/license.json --runtime-root /var/lib/mlp-license-server --vendor-public-key /tmp/vendor_public_key.pem
/opt/mlp-license-server/.venv/bin/python -m license_server.cli show-status --runtime-root /var/lib/mlp-license-server
```

## Clean-Install Notes

- customer-facing runtime behavior must not depend on the source checkout
- Windows packaging must bundle WinSW directly
- Windows packaging must not require customer IT to install Python separately
- release validation should rehearse the same happy path from packaged artifacts before customer rollout
