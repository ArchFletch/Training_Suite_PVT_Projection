# License Server Customer Install

These steps describe the intended customer IT install path for the on-prem
floating license server.

On both Windows and Linux the delivery is a service bundle rather than a source
checkout. The two are not equivalent beyond that: the Windows bundle embeds its
own Python runtime, while the Linux bundle ships application code only and the
admin builds a virtual environment on the host. A Linux install therefore needs
the host to reach both an apt mirror (for `python3-venv`) and PyPI (for the
runtime requirements); an air-gapped Linux host needs those wheels staged
separately.

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

Linux is a service-bundle workflow rooted outside the repo. Ubuntu 22.04/24.04
and other `systemd` distributions are supported. Python 3.12 is the baseline;
`python3-venv` must be installed before staging.

The delivered artifact is `mlp-license-server-linux.tar.gz`, built by
`packaging/license_server/linux/build_bundle.sh`. It extracts to a single
`mlp-license-server/` directory holding:

- `license_server/`: the application package, including `requirements.txt`
- `linux/`: `install_linux_service.sh` plus the unit and environment templates
- `shared/config.linux.toml.sample`: the config rendered by the installer
- `INSTALL.md`: the condensed version of this flow for the customer admin

Nothing else ships. The vendor signing tooling in `license_vendor/` is never
part of a customer bundle, and no customer host needs a source checkout.

### Default Linux Layout

- install root: `/opt/mlp-license-server/`
- application package: `/opt/mlp-license-server/license_server/`
- virtual environment: `/opt/mlp-license-server/.venv/`
- systemd unit: `/etc/systemd/system/mlp-license-server.service`
- config file: `/etc/mlp-license-server/config.toml` (lease timings only)
- service env file: `/etc/mlp-license-server/service.env` (the start command)
- vendor public key: `/etc/mlp-license-server/vendor_public_key.pem`
- identity file: `/var/lib/mlp-license-server/server_identity.json`
- imported license: `/var/lib/mlp-license-server/active_license.json`
- database: `/var/lib/mlp-license-server/license.db`
- logs: `journalctl -u mlp-license-server` (the service logs to the journal;
  `/var/log/mlp-license-server/` is reserved and stays empty today)

The listening port comes from the service command in `service.env`, not from
`config.toml`. Changing a port in `config.toml` has no effect.

### Linux Install Flow

Extract the delivered bundle anywhere the admin can read it, for example the
admin home directory. Every command below runs from that directory.

```bash
tar -xzf mlp-license-server-linux.tar.gz
cd mlp-license-server
```

Steps 1 and 2 must both complete before the service is started: the installer
creates the service account and unit but does **not** stage code or build a
virtual environment, so starting earlier gives a unit with nothing to run.

1. Run the installer as `root`, passing the final service command explicitly.
   Use the copy inside the extracted bundle; it reads two sibling templates and
   `../shared/config.linux.toml.sample`, so it cannot be moved on its own.

```bash
sudo bash ./linux/install_linux_service.sh \
  --install-root /opt/mlp-license-server \
  --service-command "/opt/mlp-license-server/.venv/bin/python -m uvicorn license_server.main:app --host 0.0.0.0 --port 27850"
```

This creates the `mlp-license-server` account, the config/data/log directories,
`config.toml`, `service.env`, and the systemd unit. Deliberately no `--enable`
or `--start` yet.

2. Stage the application and build its virtual environment.

```bash
sudo apt install -y python3-venv
sudo cp -rT ./license_server /opt/mlp-license-server/license_server
sudo python3 -m venv /opt/mlp-license-server/.venv
sudo /opt/mlp-license-server/.venv/bin/pip install \
  -r /opt/mlp-license-server/license_server/requirements.txt
```

Confirm the service account can import the package from the unit's working
directory before involving systemd:

```bash
sudo -u mlp-license-server env -C /opt/mlp-license-server \
  /opt/mlp-license-server/.venv/bin/python -c "import license_server; print('ok')"
```

3. Create the server identity and export the request. Run every admin command
   as the service account: a root-created `license.db` leaves the service unable
   to write it, which presents as a successful activation followed by failures
   on every seat request.

```bash
svc() { sudo -u mlp-license-server env -C /opt/mlp-license-server \
        /opt/mlp-license-server/.venv/bin/python -m license_server.cli "$@"; }

svc init --runtime-root /var/lib/mlp-license-server
svc export-request /var/lib/mlp-license-server/license_request.json \
    --runtime-root /var/lib/mlp-license-server
```

4. Send `license_request.json` to the vendor. It contains only a server ID, a
   host fingerprint, the hostname, and the OS family.

5. Install the vendor public key and import the signed license returned to you.
   The key is delivered once per customer; renewals reuse it.

```bash
sudo install -m 0644 vendor_public_key.pem /etc/mlp-license-server/vendor_public_key.pem
sudo install -o mlp-license-server -g mlp-license-server -m 0640 \
  license.json /var/lib/mlp-license-server/license.json

svc import-license /var/lib/mlp-license-server/license.json \
    --runtime-root /var/lib/mlp-license-server \
    --vendor-public-key /etc/mlp-license-server/vendor_public_key.pem
```

Import reports `license_state`. If it returns `expired` or `not_started` it also
prints a warning: the license is active on the server but no seat will be granted
until the term covers the current date.

6. Enable and start the service, then verify by the listening socket rather than
   the unit state.

```bash
sudo systemctl enable --now mlp-license-server
ss -ltnp | grep 27850
curl -s http://localhost:27850/api/v1/status
```

Expect `"ok": true` with the company name and seat count. `svc show-status`
gives the fuller admin view, including `license_state`, the term, enabled
features, and every active lease.

7. Open the chosen TCP port in the host firewall for LAN clients.

```bash
sudo ufw allow 27850/tcp          # Ubuntu / Debian
sudo firewall-cmd --permanent --add-port=27850/tcp && sudo firewall-cmd --reload
```

8. Share the LAN URL with engineers, for example `http://mlp-license-01:27850`.

## Clean-Install Notes

- customer-facing runtime behavior must not depend on the source checkout
- customer bundles must never carry `license_vendor/`, the vendor signing tooling
- Windows packaging must bundle WinSW directly
- Windows packaging must not require customer IT to install Python separately
- release validation should rehearse the same happy path from packaged artifacts before customer rollout
