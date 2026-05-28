# Deployment

One-page recipes for running the on-prem floating license server consumed by
the Surrogate Model Training Suite GUI. The GUI itself is a desktop client and
does not need to be deployed as a service.

For longer-form material, see:

- `doc/operations/license_server_customer_install.md` — Windows / Linux service install
- `packaging/license_server/README.md` — packaging assets (WinSW, systemd, sample configs)
- `doc/architecture/license_server_architecture.md` — module boundaries

## Server endpoints at a glance

- Service HTTP app: `python -m uvicorn license_server.main:app --host 0.0.0.0 --port 27850`
- Customer admin CLI: `python -m license_server.cli`
- Runtime data root: `MLP_LICENSE_SERVER_RUNTIME_DIR` env var
  (Linux default `/var/lib/mlp-license-server/`, Windows default `%PROGRAMDATA%\MLP License Server\`)
- GUI client reaches it via the **License Server URL** field in the GUI's License card
  (e.g. `http://mlp-license-01:27850`).

## Run from source (Linux / macOS)

```bash
cd GUI
pip install -r license_server/requirements.txt
export MLP_LICENSE_SERVER_RUNTIME_DIR=$PWD/.runtime
python -m uvicorn license_server.main:app --host 0.0.0.0 --port 27850
```

Customer-admin actions (import license, list seats, etc.) go through the
matching CLI:

```bash
python -m license_server.cli --help
```

## Run with Docker

```bash
cd GUI
docker build -t mlp/license-server:0.1 license_server/
docker run -d --name mlp-license \
  -p 27850:27850 \
  -v mlp-license-data:/data \
  mlp/license-server:0.1
```

The image bakes in `MLP_LICENSE_SERVER_RUNTIME_DIR=/data` and exposes 27850,
so the GUI client points at `http://<host>:27850`. Persistent state (SQLite
DB, server identity, active imported license) lives in the `mlp-license-data`
volume.

Customer-admin commands run inside the container:

```bash
docker exec -it mlp-license python -m license_server.cli --help
```

## Run with Docker Compose

`license_server/docker-compose.example.yml` is a ready-to-edit compose file.
Copy it to `docker-compose.yml` next to a customer-issued `active_license.json`
(or drop that volume mount and import via the CLI afterwards) and:

```bash
docker compose up -d
docker compose logs -f license     # follow startup logs
docker compose exec license python -m license_server.cli --help
```

## Windows / Linux service install (no Docker)

The packaging assets under `packaging/license_server/` register the server as
a native Windows service (via WinSW) or `systemd` unit. See:

- `packaging/license_server/windows/install_windows_service.ps1`
- `packaging/license_server/linux/install_linux_service.sh`

Both bootstrap scripts accept a service-command string; the current default is
`python -m uvicorn license_server.main:app --host 0.0.0.0 --port 27850`.

## Verifying a deployment

After the server is up, the GUI should report **Server Status: OK** when the
license card's `Server URL` is set and `Test Connection` is clicked. The same
status is reachable via:

```bash
curl http://<host>:27850/status
```
