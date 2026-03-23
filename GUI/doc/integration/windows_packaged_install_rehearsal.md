# Windows Packaged Install Rehearsal

## Purpose

Validate the Windows GUI installer and Windows license-server service bundle
from clean installs outside the source checkout.

## Scope

This rehearsal is the packaged-artifact counterpart to the current
source-based smoke flow in `scripts/smoke_test_license_server.ps1`.

It must prove the same happy path:

1. initialize server identity
2. export `license_request.json`
3. issue an evaluation `license.json`
4. import the signed license
5. confirm `status`
6. start the server on `27850`
7. configure the GUI with the server URL
8. confirm checkout
9. observe heartbeat behavior
10. close the GUI and confirm release

## Required Inputs

- Windows GUI installer for `Surrogate Model Traning Suite`
- Windows license-server service bundle
- internal vendor-tool runtime for issuing the evaluation license
- a clean external workspace under `MLP_modeling_v2_runtime\install-smoke\`

## Local Clean Staging

### Workspace

- create a timestamped folder such as:
  `C:\Users\<user>\Dropbox\projects\MLP_modeling_v2_runtime\install-smoke\20260316-153000\`
- keep all generated request, license, log, and validation files under that workspace
- do not write rehearsal outputs back into the repo

### Expected Install Roots

- GUI install root: `%LOCALAPPDATA%\Programs\Surrogate Model Traning Suite\`
- server install root: `C:\Program Files\MLP License Server\`
- server runtime root: `%PROGRAMDATA%\MLP License Server\`

## Happy-Path Rehearsal

### 1. Install The Server Bundle

- extract or stage the Windows server bundle into `C:\Program Files\MLP License Server\`
- confirm the bundle includes the bundled runtime, WinSW wrapper, config sample, and admin wrappers
- install and start the Windows service

### 2. Initialize The Server

- run the bundled admin command to initialize the server runtime
- export `license_request.json`
- confirm the request contains the expected `server_id`, `host_fingerprint`, `hostname`, and `os_family`

### 3. Issue And Import The License

- use the internal vendor tool to issue an evaluation `license.json` from the exported request
- import the signed license with the bundled admin command
- run `show-status` and confirm the license is loaded with `seats_in_use = 0`

### 4. Confirm The Running Service

- confirm the service is reachable at `http://mlp-license-01:27850` or the final configured hostname
- call `GET /api/v1/status`
- confirm the returned seat count and current usage match the imported license

### 5. Install The GUI

- run the GUI installer in a clean user context
- repeat with silent install as part of release validation
- confirm the installed EXE launches from `%LOCALAPPDATA%\Programs\Surrogate Model Traning Suite\`

### 6. Configure The GUI

- pre-seed or update `%APPDATA%\Surrogate Model Traning Suite\license_client.json` with:

```json
{
  "server_url": "http://mlp-license-01:27850"
}
```

- launch the packaged GUI
- confirm the packaged GUI reads the per-user config from the packaged path, not the repo

### 7. Confirm Checkout

- verify that GUI startup reaches the checked-out state when the server is healthy
- confirm `seats_in_use` increases to `1`
- if needed, verify through `show-status`, audit logs, or direct API calls

### 8. Confirm Heartbeat

- keep the GUI open long enough to verify the seat remains active
- confirm the server shows an active lease and heartbeat renewal behavior

### 9. Confirm Release

- close the GUI cleanly
- confirm `POST /api/v1/release` behavior through audit logs, API status, or admin status output
- confirm `seats_in_use` returns to `0`

## Pristine-VM Signoff

After local clean staging passes, repeat the same flow on:

- a fresh Windows server VM for the service bundle
- a fresh Windows client VM for the packaged GUI

The pristine-VM pass is the final release gate for the Windows packaging slice.

## Pass Criteria

- both artifacts install without relying on the repo
- the server bundle works without a separately managed host Python install
- the GUI writes state only to packaged-user locations
- the same happy path as the source smoke flow passes from installed artifacts

## Artifacts To Collect

- exported `license_request.json`
- issued `license.json`
- server status and audit outputs
- server logs
- GUI install logs for silent install validation
- a short run summary under the external install-smoke workspace

## Source References

- `license_server_mvp_rehearsal.md`
- `../implementation/windows_packaging_and_clean_install.md`
- `../../scripts/smoke_test_license_server.ps1`
