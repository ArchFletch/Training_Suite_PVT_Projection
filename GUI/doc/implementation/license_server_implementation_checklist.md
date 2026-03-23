# License Server Implementation Checklist

## Goal

Track the remaining work needed to deliver the Windows packaging and
clean-install slice without losing the context of what is already built.

## Completed Foundation

### Phase 0: Repo Preparation

- [x] `license_server`, `license_vendor`, and `xfmr_v2/licensing` package roots exist.
- [x] dedicated test layout exists for server, vendor, and GUI-license coverage.
- [x] local developer entrypoints exist for the server and vendor tool.

### Phase 1: Contracts And Schemas

- [x] server identity, license request, signed license, and client API schemas exist.
- [x] denial reason codes such as `all_seats_in_use` are standardized in tests and models.
- [x] JSON fixtures exist for valid and invalid contract payloads.

### Phase 2: Persistence And Paths

- [x] SQLite models, session handling, and repository helpers exist.
- [x] Windows and Linux path resolution exists for service runtime state.
- [x] desktop source-vs-packaged path policy exists in `xfmr_v2.app_paths`.

### Phase 3: Core Services

- [x] server identity generation and request export exist.
- [x] host fingerprinting and license verification exist.
- [x] license import, replacement rules, lease management, and audit logging exist.

### Phase 4: API Surface

- [x] FastAPI wiring exists.
- [x] `POST /api/v1/checkout` exists.
- [x] `POST /api/v1/heartbeat` exists.
- [x] `POST /api/v1/release` exists.
- [x] `GET /api/v1/status` exists.

### Phase 5: Customer CLI

- [x] `init` exists.
- [x] `export-request` exists.
- [x] `import-license` exists.
- [x] `show-status` exists.
- [x] `show-audit` exists.

### Phase 6: Vendor Tool

- [x] evaluation-license issuance exists.
- [x] paid-license issuance exists.
- [x] issuance-log persistence exists.
- [x] vendor-created licenses import successfully in tests.

### Phase 7: Desktop App Integration

- [x] desktop HTTP client exists.
- [x] desktop lease-state and heartbeat handling exist.
- [x] GUI server URL settings and connection-test behavior exist.
- [x] startup checkout gating and shutdown release behavior exist.

## Remaining Slice: Windows Packaging And Operations

### Phase 8: Packaging Deliverables

- [ ] define separate dependency manifests or equivalent reproducible inputs for GUI packaging and server runtime staging
- [ ] finalize the Windows GUI build and installer flow around the tracked packaging scripts
- [ ] add a Windows server bundle build path that stages the server code, runtime dependencies, bundled WinSW, and config assets
- [ ] expose bundled admin wrapper commands for customer IT
- [ ] rewrite customer and operations docs around the chosen Windows delivery model

Exit criteria:

- a release engineer can produce the GUI installer and server bundle without relying on ad hoc local state
- customer IT docs no longer assume a source checkout or external WinSW download

### Phase 9: Clean-Install Acceptance

- [ ] install the Windows server bundle into clean non-repo locations
- [ ] install the Windows GUI through the packaged installer into a clean user context
- [ ] run the packaged happy path outside the source folders
- [ ] confirm checkout, heartbeat, and release using installed artifacts
- [ ] collect logs and outputs under an external install-smoke workspace
- [ ] rerun the same flow on pristine Windows client/server VMs

Exit criteria:

- the same happy path already covered by the source smoke rehearsal passes from packaged artifacts
- packaged runtime writes occur only in the documented OS-specific locations

## Acceptance Checklist

- [ ] Fresh Windows server install can export a request and import a signed eval license.
- [ ] Fresh Windows server install can serve the client API on the configured LAN URL.
- [ ] Windows desktop client can read a packaged `license_client.json` and check out a seat.
- [ ] Windows desktop client can heartbeat and release from the installed EXE.
- [ ] GUI uninstall removes binaries and shortcuts without deleting user data.
- [ ] Packaged runtime behavior no longer depends on source folders.

## Current Baseline Validation

These behaviors already have source-based coverage and remain the fast regression baseline:

- schema and service tests under `tests/license_server/`
- vendor tool tests under `tests/license_vendor/`
- GUI licensing tests under `tests/gui_license/`
- source-based smoke rehearsal in `scripts/smoke_test_license_server.ps1`

## Next Documentation References

- `windows_packaging_and_clean_install.md`
- `license_server_repository_scaffold.md`
- `../integration/windows_packaged_install_rehearsal.md`
