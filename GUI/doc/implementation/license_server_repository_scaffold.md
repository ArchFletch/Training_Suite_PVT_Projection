# License Server Repository Snapshot

## Purpose

Describe the current repository shape for the license-server lane and the
packaging boundaries that matter for the Windows delivery slice.

## Current Layout

```text
license_server/
  api/
  cli/
  crypto/
  db/
  schemas/
  service/
  services/
  main.py
  README.md

license_vendor/
  cli.py
  issuance_log.py
  signer.py
  README.md

xfmr_v2/
  licensing/
    client.py
    client_config.py
    controller.py
    models.py
    README.md
  app_paths.py

packaging/
  license_server/
    windows/
    linux/
    shared/

tests/
  license_server/
  license_vendor/
  gui_license/
```

## Responsibility Map

- `license_server/api`: FastAPI app wiring and client routes
- `license_server/cli`: customer-admin CLI commands for init, export, import, status, and audit
- `license_server/crypto`: fingerprinting and Ed25519 verification helpers
- `license_server/db`: SQLAlchemy models, session factory, and repositories
- `license_server/schemas`: request, license, and client API contracts
- `license_server/service`: runtime bootstrap, config, path resolution, and shared utilities
- `license_server/services`: identity, license import, lease, and audit behavior
- `license_vendor`: internal-only license issuance tooling
- `xfmr_v2/licensing`: desktop-side server URL persistence, HTTP client, and lease controller
- `packaging/license_server`: service templates, config samples, and install helpers
- `tests/*`: source-based regression coverage for the server, vendor tool, and GUI licensing layer

## Current Packaging Boundaries

### Server Code Boundary

- `license_server` must remain importable without PySide6 or training dependencies
- runtime path handling belongs in `license_server/service`
- packaged Windows service assumptions must stay out of the core business-rule modules

### Vendor Tool Boundary

- `license_vendor` remains internal-only
- vendor signing logic must not ship inside the customer server bundle
- source-based vendor issuance is acceptable for this packaging slice

### Desktop Boundary

- GUI licensing belongs in `xfmr_v2/licensing`
- packaged GUI state location is controlled by `xfmr_v2/app_paths.py`
- the desktop app must not hard-code repo-relative paths when packaged

### Packaging Asset Boundary

- installer scripts, WinSW templates, and config samples stay under `packaging/`
- customer install roots and runtime data roots are documented separately from package code
- packaged validation outputs belong in an external runtime workspace, not in the repo

## Next Packaging-Oriented Additions

The next repository additions should focus on delivery, not on new licensing behavior:

- reproducible dependency inputs for the Windows server bundle
- bundle-staging logic for the Windows service package
- bundled admin wrapper commands for customer IT
- installed-artifact smoke support outside the repo

## What Should Stay Out Of Scope

- browser-based customer admin UI
- multi-license pools on one server
- borrowed or offline seats
- redundant failover servers
- packaging the internal vendor tool for customer delivery

## Current Validation Boundary

Today the repo already supports:

1. source-based server init and request export
2. source-based vendor issuance
3. source-based license import and status
4. source-based checkout and release
5. GUI-side licensing tests

The remaining gap is Windows packaging plus clean-install validation from installed artifacts.

## Related Docs

- `windows_packaging_and_clean_install.md`
- `license_server_implementation_checklist.md`
- `../architecture/license_server_architecture.md`
