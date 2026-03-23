# Tech Stack: Deployment

## Current Stack

- `pyside6-deploy` + Nuitka for the Windows GUI standalone build
- Inno Setup 6 for the per-user Windows GUI installer
- bundled Python runtime or dedicated bundled virtual environment for the Windows server
- WinSW for Windows service hosting
- PowerShell for Windows install and service bootstrap
- installed-artifact smoke rehearsal outside the repo

## Why This Fits

- It keeps the GUI aligned with the current Qt-for-Python direction.
- It gives design engineers a normal Windows installer experience.
- It keeps the Windows server operational model familiar to customer IT.
- It avoids requiring a separate host Python install for the customer server.
- It gives release validation a clean boundary between source-based tests and packaged-install acceptance.

## GUI Delivery Stack

- `launch_gui.py` as the GUI entrypoint
- `pyside6-deploy` spec rendering from tracked templates
- Nuitka standalone output
- Inno Setup wrapping for per-user install, shortcuts, and uninstall behavior

## Server Delivery Stack

- `license_server` package code
- FastAPI + Uvicorn runtime dependencies bundled with the server
- WinSW service host and rendered XML
- PowerShell install script
- wrapper commands for customer-admin CLI access

## Validation Stack

- source-based pytest coverage for fast regressions
- source-based smoke rehearsal as the development baseline
- packaged-install rehearsal under `MLP_modeling_v2_runtime\install-smoke\...`

## Keep

- separate GUI and server packaging inputs
- writable state outside install directories
- CLI-first customer admin model
- clean-install validation as a release gate

## Avoid

- relying on the repo `.venv` as the customer runtime
- requiring customer IT to fetch WinSW separately
- mixing source-checkout path assumptions into packaged artifacts
- turning clean-install acceptance into a purely manual memory-based process

## Bottom Line

Use **`pyside6-deploy` + Nuitka + Inno Setup** for the Windows GUI and
**bundled Python runtime + WinSW + PowerShell** for the Windows license server,
with a dedicated installed-artifact rehearsal step outside the repo.

## References

- `tech_stack_gui.md`
- `tech_stack_license_server.md`
- `../implementation/windows_packaging_and_clean_install.md`
