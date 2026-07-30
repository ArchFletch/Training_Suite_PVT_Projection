# Tech Stack: GUI

## Current Stack

- `PySide6` for the desktop application shell
- `pyqtgraph` for live baseline and transfer plots
- Qt worker threads and signal-slot callback delivery for long-running tasks
- JSON config, session, and license-client persistence
- `pytest` plus `pytest-qt` for GUI regression coverage
- `pyside6-deploy` + Nuitka for the Windows standalone build
- Inno Setup 6 for the per-user Windows installer

## Why This Fits

- `PySide6` is a strong fit for a technical desktop app on Windows and Linux.
- `pyqtgraph` performs better than Matplotlib for live-updating Qt-native plots.
- Qt worker threads keep the UI responsive during scans, suggestion generation, and training.
- JSON persistence is easy to inspect, debug, and support across CLI and GUI workflows.
- `pytest-qt` gives us headless and offscreen GUI verification instead of relying only on manual checks.
- Inno Setup gives Windows users a normal installer, shortcuts, and uninstall behavior.

## Plotting Strategy

- use `pyqtgraph` for live interactive plots inside the GUI
- continue allowing Matplotlib or other backend-side figure generation for saved training artifacts
- keep plot styling centralized in the GUI theme layer

## Testing Strategy

- use `pytest` for backend and GUI tests
- use `pytest-qt` for widget interaction and progress-update tests
- use `QT_QPA_PLATFORM=offscreen` for automated GUI smoke checks in non-interactive environments
- add packaged-install validation for the installed Windows EXE outside the repo

## Keep

- thin GUI over backend modules
- backend logic outside widget code
- Qt-native threading model
- file-based output model
- plain-dict progress payloads between backend and UI
- per-user Windows install with writable state outside the install tree

## Avoid

- Electron for this local technical workflow
- Tauri unless there is a future product reason to pivot to web technologies
- browser-first local web UI for the current single-user desktop workflow
- writing packaged runtime state into `%LOCALAPPDATA%\Programs\Surrogate Model Training Suite\`

## Likely Later Additions

- code signing for the Windows installer
- dedicated artifact and result browser
- deeper license-state UX polish

## Bottom Line

Stay with **PySide6 + pyqtgraph + Qt worker threads + JSON persistence +
GUI test automation through `pytest-qt` + `pyside6-deploy` + Nuitka + Inno Setup**.

## References

- [Qt for Python Docs](https://doc.qt.io/qtforpython-6/)
- [Qt `pyside6-deploy` Docs](https://doc.qt.io/qtforpython-6/deployment/deployment-pyside6-deploy.html)
- [pytest-qt Docs](https://pytest-qt.readthedocs.io/)
- `tech_stack_deployment.md`
