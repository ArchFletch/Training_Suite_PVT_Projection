"""Launch the PySide6 desktop GUI for the surrogate-model workflow.

This file intentionally stays tiny so there is one obvious GUI entry point for:

- local development
- manual smoke tests`
- future packaging into a desktop application
"""

from __future__ import annotations

from xfmr_v2.gui_window import create_application


def main() -> None:
    # `create_application` centralizes theme setup and window construction so the
    # entrypoint only needs to show the window and hand control to Qt's event loop.
    app, window = create_application()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
