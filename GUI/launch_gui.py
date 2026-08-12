"""Launch the PySide6 desktop GUI for the surrogate-model workflow.

This file intentionally stays tiny so there is one obvious GUI entry point for:

- local development
- manual smoke tests`
- future packaging into a desktop application
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def _preload_xcb_cursor() -> None:
    """Make Qt's xcb platform plugin loadable on hosts without libxcb-cursor.

    PySide6 >= 6.5 links its xcb plugin against ``libxcb-cursor.so.0``, which RHEL 8
    and similar distributions do not ship (the package is ``libxcb-cursor0`` /
    ``xcb-cursor0``). Qt then aborts with "Could not load the Qt platform plugin
    xcb". Conda/venv environments usually do provide the library, but their ``lib``
    directory is not on the dynamic loader's search path, so the plugin's own
    ``dlopen`` cannot see it. Loading it into this process first with RTLD_GLOBAL
    satisfies the plugin without requiring LD_LIBRARY_PATH to be set by hand.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        # Already resolvable (system package present): nothing to do.
        ctypes.CDLL("libxcb-cursor.so.0", mode=ctypes.RTLD_GLOBAL)
        return
    except OSError:
        pass
    candidate = Path(sys.prefix) / "lib" / "libxcb-cursor.so.0"
    try:
        ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        # Leave Qt to report the problem itself rather than masking it here.
        pass


_preload_xcb_cursor()

from xfmr_v2.gui_window import create_application  # noqa: E402  (must follow the preload)


def main() -> None:
    # `create_application` centralizes theme setup and window construction so the
    # entrypoint only needs to show the window and hand control to Qt's event loop.
    app, window = create_application()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
