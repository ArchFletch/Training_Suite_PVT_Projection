"""Persistence helpers for desktop-side license client settings.

``license_client.json`` is the per-user file an administrator pre-seeds with
the licence server URL (scripts/configure_gui_license_server.ps1, the install
docs). The GUI reads it at startup -- it wins over the URL remembered in the
session file -- and rewrites it on close, so the two only disagree after a
deliberate re-seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..app_paths import current_runtime_paths
from ..atomic_json import write_json_atomically


def license_client_config_path() -> Path:
    """Return the shared per-user config file for the desktop license client."""

    return current_runtime_paths().license_client_path


def load_license_client_config() -> dict[str, Any]:
    """Load the saved desktop licensing settings when available."""

    path = license_client_config_path()
    if not path.is_file():
        return {}
    # utf-8-sig: Windows PowerShell 5's `Set-Content -Encoding UTF8` -- what the
    # pre-seed helper scripts/configure_gui_license_server.ps1 uses -- writes a
    # UTF-8 BOM, and json.loads rejects a leading BOM in a str.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_license_client_config(payload: dict[str, Any]) -> Path:
    """Persist desktop licensing settings and return the saved file path."""

    return write_json_atomically(license_client_config_path(), payload)
