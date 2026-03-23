"""Persistence helpers for desktop-side license client settings.

The packaging work establishes the writable location contract for the future GUI
licensing flow. The licensing agent can store the on-prem server URL here
without needing to re-decide OS-specific paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..app_paths import current_runtime_paths


def license_client_config_path() -> Path:
    """Return the shared per-user config file for the desktop license client."""

    return current_runtime_paths().license_client_path


def load_license_client_config() -> dict[str, Any]:
    """Load the saved desktop licensing settings when available."""

    path = license_client_config_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_license_client_config(payload: dict[str, Any]) -> Path:
    """Persist desktop licensing settings and return the saved file path."""

    path = license_client_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
