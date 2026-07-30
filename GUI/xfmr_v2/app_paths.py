"""Runtime path policy for source checkouts and packaged desktop builds.

The repository's source-checkout workflow has historically used repo-relative
`artifacts/` paths, which is convenient during development and for tests.
Packaged desktop installs need a different policy because the install directory
is read-only and runtime state should live in user-writable locations.

This module centralizes that split so the GUI, packaging scripts, and future
desktop licensing work all agree on the same writable locations.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

APP_NAME = "Surrogate Model Training Suite"
APP_SLUG = "mlp-training-studio"
LICENSE_CLIENT_FILENAME = "license_client.json"
LAST_SESSION_FILENAME = "last_session.json"
_PATH_MODE_ENV = "MLP_APP_PATH_MODE"


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved writable locations for the current runtime mode."""

    mode: str
    config_dir: Path
    state_dir: Path
    cache_dir: Path
    log_dir: Path
    default_output_dir: Path
    license_client_path: Path
    last_session_path: Path


def current_runtime_paths(
    *,
    platform: str | None = None,
    env: Mapping[str, str] | None = None,
) -> RuntimePaths:
    """Return writable locations for either source or packaged execution.

    `MLP_APP_PATH_MODE=packaged` is intentionally supported so packaging smoke
    tests and documentation helpers can exercise the packaged path policy
    without needing a frozen executable.
    """

    active_platform = platform or sys.platform
    active_env = dict(os.environ if env is None else env)
    if _is_packaged_runtime(env=active_env):
        return _packaged_runtime_paths(platform=active_platform, env=active_env)
    return _source_checkout_runtime_paths()


def _is_packaged_runtime(*, env: Mapping[str, str]) -> bool:
    mode_override = str(env.get(_PATH_MODE_ENV, "")).strip().lower()
    if mode_override == "packaged":
        return True
    if mode_override == "source":
        return False
    return bool(getattr(sys, "frozen", False))


def _source_checkout_runtime_paths() -> RuntimePaths:
    repo_root = Path(__file__).resolve().parent.parent
    artifacts_dir = repo_root / "artifacts"
    gui_state_dir = artifacts_dir / "gui"
    return RuntimePaths(
        mode="source",
        config_dir=gui_state_dir,
        state_dir=gui_state_dir,
        cache_dir=artifacts_dir / "cache",
        log_dir=artifacts_dir / "logs",
        default_output_dir=artifacts_dir / "output",
        license_client_path=gui_state_dir / LICENSE_CLIENT_FILENAME,
        last_session_path=gui_state_dir / LAST_SESSION_FILENAME,
    )


def _packaged_runtime_paths(*, platform: str, env: Mapping[str, str]) -> RuntimePaths:
    home = Path(env.get("USERPROFILE") or env.get("HOME") or Path.home())

    if platform.startswith("win"):
        config_dir = Path(env.get("APPDATA") or (home / "AppData" / "Roaming")) / APP_NAME
        state_dir = Path(env.get("LOCALAPPDATA") or (home / "AppData" / "Local")) / APP_NAME
        default_output_dir = home / "Documents" / APP_NAME / "Runs"
    else:
        config_dir = Path(env.get("XDG_CONFIG_HOME") or (home / ".config")) / APP_SLUG
        state_dir = Path(env.get("XDG_STATE_HOME") or (home / ".local" / "state")) / APP_SLUG
        default_output_dir = home / "Documents" / APP_NAME / "runs"

    return RuntimePaths(
        mode="packaged",
        config_dir=config_dir,
        state_dir=state_dir,
        cache_dir=state_dir / "cache",
        log_dir=state_dir / "logs",
        default_output_dir=default_output_dir,
        license_client_path=config_dir / LICENSE_CLIENT_FILENAME,
        last_session_path=state_dir / LAST_SESSION_FILENAME,
    )
