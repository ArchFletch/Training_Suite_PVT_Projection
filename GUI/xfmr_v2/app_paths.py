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
    without needing a frozen executable. The override only works in that
    direction: a frozen or compiled build is always packaged, whatever the
    variable says.
    """

    active_platform = platform or sys.platform
    active_env = dict(os.environ if env is None else env)
    if _is_packaged_runtime(env=active_env):
        return _packaged_runtime_paths(platform=active_platform, env=active_env)
    return _source_checkout_runtime_paths()


def _is_packaged_runtime(*, env: Mapping[str, str]) -> bool:
    # A frozen build is packaged, full stop. `MLP_APP_PATH_MODE=source` used to
    # be honoured here ahead of the freeze markers, so two environment variables
    # a customer could set talked a shipped build into source mode -- which is
    # exactly the mode the development licence bypass (gui_window.dev_unlicensed_mode)
    # keys off, and which also points every writable path back into the install
    # directory. The override now only promotes a source checkout to packaged.
    if _is_frozen_build():
        return True
    return str(env.get(_PATH_MODE_ENV, "")).strip().lower() == "packaged"


def _is_frozen_build() -> bool:
    # `sys.frozen` alone is not enough: PyInstaller and cx_Freeze set it, but
    # Nuitka -- the freezer this project actually ships with -- does not. It marks
    # every compiled module with `__compiled__` instead. Relying on sys.frozen
    # alone made the packaged app fall through to source mode, where repo_root is
    # derived from __file__, so an installed build wrote its config, cache, logs
    # and default run outputs into its own install directory. That is writable
    # under %LOCALAPPDATA%, so it looked fine and then lost user data on upgrade
    # or uninstall.
    return bool(getattr(sys, "frozen", False)) or "__compiled__" in globals()


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
