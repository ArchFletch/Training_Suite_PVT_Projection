"""Tests for the shared runtime path policy."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from xfmr_v2.app_paths import APP_NAME, APP_SLUG, current_runtime_paths


def test_source_checkout_paths_keep_repo_artifacts_layout() -> None:
    paths = current_runtime_paths(platform="win32", env={"MLP_APP_PATH_MODE": "source"})
    repo_root = Path(__file__).resolve().parents[1]

    assert paths.mode == "source"
    assert paths.config_dir == repo_root / "artifacts" / "gui"
    assert paths.state_dir == repo_root / "artifacts" / "gui"
    assert paths.cache_dir == repo_root / "artifacts" / "cache"
    assert paths.default_output_dir == repo_root / "artifacts" / "output"
    assert paths.license_client_path == repo_root / "artifacts" / "gui" / "license_client.json"
    assert paths.last_session_path == repo_root / "artifacts" / "gui" / "last_session.json"


def test_packaged_windows_paths_use_appdata_and_localappdata(tmp_path: Path) -> None:
    env = {
        "MLP_APP_PATH_MODE": "packaged",
        "APPDATA": str(tmp_path / "AppData" / "Roaming"),
        "LOCALAPPDATA": str(tmp_path / "AppData" / "Local"),
        "USERPROFILE": str(tmp_path / "User"),
    }

    paths = current_runtime_paths(platform="win32", env=env)

    assert paths.mode == "packaged"
    assert paths.config_dir == tmp_path / "AppData" / "Roaming" / APP_NAME
    assert paths.state_dir == tmp_path / "AppData" / "Local" / APP_NAME
    assert paths.cache_dir == tmp_path / "AppData" / "Local" / APP_NAME / "cache"
    assert paths.log_dir == tmp_path / "AppData" / "Local" / APP_NAME / "logs"
    assert paths.default_output_dir == tmp_path / "User" / "Documents" / APP_NAME / "Runs"
    assert paths.license_client_path == paths.config_dir / "license_client.json"
    assert paths.last_session_path == paths.state_dir / "last_session.json"


def test_packaged_linux_paths_use_xdg_locations(tmp_path: Path) -> None:
    env = {
        "MLP_APP_PATH_MODE": "packaged",
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg" / "config"),
        "XDG_STATE_HOME": str(tmp_path / "xdg" / "state"),
    }

    paths = current_runtime_paths(platform="linux", env=env)

    assert paths.mode == "packaged"
    assert paths.config_dir == tmp_path / "xdg" / "config" / APP_SLUG
    assert paths.state_dir == tmp_path / "xdg" / "state" / APP_SLUG
    assert paths.cache_dir == tmp_path / "xdg" / "state" / APP_SLUG / "cache"
    assert paths.log_dir == tmp_path / "xdg" / "state" / APP_SLUG / "logs"
    assert paths.default_output_dir == tmp_path / "home" / "Documents" / APP_NAME / "runs"
    assert paths.license_client_path == paths.config_dir / "license_client.json"
    assert paths.last_session_path == paths.state_dir / "last_session.json"


@pytest.mark.parametrize("override", ["source", "SOURCE", " source ", "", "garbage"])
def test_a_frozen_build_is_packaged_whatever_the_override_says(monkeypatch, tmp_path: Path, override: str) -> None:
    """The source override exists for tests and docs helpers running from a
    checkout. A shipped build must ignore it: it used to be honoured ahead of
    the freeze markers, so `MLP_APP_PATH_MODE=source` talked a frozen build into
    source mode -- the mode the development licence bypass keys off, and the one
    that writes state into the install directory."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    paths = current_runtime_paths(
        platform="linux", env={"MLP_APP_PATH_MODE": override, "HOME": str(tmp_path)}
    )

    assert paths.mode == "packaged", f"MLP_APP_PATH_MODE={override!r} downgraded a frozen build"
    assert paths.state_dir == tmp_path / ".local" / "state" / APP_SLUG


def test_the_override_still_promotes_a_source_checkout_to_packaged(monkeypatch, tmp_path: Path) -> None:
    """The one direction the override is for: exercising the packaged path
    policy from a checkout without a frozen executable."""
    monkeypatch.delattr(sys, "frozen", raising=False)

    packaged = current_runtime_paths(
        platform="linux", env={"MLP_APP_PATH_MODE": "packaged", "HOME": str(tmp_path)}
    )
    source = current_runtime_paths(platform="linux", env={"MLP_APP_PATH_MODE": "source"})

    assert packaged.mode == "packaged"
    assert source.mode == "source"
