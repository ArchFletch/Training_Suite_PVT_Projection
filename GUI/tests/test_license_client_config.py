"""Tests for the desktop license client config helper."""

from __future__ import annotations

import xfmr_v2.app_paths as app_paths
from xfmr_v2.licensing.client_config import (
    license_client_config_path,
    load_license_client_config,
    save_license_client_config,
)


def test_license_client_config_round_trips_in_packaged_linux_mode(monkeypatch, tmp_path) -> None:
    home_dir = tmp_path / "home"
    config_home = tmp_path / "xdg" / "config"
    home_dir.mkdir(parents=True)

    monkeypatch.setenv("MLP_APP_PATH_MODE", "packaged")
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg" / "state"))
    monkeypatch.setattr(app_paths.sys, "platform", "linux")

    payload = {"server_url": "http://mlp-license-01:27850"}
    path = save_license_client_config(payload)

    assert path == license_client_config_path()
    assert path.is_file()
    assert load_license_client_config() == payload


def test_a_bom_prefixed_config_from_powershell_is_read() -> None:
    """Windows PowerShell 5's `Set-Content -Encoding UTF8`, which the pre-seed helper
    scripts/configure_gui_license_server.ps1 uses, writes a UTF-8 BOM; json.loads
    rejects a leading BOM, so the file IT actually creates on Windows was reported
    as unreadable and ignored."""
    path = license_client_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xef\xbb\xbf" + b'{\r\n    "server_url":  "http://mlp-license-01:27850"\r\n}\r\n')

    assert load_license_client_config() == {"server_url": "http://mlp-license-01:27850"}
