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
