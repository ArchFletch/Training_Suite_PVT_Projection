"""Tests for deployment config loading and the warnings on its fallback paths."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from license_server.service.config import (
    CONFIG_PATH_ENV,
    DEFAULT_GRACE_SECONDS,
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    DEFAULT_LEASE_TTL_SECONDS,
    build_server_config,
    load_lease_settings,
)


def _write_config(tmp_path: Path, body: str) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def test_missing_config_file_is_silent(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Local development runs without a config file, so absence must not warn."""

    caplog.set_level(logging.DEBUG)

    assert load_lease_settings(tmp_path / "config.toml") == {}
    assert caplog.records == []


def test_unreadable_config_file_warns_and_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A config the service account cannot open must not be a silent downgrade."""

    config_path = _write_config(tmp_path, "[leases]\nlease_ttl_seconds = 90\n")
    real_open = Path.open

    # chmod would not deny the owner on every platform, so deny the read directly.
    def denied_open(self: Path, *args: object, **kwargs: object):
        if self == config_path:
            raise PermissionError(13, "Permission denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_open)
    caplog.set_level(logging.WARNING)

    assert load_lease_settings(config_path) == {}
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert str(config_path) in record.getMessage()
    assert "Permission denied" in record.getMessage()


def test_malformed_config_file_warns_and_falls_back(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_path = _write_config(tmp_path, "[leases\nlease_ttl_seconds = 90\n")
    caplog.set_level(logging.WARNING)

    assert load_lease_settings(config_path) == {}
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert str(config_path) in caplog.records[0].getMessage()


def test_valid_config_file_applies_lease_settings(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config_path = _write_config(
        tmp_path,
        "[leases]\nheartbeat_interval_seconds = 45\nlease_ttl_seconds = 180\nheartbeat_grace_seconds = 600\n",
    )
    caplog.set_level(logging.WARNING)

    assert load_lease_settings(config_path) == {
        "heartbeat_interval_seconds": 45,
        "lease_ttl_seconds": 180,
        "grace_seconds": 600,
    }
    assert caplog.records == []


def test_rejected_lease_values_warn_and_are_ignored(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config_path = _write_config(
        tmp_path,
        '[leases]\nheartbeat_interval_seconds = "45"\nlease_ttl_seconds = 0\nheartbeat_grace_seconds = 600\n',
    )
    caplog.set_level(logging.WARNING)

    assert load_lease_settings(config_path) == {"grace_seconds": 600}
    warned_keys = [
        key
        for key in ("heartbeat_interval_seconds", "lease_ttl_seconds")
        if any(key in record.getMessage() for record in caplog.records)
    ]
    assert warned_keys == ["heartbeat_interval_seconds", "lease_ttl_seconds"]
    assert len(caplog.records) == 2


def test_non_table_leases_block_warns_and_falls_back(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config_path = _write_config(tmp_path, "leases = 90\n")
    caplog.set_level(logging.WARNING)

    assert load_lease_settings(config_path) == {}
    assert len(caplog.records) == 1
    assert str(config_path) in caplog.records[0].getMessage()


def test_server_config_still_builds_on_defaults_when_config_is_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Refusing to boot on a bad config file would be worse than the fallback."""

    config_path = _write_config(tmp_path, "[leases\nlease_ttl_seconds = 90\n")
    monkeypatch.setenv(CONFIG_PATH_ENV, str(config_path))
    caplog.set_level(logging.WARNING)

    config = build_server_config(runtime_root=tmp_path / "runtime", os_family="windows")

    assert config.heartbeat_interval_seconds == DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    assert config.lease_ttl_seconds == DEFAULT_LEASE_TTL_SECONDS
    assert config.grace_seconds == DEFAULT_GRACE_SECONDS
    assert len(caplog.records) == 1
