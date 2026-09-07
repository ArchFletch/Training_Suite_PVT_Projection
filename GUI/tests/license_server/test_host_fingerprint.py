"""Stability of the host fingerprint the license is bound to.

The fingerprint is computed once by `init` and stored, so it only has to be
reproducible when the runtime directory is rebuilt -- which is exactly when
getting it wrong is expensive, because the license no longer imports and has to
be re-issued. Anything that is not a genuine property of the host must not
change it.
"""

from __future__ import annotations

import platform
import socket

from license_server.crypto import build_host_fingerprint, detect_hostname


def _fingerprint(hostname: str, os_family: str = "linux") -> str:
    # A fixed machine token: /etc/machine-id is the third input and is not what
    # these tests are about.
    return build_host_fingerprint(hostname=hostname, os_family=os_family, machine_token="token-fixed")


def test_hostname_casing_does_not_change_the_fingerprint() -> None:
    """`PC` and `pc` are the same host. Which casing `gethostname()` returns
    depends on the resolution path that answers, so an unfolded name made the
    identity depend on something outside the host's control."""

    assert _fingerprint("PC") == _fingerprint("pc")
    assert _fingerprint("MLP-License-01") == _fingerprint("mlp-license-01")


def test_surrounding_whitespace_does_not_change_the_fingerprint() -> None:
    assert _fingerprint(" pc ") == _fingerprint("pc")


def test_os_family_casing_does_not_change_the_fingerprint() -> None:
    """`platform.system()` returns "Linux" while the config default is "linux";
    both reach build_host_fingerprint depending on the call path."""

    assert _fingerprint("pc", "Linux") == _fingerprint("pc", "linux")
    assert _fingerprint("pc", "Windows") == _fingerprint("pc", "windows")


def test_a_genuine_rename_still_changes_the_fingerprint() -> None:
    """Folding case must not weaken the binding itself."""

    assert _fingerprint("pc") != _fingerprint("mlp-license-01")


def test_the_machine_token_is_part_of_the_binding() -> None:
    """Reinstalling the OS or cloning the VM changes /etc/machine-id, and that
    has to invalidate the license the same way a rename does."""

    first = build_host_fingerprint(hostname="pc", os_family="linux", machine_token="token-a")
    second = build_host_fingerprint(hostname="pc", os_family="linux", machine_token="token-b")
    assert first != second


def test_detect_hostname_folds_case(monkeypatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "MLP-License-01")
    assert detect_hostname() == "mlp-license-01"


def test_detect_hostname_falls_back_and_still_folds(monkeypatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "   ")
    monkeypatch.setattr(platform, "node", lambda: "Fallback-Host")
    assert detect_hostname() == "fallback-host"


def test_detect_hostname_never_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "")
    monkeypatch.setattr(platform, "node", lambda: "")
    assert detect_hostname() == "unknown-host"
