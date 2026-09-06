from __future__ import annotations

import pytest

import json

from xfmr_v2.licensing import LicenseHttpClient, LicenseIdentity
import xfmr_v2.licensing.client as license_client_module


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_http_client_calls_status_checkout_heartbeat_and_release(monkeypatch) -> None:
    requests: list[tuple[str, str, dict[str, object] | None, float]] = []
    responses = iter(
        [
            {"ok": True, "company_name": "Acme", "seat_count": 2, "seats_in_use": 1},
            {"granted": True, "lease_id": "lease_001", "heartbeat_interval_seconds": 30, "lease_ttl_seconds": 120, "grace_seconds": 300},
            {"ok": True, "expires_at": "2026-03-13T23:15:00Z"},
            {"ok": True},
        ]
    )

    def fake_urlopen(request, timeout: float):
        payload = json.loads(request.data.decode("utf-8")) if request.data else None
        requests.append((request.full_url, request.get_method(), payload, timeout))
        return _FakeResponse(next(responses))

    monkeypatch.setattr(license_client_module.urllib.request, "urlopen", fake_urlopen)

    client = LicenseHttpClient("http://license-host:27850", timeout_seconds=9.5)
    identity = LicenseIdentity(
        product_version="0.1.0",
        machine_id="gui_machine",
        hostname="eda-win-17",
        username="jdoe",
        platform="windows",
    )

    status = client.get_status()
    checkout = client.checkout(identity)
    heartbeat = client.heartbeat(lease_id="lease_001", machine_id="gui_machine")
    release = client.release(lease_id="lease_001", machine_id="gui_machine")

    assert status.company_name == "Acme"
    assert checkout.granted is True
    assert checkout.lease_id == "lease_001"
    assert heartbeat.ok is True
    assert release.ok is True
    assert requests == [
        ("http://license-host:27850/api/v1/status", "GET", None, 9.5),
        (
            "http://license-host:27850/api/v1/checkout",
            "POST",
            {
                "product": "Surrogate Model Training Suite",
                "product_version": "0.1.0",
                "machine_id": "gui_machine",
                "hostname": "eda-win-17",
                "username": "jdoe",
                "platform": "windows",
            },
            9.5,
        ),
        (
            "http://license-host:27850/api/v1/heartbeat",
            "POST",
            {"lease_id": "lease_001", "machine_id": "gui_machine"},
            9.5,
        ),
        (
            "http://license-host:27850/api/v1/release",
            "POST",
            {"lease_id": "lease_001", "machine_id": "gui_machine"},
            9.5,
        ),
    ]


def test_release_result_carries_the_server_rejection_reason(monkeypatch) -> None:
    def fake_urlopen(request, timeout: float):
        return _FakeResponse(
            {
                "ok": False,
                "reason_code": "machine_mismatch",
                "message": "The lease is owned by a different machine_id.",
            }
        )

    monkeypatch.setattr(license_client_module.urllib.request, "urlopen", fake_urlopen)

    result = LicenseHttpClient("http://license-host:27850").release(
        lease_id="lease_001", machine_id="gui_other_machine"
    )

    assert result.ok is False
    assert result.reason_code == "machine_mismatch"
    assert result.message == "The lease is owned by a different machine_id."


def test_a_bare_host_gets_a_scheme_instead_of_a_traceback() -> None:
    """'licsrv01:27850' is the most likely thing a user types into a field labelled
    License Server. Without a scheme urllib raised a bare ValueError out of the
    client, which reached the GUI as a raw traceback dialog."""
    from xfmr_v2.licensing import normalize_server_url

    assert normalize_server_url("licsrv01:27850") == "http://licsrv01:27850"
    assert normalize_server_url("  licsrv01/ ") == "http://licsrv01"
    assert normalize_server_url("https://lic.example.com/") == "https://lic.example.com"
    assert normalize_server_url("") == ""
    assert license_client_module.LicenseHttpClient("licsrv01:27850").api_base == "http://licsrv01:27850/api/v1"


def test_a_malformed_url_is_a_license_error_not_a_raw_exception(monkeypatch) -> None:
    """A space in the host makes http.client raise InvalidURL, which is neither a
    URLError nor an OSError and so escaped every `except LicenseClientError`."""
    import http.client

    def fake_urlopen(request, timeout: float):
        raise http.client.InvalidURL("URL can't contain control characters")

    monkeypatch.setattr(license_client_module.urllib.request, "urlopen", fake_urlopen)
    client = license_client_module.LicenseHttpClient("http://lic srv01:27850")

    with pytest.raises(license_client_module.LicenseClientError) as excinfo:
        client.get_status()
    assert "not a valid license server URL" in str(excinfo.value)
    assert "http://host:port" in str(excinfo.value)
