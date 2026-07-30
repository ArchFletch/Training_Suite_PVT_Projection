from __future__ import annotations

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
    release = client.release(lease_id="lease_001")

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
            {"lease_id": "lease_001"},
            9.5,
        ),
    ]
