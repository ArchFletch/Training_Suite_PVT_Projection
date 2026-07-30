"""API tests for the customer-facing license-server routes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from fastapi.testclient import TestClient

from license_server.api.app import create_app
from license_server.crypto import sign_license_payload
from license_server.schemas import LicensePayload, SignedLicense
from license_server.service import build_server_config


def _generate_signing_keys() -> tuple[str, str]:
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    return private_pem, public_pem


def _build_license(identity_payload: dict[str, object], *, private_key_pem: str, seat_count: int = 1) -> SignedLicense:
    now = datetime.now(timezone.utc)
    payload = LicensePayload.create(
        license_id="lic_test_001",
        license_type="evaluation",
        company_name="Acme Design House",
        server_id=str(identity_payload["server_id"]),
        host_fingerprint=str(identity_payload["host_fingerprint"]),
        seat_count=seat_count,
        features=["baseline", "transfer"],
        starts_at=now - timedelta(days=1),
        ends_at=now + timedelta(days=7),
        notes="Local API route test license.",
    )
    return SignedLicense(
        algorithm="Ed25519",
        key_id="test-key",
        payload=payload,
        signature=sign_license_payload(payload, private_key_pem=private_key_pem),
    )


def test_client_routes_cover_a_full_single_lease_flow(tmp_path: Path) -> None:
    """A single desktop client should be able to status-check, lease, heartbeat, and release."""

    private_key_pem, public_key_pem = _generate_signing_keys()
    config = build_server_config(
        runtime_root=tmp_path / "server-runtime",
        os_family="windows",
        vendor_public_key_pem=public_key_pem,
    )
    app = create_app(config=config)
    runtime = app.state.runtime
    identity = runtime.identity_service.ensure_server_identity()
    runtime.license_service.import_license(
        _build_license(identity.to_dict(), private_key_pem=private_key_pem, seat_count=1)
    )

    with TestClient(app) as client:
        status_before = client.get("/api/v1/status")
        assert status_before.status_code == 200
        assert status_before.json()["ok"] is True
        assert status_before.json()["seat_count"] == 1
        assert status_before.json()["seats_in_use"] == 0

        checkout_response = client.post(
            "/api/v1/checkout",
            json={
                "product": "Surrogate Model Training Suite",
                "product_version": "0.1.0",
                "machine_id": "cli_7bde9f61",
                "hostname": "eda-win-17",
                "username": "jdoe",
                "platform": "windows",
            },
        )
        assert checkout_response.status_code == 200
        checkout_payload = checkout_response.json()
        assert checkout_payload["granted"] is True
        assert checkout_payload["seat_count"] == 1
        assert checkout_payload["seats_in_use"] == 1
        assert checkout_payload["lease_id"].startswith("lease_")

        denied_response = client.post(
            "/api/v1/checkout",
            json={
                "product": "Surrogate Model Training Suite",
                "product_version": "0.1.0",
                "machine_id": "cli_7bde9f62",
                "hostname": "eda-win-18",
                "username": "asmith",
                "platform": "windows",
            },
        )
        assert denied_response.status_code == 200
        assert denied_response.json() == {
            "granted": False,
            "reason_code": "all_seats_in_use",
            "message": "All floating seats are currently in use.",
        }

        heartbeat_response = client.post(
            "/api/v1/heartbeat",
            json={
                "lease_id": checkout_payload["lease_id"],
                "machine_id": "cli_7bde9f61",
            },
        )
        assert heartbeat_response.status_code == 200
        assert heartbeat_response.json()["ok"] is True
        assert "expires_at" in heartbeat_response.json()

        release_response = client.post(
            "/api/v1/release",
            json={"lease_id": checkout_payload["lease_id"]},
        )
        assert release_response.status_code == 200
        assert release_response.json() == {
            "ok": True,
            "message": "Lease released.",
        }

        status_after = client.get("/api/v1/status")
        assert status_after.status_code == 200
        assert status_after.json()["seats_in_use"] == 0

    with runtime.session_factory.session() as connection:
        audit_events = runtime.repository.list_recent_audit_events(connection, limit=10)
    event_types = [event.event_type for event in audit_events]
    assert "checkout_granted" in event_types
    assert "heartbeat_renewed" in event_types
    assert "lease_released" in event_types
