"""CLI tests for the customer-admin license-server workflow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from typer.testing import CliRunner

from license_server.cli.main import app
from license_server.crypto import sign_license_payload
from license_server.schemas import LicensePayload, SignedLicense


RUNNER = CliRunner()


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


def _build_license(request_payload: dict[str, object], *, private_key_pem: str, seat_count: int = 2) -> SignedLicense:
    now = datetime.now(timezone.utc)
    payload = LicensePayload.create(
        license_id="lic_cli_001",
        license_type="evaluation",
        company_name="Acme Design House",
        server_id=str(request_payload["server_id"]),
        host_fingerprint=str(request_payload["host_fingerprint"]),
        seat_count=seat_count,
        features=["baseline", "transfer"],
        starts_at=now - timedelta(days=1),
        ends_at=now + timedelta(days=14),
        notes="Local CLI test license.",
    )
    return SignedLicense(
        algorithm="Ed25519",
        key_id="test-key",
        payload=payload,
        signature=sign_license_payload(payload, private_key_pem=private_key_pem),
    )


def test_customer_cli_covers_init_export_import_status_and_audit(tmp_path: Path) -> None:
    """Customer IT should be able to complete the MVP admin path from the CLI only."""

    private_key_pem, public_key_pem = _generate_signing_keys()
    runtime_root = tmp_path / "server-runtime"
    request_path = tmp_path / "license_request.json"
    license_path = tmp_path / "license.json"
    public_key_path = tmp_path / "vendor_public_key.pem"
    public_key_path.write_text(public_key_pem, encoding="utf-8")

    init_result = RUNNER.invoke(
        app,
        ["init", "--runtime-root", str(runtime_root)],
    )
    assert init_result.exit_code == 0, init_result.stdout
    init_payload = json.loads(init_result.stdout)
    assert init_payload["ok"] is True
    assert init_payload["identity"]["server_id"].startswith("srv_")

    export_result = RUNNER.invoke(
        app,
        [
            "export-request",
            str(request_path),
            "--runtime-root",
            str(runtime_root),
            "--requested-by",
            "cli-admin",
        ],
    )
    assert export_result.exit_code == 0, export_result.stdout
    export_payload = json.loads(export_result.stdout)
    assert export_payload["ok"] is True
    assert request_path.exists()

    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    license_path.write_text(
        _build_license(request_payload, private_key_pem=private_key_pem).to_json(),
        encoding="utf-8",
    )

    import_result = RUNNER.invoke(
        app,
        [
            "import-license",
            str(license_path),
            "--runtime-root",
            str(runtime_root),
            "--vendor-public-key",
            str(public_key_path),
        ],
    )
    assert import_result.exit_code == 0, import_result.stdout
    import_payload = json.loads(import_result.stdout)
    assert import_payload["ok"] is True
    assert import_payload["license"]["company_name"] == "Acme Design House"

    status_result = RUNNER.invoke(
        app,
        ["show-status", "--runtime-root", str(runtime_root)],
    )
    assert status_result.exit_code == 0, status_result.stdout
    status_payload = json.loads(status_result.stdout)
    assert status_payload["license_loaded"] is True
    assert status_payload["license_id"] == "lic_cli_001"
    assert status_payload["seat_count"] == 2
    assert status_payload["active_leases"] == []

    audit_result = RUNNER.invoke(
        app,
        ["show-audit", "--runtime-root", str(runtime_root), "--limit", "10"],
    )
    assert audit_result.exit_code == 0, audit_result.stdout
    audit_payload = json.loads(audit_result.stdout)
    assert audit_payload["ok"] is True
    event_types = [event["event_type"] for event in audit_payload["events"]]
    assert "server_initialized" in event_types
    assert "request_exported" in event_types
    assert "license_imported" in event_types
