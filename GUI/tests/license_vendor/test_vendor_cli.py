"""Tests for the vendor CLI issuance flow."""

from __future__ import annotations

import json
from pathlib import Path

from license_server.crypto import verify_signed_license
from license_server.schemas import DEFAULT_PRODUCT, LicenseRequest, SignedLicenseEnvelope
from license_vendor.cli import main
from license_vendor.issuance_log import read_issuance_log
from license_vendor.signer import generate_signing_key, save_signing_key, verify_license_signature


def _write_request_file(path: Path) -> Path:
    request = LicenseRequest(
        schema_version=1,
        product=DEFAULT_PRODUCT,
        server_id="srv_demo001",
        host_fingerprint="host_demo001",
        hostname="mlp-license-01",
        os_family="windows",
        generated_at="2026-03-13T12:00:00Z",
        requested_by="customer_it",
    )
    path.write_text(request.to_json(), encoding="utf-8")
    return path


def test_cli_issues_eval_and_paid_licenses_for_same_server_and_appends_log(
    tmp_path: Path, capsys
) -> None:
    """Eval and paid issuance should share one server identity and one append-only log."""

    request_path = _write_request_file(tmp_path / "license_request.json")
    key_path = tmp_path / "vendor_signing_key.json"
    issuance_log_path = tmp_path / "issuance_log.jsonl"
    eval_output = tmp_path / "eval_license.json"
    paid_output = tmp_path / "paid_license.json"

    signing_key = generate_signing_key(
        "2026-01", seed=bytes.fromhex("ab" * 32)
    )
    save_signing_key(key_path, signing_key)

    exit_code = main(
        [
            "issue-eval",
            "--request-file",
            str(request_path),
            "--signing-key-file",
            str(key_path),
            "--company-name",
            "Acme Design House",
            "--seat-count",
            "2",
            "--starts-at",
            "2026-03-13",
            "--term-days",
            "14",
            "--license-id",
            "lic_eval_0001",
            "--output",
            str(eval_output),
            "--issuance-log",
            str(issuance_log_path),
        ]
    )
    eval_summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert eval_summary["status"] == "ok"
    assert eval_summary["license_type"] == "evaluation"
    assert eval_summary["server_id"] == "srv_demo001"

    exit_code = main(
        [
            "issue-paid",
            "--request-file",
            str(request_path),
            "--signing-key-file",
            str(key_path),
            "--company-name",
            "Acme Design House",
            "--seat-count",
            "5",
            "--starts-at",
            "2026-03-27",
            "--ends-at",
            "2027-03-26",
            "--feature",
            "baseline",
            "--feature",
            "transfer",
            "--license-id",
            "lic_paid_0001",
            "--output",
            str(paid_output),
            "--issuance-log",
            str(issuance_log_path),
        ]
    )
    paid_summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert paid_summary["status"] == "ok"
    assert paid_summary["license_type"] == "paid"
    assert paid_summary["server_id"] == "srv_demo001"

    eval_envelope = SignedLicenseEnvelope.from_path(eval_output)
    paid_envelope = SignedLicenseEnvelope.from_path(paid_output)

    assert eval_envelope.payload.server_id == paid_envelope.payload.server_id == "srv_demo001"
    assert eval_envelope.payload.host_fingerprint == paid_envelope.payload.host_fingerprint
    assert eval_envelope.payload.license_type == "evaluation"
    assert paid_envelope.payload.license_type == "paid"
    assert eval_envelope.payload.features == paid_envelope.payload.features == [
        "baseline",
        "transfer",
    ]
    assert verify_license_signature(eval_envelope, signing_key.public_key)
    assert verify_license_signature(paid_envelope, signing_key.public_key)

    log_records = read_issuance_log(issuance_log_path)

    assert [record["license_type"] for record in log_records] == ["evaluation", "paid"]
    assert {record["server_id"] for record in log_records} == {"srv_demo001"}
    assert [record["license_id"] for record in log_records] == [
        "lic_eval_0001",
        "lic_paid_0001",
    ]
    assert log_records[0]["output_path"] == str(eval_output)
    assert log_records[1]["output_path"] == str(paid_output)


def test_cli_exports_public_key_pem_that_verifies_its_own_licenses(
    tmp_path: Path, capsys
) -> None:
    """The exported PEM should be the key the customer server verifies vendor licenses with."""

    # Imported here rather than at module scope: the vendor side signs with a
    # pure-Python Ed25519 and license_server.crypto lazy-imports cryptography behind
    # a friendly error, so a module-level import would make the whole tests/license_vendor
    # package uncollectable on a machine that deliberately does without it.
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    request_path = _write_request_file(tmp_path / "license_request.json")
    key_path = tmp_path / "vendor_signing_key.json"
    pem_path = tmp_path / "vendor_public_key.pem"
    license_output = tmp_path / "license.json"

    signing_key = generate_signing_key("2026-03", seed=bytes.fromhex("cd" * 32))
    save_signing_key(key_path, signing_key)

    exit_code = main(
        [
            "export-public-key",
            "--signing-key-file",
            str(key_path),
            "--output",
            str(pem_path),
        ]
    )
    export_summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert export_summary["status"] == "ok"
    assert export_summary["key_id"] == "2026-03"
    assert export_summary["output_path"] == str(pem_path)

    public_key_pem = pem_path.read_bytes()
    exported_key = serialization.load_pem_public_key(public_key_pem)

    assert isinstance(exported_key, ed25519.Ed25519PublicKey)
    assert (
        exported_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        == signing_key.public_key
    )

    exit_code = main(
        [
            "issue-eval",
            "--request-file",
            str(request_path),
            "--signing-key-file",
            str(key_path),
            "--company-name",
            "Acme Design House",
            "--starts-at",
            "2026-03-13",
            "--term-days",
            "14",
            "--license-id",
            "lic_eval_0002",
            "--output",
            str(license_output),
            "--issuance-log",
            str(tmp_path / "issuance_log.jsonl"),
        ]
    )
    capsys.readouterr()

    assert exit_code == 0

    envelope = SignedLicenseEnvelope.from_path(license_output)

    verify_signed_license(envelope, public_key_pem=public_key_pem)
