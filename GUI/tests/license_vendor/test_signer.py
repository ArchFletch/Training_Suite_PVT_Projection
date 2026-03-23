"""Tests for vendor-side Ed25519 signing helpers."""

from __future__ import annotations

from license_server.schemas import DEFAULT_PRODUCT, LicensePayload, SCHEMA_VERSION
from license_vendor.signer import (
    VendorSigningKey,
    derive_public_key,
    sign_license_payload,
    sign_message,
    verify_license_signature,
)


def test_sign_message_matches_rfc8032_test_vector_1() -> None:
    """The pure-Python helper should match the canonical Ed25519 reference vector."""

    secret_seed = bytes.fromhex(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"
    )
    expected_public_key = bytes.fromhex(
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    )
    expected_signature = bytes.fromhex(
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
    )

    assert derive_public_key(secret_seed) == expected_public_key
    assert sign_message(secret_seed, b"") == expected_signature


def test_sign_license_payload_round_trips_with_shared_contract() -> None:
    """Signed payloads should validate against the shared envelope contract."""

    secret_seed = bytes.fromhex("1f" * 32)
    signing_key = VendorSigningKey(
        key_id="test-key",
        private_key_seed=secret_seed,
        public_key=derive_public_key(secret_seed),
        created_at="2026-03-13T00:00:00Z",
    )
    payload = LicensePayload(
        schema_version=SCHEMA_VERSION,
        license_id="lic_eval_demo",
        license_type="evaluation",
        product=DEFAULT_PRODUCT,
        company_name="Acme Design House",
        server_id="srv_demo001",
        host_fingerprint="host_demo001",
        seat_count=2,
        features=("baseline", "transfer"),
        starts_at="2026-03-13T00:00:00Z",
        ends_at="2026-03-27T23:59:59Z",
        notes="14-day floating POC, non-production",
    )

    envelope = sign_license_payload(payload, signing_key)

    assert envelope.to_dict().keys() == {"algorithm", "key_id", "payload", "signature"}
    assert set(envelope.payload.to_dict()) == {
        "schema_version",
        "license_id",
        "license_type",
        "product",
        "company_name",
        "server_id",
        "host_fingerprint",
        "seat_count",
        "features",
        "starts_at",
        "ends_at",
        "notes",
    }
    assert verify_license_signature(envelope, signing_key.public_key)
