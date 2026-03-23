"""Ed25519 signature helpers for signed license verification."""

from __future__ import annotations

import base64
import json

from license_server.schemas import LicensePayload, SignedLicense


class LicenseVerificationError(ValueError):
    """Raised when a signed license fails cryptographic verification."""


def canonicalize_payload(payload: LicensePayload) -> bytes:
    """Serialize the payload in a deterministic form before signing or verifying."""

    return json.dumps(payload.to_dict(), separators=(",", ":"), sort_keys=True).encode("utf-8")


def _load_ed25519_private_key(private_key_pem: bytes):
    try:
        from cryptography.hazmat.primitives import serialization
    except ImportError as exc:
        raise RuntimeError("cryptography is required for license signing and verification.") from exc
    return serialization.load_pem_private_key(private_key_pem, password=None)


def _load_ed25519_public_key(public_key_pem: bytes):
    try:
        from cryptography.hazmat.primitives import serialization
    except ImportError as exc:
        raise RuntimeError("cryptography is required for license signing and verification.") from exc
    return serialization.load_pem_public_key(public_key_pem)


def sign_license_payload(payload: LicensePayload, *, private_key_pem: str | bytes) -> str:
    """Sign a payload with an Ed25519 private key and return the base64 signature."""

    private_key = _load_ed25519_private_key(
        private_key_pem.encode("utf-8") if isinstance(private_key_pem, str) else private_key_pem
    )
    try:
        signature = private_key.sign(canonicalize_payload(payload))
    except AttributeError as exc:
        raise LicenseVerificationError("Private key must be an Ed25519 private key.") from exc
    return base64.b64encode(signature).decode("ascii")


def verify_signed_license(envelope: SignedLicense, *, public_key_pem: str | bytes) -> None:
    """Verify the signature on a vendor-issued license envelope."""

    if envelope.algorithm != "Ed25519":
        raise LicenseVerificationError(f"Unsupported license algorithm: {envelope.algorithm}")

    public_key = _load_ed25519_public_key(
        public_key_pem.encode("utf-8") if isinstance(public_key_pem, str) else public_key_pem
    )
    try:
        signature = base64.b64decode(envelope.signature, validate=True)
    except ValueError as exc:
        raise LicenseVerificationError("License signature is not valid base64.") from exc

    try:
        public_key.verify(signature, canonicalize_payload(envelope.payload))
    except AttributeError as exc:
        raise LicenseVerificationError("Public key must be an Ed25519 public key.") from exc
    except Exception as exc:  # pragma: no cover - cryptography exposes an internal exception type.
        raise LicenseVerificationError("License signature verification failed.") from exc
