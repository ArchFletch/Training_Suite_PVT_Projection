"""Internal vendor tooling for issuing signed evaluation and paid licenses."""

from .issuance_log import append_issuance_record, build_issuance_record, read_issuance_log
from .signer import (
    VendorSigningKey,
    derive_public_key,
    generate_signing_key,
    load_signing_key,
    save_signing_key,
    sign_license_payload,
    sign_message,
    verify_license_signature,
    verify_message,
)

__all__ = [
    "VendorSigningKey",
    "append_issuance_record",
    "build_issuance_record",
    "derive_public_key",
    "generate_signing_key",
    "load_signing_key",
    "read_issuance_log",
    "save_signing_key",
    "sign_license_payload",
    "sign_message",
    "verify_license_signature",
    "verify_message",
]
