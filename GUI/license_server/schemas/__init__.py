"""Pydantic schemas for request files, signed licenses, and API payloads."""

from .client_api import (
    CheckoutDeniedResponse as CheckoutDenied,
    CheckoutDeniedResponse,
    CheckoutGrantedResponse as CheckoutGranted,
    CheckoutGrantedResponse,
    CheckoutRequest,
    CheckoutResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    ReleaseRequest,
    ReleaseResponse,
    StatusResponse,
)
from .common import (
    DEFAULT_PRODUCT,
    DEFAULT_SCHEMA_VERSION,
    PRODUCT_NAME,
    SCHEMA_VERSION,
    ClientPlatform,
    LicenseType,
    OSFamily,
    SigningAlgorithm,
)
from .license_file import (
    LicensePayload,
    LicenseRequest,
    ServerIdentity,
    SignedLicenseEnvelope,
    canonical_payload_bytes,
)
from .reason_codes import CHECKOUT_DENIAL_REASON_CODES, DenialReasonCode

LicenseEnvelope = SignedLicenseEnvelope
SignedLicense = SignedLicenseEnvelope
SignedLicensePayload = LicensePayload
SUPPORTED_ALGORITHM = SigningAlgorithm.ED25519.value

__all__ = [
    "CHECKOUT_DENIAL_REASON_CODES",
    "DEFAULT_PRODUCT",
    "DEFAULT_SCHEMA_VERSION",
    "LicenseEnvelope",
    "PRODUCT_NAME",
    "SCHEMA_VERSION",
    "SUPPORTED_ALGORITHM",
    "SignedLicense",
    "SignedLicensePayload",
    "CheckoutDenied",
    "CheckoutDeniedResponse",
    "CheckoutGranted",
    "CheckoutGrantedResponse",
    "CheckoutRequest",
    "CheckoutResponse",
    "ClientPlatform",
    "DenialReasonCode",
    "HeartbeatRequest",
    "HeartbeatResponse",
    "LicensePayload",
    "LicenseRequest",
    "LicenseType",
    "OSFamily",
    "ReleaseRequest",
    "ReleaseResponse",
    "ServerIdentity",
    "SignedLicenseEnvelope",
    "SigningAlgorithm",
    "StatusResponse",
    "canonical_payload_bytes",
]
