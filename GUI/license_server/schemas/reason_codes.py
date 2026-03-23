"""Standard denial reason codes shared across the client API contract."""

from __future__ import annotations

from enum import Enum


class DenialReasonCode(str, Enum):
    """Checkout denial codes called out in the MVP contract."""

    ALL_SEATS_IN_USE = "all_seats_in_use"
    LICENSE_EXPIRED = "license_expired"
    LICENSE_NOT_STARTED = "license_not_started"
    FEATURE_NOT_ENABLED = "feature_not_enabled"
    SERVER_BINDING_MISMATCH = "server_binding_mismatch"
    INVALID_LICENSE = "invalid_license"


CHECKOUT_DENIAL_REASON_CODES = tuple(reason.value for reason in DenialReasonCode)

