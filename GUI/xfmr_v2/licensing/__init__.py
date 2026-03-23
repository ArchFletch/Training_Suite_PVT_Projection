"""Desktop-side licensing client code for the GUI application."""

from .client import (
    LicenseClientError,
    LicenseConnectionError,
    LicenseHttpClient,
    LicenseProtocolError,
    test_license_connection,
)
from .controller import LicenseLeaseController
from .models import (
    DEFAULT_PRODUCT_VERSION,
    PRODUCT_NAME,
    LicenseCheckoutResult,
    LicenseHeartbeatResult,
    LicenseIdentity,
    LicenseLeaseState,
    LicenseReleaseResult,
    LicenseStatus,
    format_utc_timestamp,
    normalize_server_url,
)

__all__ = [
    "DEFAULT_PRODUCT_VERSION",
    "PRODUCT_NAME",
    "LicenseCheckoutResult",
    "LicenseClientError",
    "LicenseConnectionError",
    "LicenseHeartbeatResult",
    "LicenseHttpClient",
    "LicenseIdentity",
    "LicenseLeaseController",
    "LicenseLeaseState",
    "LicenseProtocolError",
    "LicenseReleaseResult",
    "LicenseStatus",
    "format_utc_timestamp",
    "normalize_server_url",
    "test_license_connection",
]
