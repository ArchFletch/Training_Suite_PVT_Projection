"""Service-layer exceptions for the license server core."""

from __future__ import annotations


class LicenseServerError(RuntimeError):
    """Base exception for service-layer failures."""


class LicenseImportError(LicenseServerError):
    """Raised when a signed license cannot be imported."""

