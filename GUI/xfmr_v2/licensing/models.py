"""Shared models for the desktop-side floating-license client."""

from __future__ import annotations

import getpass
import hashlib
import socket
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


PRODUCT_NAME = "Surrogate Model Traning Suite"
DEFAULT_PRODUCT_VERSION = "desktop"


def normalize_server_url(server_url: str) -> str:
    """Normalize a configured server URL into a stable form."""

    return server_url.strip().rstrip("/")


def parse_utc_timestamp(value: str | None) -> datetime | None:
    """Parse one ISO-8601 timestamp into an aware UTC datetime."""

    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_utc_timestamp(value: datetime | None) -> str | None:
    """Serialize one datetime into the wire format used by the license server."""

    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def default_machine_id() -> str:
    """Build a deterministic machine identifier suitable for the MVP server API."""

    host = socket.gethostname().strip().lower() or "unknown-host"
    seed = f"{host}:{uuid.getnode():012x}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    return f"gui_{digest}"


def current_platform_name() -> str:
    """Return the small platform label expected by the MVP checkout request."""

    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


@dataclass(frozen=True)
class LicenseStatus:
    """Connection-test summary returned by ``GET /status``."""

    ok: bool
    product: str | None = None
    company_name: str | None = None
    license_type: str | None = None
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    seat_count: int | None = None
    seats_in_use: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LicenseStatus":
        return cls(
            ok=bool(payload.get("ok", False)),
            product=_optional_str(payload.get("product")),
            company_name=_optional_str(payload.get("company_name")),
            license_type=_optional_str(payload.get("license_type")),
            starts_at=parse_utc_timestamp(_optional_str(payload.get("starts_at"))),
            ends_at=parse_utc_timestamp(_optional_str(payload.get("ends_at"))),
            seat_count=_optional_int(payload.get("seat_count")),
            seats_in_use=_optional_int(payload.get("seats_in_use")),
            raw=dict(payload),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "product": self.product,
            "company_name": self.company_name,
            "license_type": self.license_type,
            "starts_at": format_utc_timestamp(self.starts_at),
            "ends_at": format_utc_timestamp(self.ends_at),
            "seat_count": self.seat_count,
            "seats_in_use": self.seats_in_use,
        }

    @property
    def seat_summary(self) -> str:
        if self.seat_count is None or self.seats_in_use is None:
            return "Seat usage unavailable"
        return f"{self.seats_in_use}/{self.seat_count} seats in use"

    @property
    def window_summary(self) -> str:
        if self.starts_at is None and self.ends_at is None:
            return "License window unavailable"
        if self.starts_at is None:
            return f"Ends {format_utc_timestamp(self.ends_at)}"
        if self.ends_at is None:
            return f"Starts {format_utc_timestamp(self.starts_at)}"
        return f"{format_utc_timestamp(self.starts_at)} to {format_utc_timestamp(self.ends_at)}"


@dataclass(frozen=True)
class LicenseCheckoutResult:
    """Checkout response returned by ``POST /checkout``."""

    granted: bool
    message: str = ""
    reason_code: str | None = None
    lease_id: str | None = None
    heartbeat_interval_seconds: float | None = None
    lease_ttl_seconds: float | None = None
    grace_seconds: float | None = None
    expires_at: datetime | None = None
    license_type: str | None = None
    company_name: str | None = None
    seat_count: int | None = None
    seats_in_use: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LicenseCheckoutResult":
        return cls(
            granted=bool(payload.get("granted", False)),
            message=_optional_str(payload.get("message")) or "",
            reason_code=_optional_str(payload.get("reason_code")),
            lease_id=_optional_str(payload.get("lease_id")),
            heartbeat_interval_seconds=_optional_float(payload.get("heartbeat_interval_seconds")),
            lease_ttl_seconds=_optional_float(payload.get("lease_ttl_seconds")),
            grace_seconds=_optional_float(payload.get("grace_seconds")),
            expires_at=parse_utc_timestamp(_optional_str(payload.get("expires_at"))),
            license_type=_optional_str(payload.get("license_type")),
            company_name=_optional_str(payload.get("company_name")),
            seat_count=_optional_int(payload.get("seat_count")),
            seats_in_use=_optional_int(payload.get("seats_in_use")),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class LicenseHeartbeatResult:
    """Heartbeat response returned by ``POST /heartbeat``."""

    ok: bool
    expires_at: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LicenseHeartbeatResult":
        return cls(
            ok=bool(payload.get("ok", False)),
            expires_at=parse_utc_timestamp(_optional_str(payload.get("expires_at"))),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class LicenseReleaseResult:
    """Release response returned by ``POST /release``."""

    ok: bool
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LicenseReleaseResult":
        return cls(ok=bool(payload.get("ok", False)), raw=dict(payload))


@dataclass(frozen=True)
class LicenseIdentity:
    """Stable client identity fields sent during seat checkout."""

    product: str = PRODUCT_NAME
    product_version: str = DEFAULT_PRODUCT_VERSION
    machine_id: str = field(default_factory=default_machine_id)
    hostname: str = field(default_factory=socket.gethostname)
    username: str = field(default_factory=getpass.getuser)
    platform: str = field(default_factory=current_platform_name)

    def to_checkout_payload(self) -> dict[str, str]:
        return {
            "product": self.product,
            "product_version": self.product_version,
            "machine_id": self.machine_id,
            "hostname": self.hostname,
            "username": self.username,
            "platform": self.platform,
        }


@dataclass(frozen=True)
class LicenseLeaseState:
    """GUI-friendly view of the current lease lifecycle."""

    phase: str = "unconfigured"
    badge_text: str = "Unconfigured"
    server_url: str = ""
    message: str = "Enter a license server URL to enable floating-seat checkout."
    last_status: LicenseStatus | None = None
    lease_id: str | None = None
    machine_id: str | None = None
    expires_at: datetime | None = None
    heartbeat_interval_seconds: float | None = None
    lease_ttl_seconds: float | None = None
    grace_seconds: float | None = None
    grace_deadline: datetime | None = None
    last_heartbeat_at: datetime | None = None
    heartbeat_failures: int = 0
    last_error: str | None = None

    @property
    def licensing_enabled(self) -> bool:
        return bool(self.server_url)

    @property
    def has_active_lease(self) -> bool:
        return bool(self.lease_id) and self.phase in {"checked_out", "heartbeat_warning"}

    @property
    def can_start_runs(self) -> bool:
        return (not self.licensing_enabled) or self.phase == "checked_out"

    @property
    def seat_summary(self) -> str:
        if self.last_status is None:
            return "Seat usage unavailable"
        return self.last_status.seat_summary

    @property
    def window_summary(self) -> str:
        if self.last_status is None:
            return "License window unavailable"
        return self.last_status.window_summary


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
