"""Internal runtime models for the license-server core layer.

These objects are intentionally separate from ``license_server.schemas``.
The shared schema package owns wire-format request/response and signed-file
contracts, while this module owns runtime-only records and service results.
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any

from license_server.schemas import SignedLicense


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        normalized = value.astimezone(timezone.utc) if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return normalized.isoformat().replace("+00:00", "Z")
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class JsonRecord:
    """Small serialization helper shared by runtime records."""

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class AuditEvent(JsonRecord):
    """Append-only audit entry returned from the SQLite repository."""

    event_type: str
    event_time: str
    details: dict[str, Any]


@dataclass(frozen=True)
class LeaseRecord(JsonRecord):
    """In-memory representation of one floating-seat lease row."""

    lease_id: str
    machine_id: str
    hostname: str
    username: str | None
    platform: str
    product_version: str
    checked_out_at: str
    expires_at: str
    released_at: str | None = None
    release_reason: str | None = None

    @property
    def is_released(self) -> bool:
        return self.released_at is not None


@dataclass(frozen=True)
class ImportedLicense(JsonRecord):
    """Active license envelope plus the local import timestamp."""

    envelope: SignedLicense
    imported_at: str

    @property
    def payload(self):
        return self.envelope.payload


@dataclass(frozen=True)
class LicenseImportResult(JsonRecord):
    """Structured outcome returned by the license-import path."""

    ok: bool
    license_id: str | None = None
    imported_at: str | None = None
    evicted_lease_ids: tuple[str, ...] = ()
    message: str | None = None
    license_state: str | None = None
    warning: str | None = None


@dataclass(frozen=True)
class CheckoutGranted(JsonRecord):
    """Successful checkout result returned by the lease service."""

    granted: bool
    lease_id: str
    heartbeat_interval_seconds: int
    lease_ttl_seconds: int
    grace_seconds: int
    expires_at: str
    license_type: str
    company_name: str
    seat_count: int
    seats_in_use: int


@dataclass(frozen=True)
class CheckoutDenied(JsonRecord):
    """Denied checkout result returned by the lease service."""

    granted: bool
    reason_code: str
    message: str


@dataclass(frozen=True)
class HeartbeatResponse(JsonRecord):
    """Heartbeat outcome returned by the lease service."""

    ok: bool
    expires_at: str | None = None
    reason_code: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class ReleaseResponse(JsonRecord):
    """Release outcome returned by the lease service."""

    ok: bool
    reason_code: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class StatusResponse(JsonRecord):
    """Client-readable license summary returned by the core layer."""

    ok: bool
    product: str
    company_name: str | None
    license_type: str | None
    starts_at: str | None
    ends_at: str | None
    seat_count: int
    seats_in_use: int
