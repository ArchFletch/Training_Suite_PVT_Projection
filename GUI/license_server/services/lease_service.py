"""Floating-seat checkout, heartbeat, release, cleanup, and eviction rules."""

from __future__ import annotations

from datetime import timedelta
import os

from license_server.core_models import (
    CheckoutDenied,
    CheckoutGranted,
    HeartbeatResponse,
    LeaseRecord,
    ReleaseResponse,
)
from license_server.db import LicenseServerRepository, SQLiteSessionFactory
from license_server.schemas import (
    CheckoutRequest,
    HeartbeatRequest,
    ReleaseRequest,
)
from license_server.service import ServerConfig, coerce_utc_datetime, format_utc_datetime, utc_now

from .license_service import LicenseService


class LeaseService:
    """Manage floating-seat lifecycle rules on top of the active license."""

    def __init__(
        self,
        *,
        config: ServerConfig,
        session_factory: SQLiteSessionFactory,
        repository: LicenseServerRepository,
        license_service: LicenseService,
    ) -> None:
        self.config = config
        self.session_factory = session_factory
        self.repository = repository
        self.license_service = license_service

    def checkout(self, request: CheckoutRequest) -> CheckoutGranted | CheckoutDenied:
        """Attempt to allocate one floating seat to the requesting client."""

        now = format_utc_datetime(utc_now())

        with self.session_factory.session(write_lock=True) as connection:
            timed_out_leases = self._cleanup_expired_with_connection(connection, now=now)
            license_or_denial = self.license_service.validate_checkout(
                now=now,
                product=request.product,
                requested_feature=request.requested_feature,
                connection=connection,
            )
            if isinstance(license_or_denial, CheckoutDenied):
                return license_or_denial

            imported_license = license_or_denial
            active_count = self.repository.count_active_leases(connection, now=now)
            if active_count >= imported_license.payload.seat_count:
                self.repository.add_audit_event(
                    connection,
                    event_type="checkout_denied",
                    event_time=now,
                    details={
                        "reason_code": "all_seats_in_use",
                        "machine_id": request.machine_id,
                        "hostname": request.hostname,
                    },
                )
                return CheckoutDenied(
                    granted=False,
                    reason_code="all_seats_in_use",
                    message="All floating seats are currently in use.",
                )

            expires_at = format_utc_datetime(coerce_utc_datetime(now) + self._lease_ttl_delta)
            lease = LeaseRecord(
                lease_id=f"lease_{os.urandom(6).hex()}",
                machine_id=request.machine_id,
                hostname=request.hostname,
                username=request.username,
                platform=request.platform,
                product_version=request.product_version,
                checked_out_at=now,
                expires_at=expires_at,
            )
            self.repository.insert_lease(connection, lease)
            seats_in_use = self.repository.count_active_leases(connection, now=now)
            self.repository.add_audit_event(
                connection,
                event_type="checkout_granted",
                event_time=now,
                details={
                    "lease_id": lease.lease_id,
                    "license_id": imported_license.payload.license_id,
                    "machine_id": request.machine_id,
                    "hostname": request.hostname,
                    "timed_out_lease_count": len(timed_out_leases),
                },
            )

        return CheckoutGranted(
            granted=True,
            lease_id=lease.lease_id,
            heartbeat_interval_seconds=self.config.heartbeat_interval_seconds,
            lease_ttl_seconds=self.config.lease_ttl_seconds,
            grace_seconds=self.config.grace_seconds,
            expires_at=expires_at,
            license_type=imported_license.payload.license_type,
            company_name=imported_license.payload.company_name,
            seat_count=imported_license.payload.seat_count,
            seats_in_use=seats_in_use,
        )

    def heartbeat(self, request: HeartbeatRequest) -> HeartbeatResponse:
        """Extend an existing active lease if it is still valid."""

        now = format_utc_datetime(utc_now())
        with self.session_factory.session(write_lock=True) as connection:
            self._cleanup_expired_with_connection(connection, now=now)
            current = self.repository.get_lease(connection, request.lease_id)
            if current is None or current.released_at is not None or current.expires_at <= now:
                return HeartbeatResponse(
                    ok=False,
                    reason_code="invalid_lease",
                    message="The lease is not active.",
                )
            if current.machine_id != request.machine_id:
                return HeartbeatResponse(
                    ok=False,
                    reason_code="machine_mismatch",
                    message="The lease is owned by a different machine_id.",
                )
            expires_at = format_utc_datetime(coerce_utc_datetime(now) + self._lease_ttl_delta)
            updated = self.repository.update_lease_expiry(
                connection,
                lease_id=request.lease_id,
                machine_id=request.machine_id,
                expires_at=expires_at,
            )
            if updated is None or updated.released_at is not None:
                return HeartbeatResponse(
                    ok=False,
                    reason_code="invalid_lease",
                    message="The lease is no longer active.",
                )
            self.repository.add_audit_event(
                connection,
                event_type="heartbeat_renewed",
                event_time=now,
                details={
                    "lease_id": request.lease_id,
                    "machine_id": request.machine_id,
                    "expires_at": expires_at,
                },
            )
        return HeartbeatResponse(ok=True, expires_at=expires_at)

    def release(self, request: ReleaseRequest) -> ReleaseResponse:
        """Release an active lease on clean client shutdown."""

        now = format_utc_datetime(utc_now())
        with self.session_factory.session(write_lock=True) as connection:
            existing = self.repository.get_lease(connection, request.lease_id)
            if existing is None:
                return ReleaseResponse(ok=False, reason_code="invalid_lease", message="Lease not found.")
            updated = self.repository.release_lease(
                connection,
                lease_id=request.lease_id,
                released_at=now,
                release_reason="client_release",
            )
            if existing.released_at is None and updated is not None and updated.released_at == now:
                self.repository.add_audit_event(
                    connection,
                    event_type="lease_released",
                    event_time=now,
                    details={"lease_id": request.lease_id, "reason": "client_release"},
                )
        return ReleaseResponse(ok=True, message="Lease released.")

    def cleanup_expired_leases(self, *, now: str | None = None) -> list[LeaseRecord]:
        """Mark expired leases as timed out and return the released records."""

        current_time = format_utc_datetime(now or utc_now())
        with self.session_factory.session(write_lock=True) as connection:
            return self._cleanup_expired_with_connection(connection, now=current_time)

    def list_active_leases(self, *, now: str | None = None) -> list[LeaseRecord]:
        """Return the current active lease rows for API/CLI wiring."""

        current_time = format_utc_datetime(now or utc_now())
        with self.session_factory.session() as connection:
            return self.repository.list_active_leases(connection, now=current_time)

    def _cleanup_expired_with_connection(self, connection, *, now: str) -> list[LeaseRecord]:
        expired = self.repository.list_expired_unreleased_leases(connection, now=now)
        if not expired:
            return []
        lease_ids = [lease.lease_id for lease in expired]
        timed_out = self.repository.release_leases(
            connection,
            lease_ids=lease_ids,
            released_at=now,
            release_reason="heartbeat_timeout",
        )
        for lease in timed_out:
            self.repository.add_audit_event(
                connection,
                event_type="lease_timed_out",
                event_time=now,
                details={"lease_id": lease.lease_id, "machine_id": lease.machine_id},
            )
        return timed_out

    @property
    def _lease_ttl_delta(self) -> timedelta:
        return timedelta(seconds=self.config.lease_ttl_seconds)
