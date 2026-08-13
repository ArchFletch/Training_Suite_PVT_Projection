"""License import, verification, and active-license status helpers."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from license_server.core_models import CheckoutDenied, ImportedLicense, LicenseImportResult, StatusResponse
from license_server.crypto import LicenseVerificationError, verify_signed_license
from license_server.db import LicenseServerRepository, SQLiteSessionFactory
from license_server.schemas import SignedLicense
from license_server.service import ServerConfig, coerce_utc_datetime, format_utc_datetime, utc_now

from .exceptions import LicenseImportError


LICENSE_STATE_ACTIVE = "active"
LICENSE_STATE_EXPIRED = "expired"
LICENSE_STATE_NOT_STARTED = "not_started"
LICENSE_STATE_NONE = "none"


def resolve_license_state(payload, *, now: str) -> str:
    """Classify an imported license against the current time.

    Term validity is deliberately separate from signature validity: a license
    can verify perfectly and still be outside its window.
    """

    current_time = coerce_utc_datetime(now)
    if current_time < coerce_utc_datetime(payload.starts_at):
        return LICENSE_STATE_NOT_STARTED
    if current_time > coerce_utc_datetime(payload.ends_at):
        return LICENSE_STATE_EXPIRED
    return LICENSE_STATE_ACTIVE


class LicenseService:
    """Manage active-license persistence, verification, and status rules."""

    def __init__(
        self,
        *,
        config: ServerConfig,
        session_factory: SQLiteSessionFactory,
        repository: LicenseServerRepository,
    ) -> None:
        self.config = config
        self.session_factory = session_factory
        self.repository = repository

    def import_license(self, payload: Mapping[str, Any] | SignedLicense | str | bytes) -> LicenseImportResult:
        """Verify, store, and activate a signed license for the current server."""

        envelope = self._coerce_envelope(payload)
        if not self.config.vendor_public_key_pem.strip():
            raise LicenseImportError("A vendor public key is required before importing a license.")

        try:
            verify_signed_license(envelope, public_key_pem=self.config.vendor_public_key_pem)
        except LicenseVerificationError as exc:
            raise LicenseImportError(str(exc)) from exc

        imported_at = format_utc_datetime(utc_now())
        imported_license = ImportedLicense(envelope=envelope, imported_at=imported_at)

        with self.session_factory.session(write_lock=True) as connection:
            identity = self.repository.get_server_identity(connection)
            if identity is None:
                raise LicenseImportError("Server identity must exist before importing a license.")

            license_payload = imported_license.payload
            if license_payload.product != self.config.product:
                raise LicenseImportError("License product does not match this server.")
            if license_payload.server_id != identity.server_id:
                raise LicenseImportError("License server_id does not match this server identity.")
            if license_payload.host_fingerprint != identity.host_fingerprint:
                raise LicenseImportError("License host_fingerprint does not match this server identity.")

            self.repository.replace_active_license(connection, imported_license)

            active_leases = self.repository.list_active_leases(
                connection,
                now=imported_at,
                newest_first=True,
            )
            excess_count = max(0, len(active_leases) - license_payload.seat_count)
            evicted_ids = [lease.lease_id for lease in active_leases[:excess_count]]
            if evicted_ids:
                self.repository.release_leases(
                    connection,
                    lease_ids=evicted_ids,
                    released_at=imported_at,
                    release_reason="license_downgrade",
                )
                for lease_id in evicted_ids:
                    self.repository.add_audit_event(
                        connection,
                        event_type="lease_evicted",
                        event_time=imported_at,
                        details={
                            "lease_id": lease_id,
                            "reason": "license_downgrade",
                            "license_id": license_payload.license_id,
                        },
                    )

            license_state = resolve_license_state(license_payload, now=imported_at)

            self.repository.add_audit_event(
                connection,
                event_type="license_imported",
                event_time=imported_at,
                details={
                    "license_id": license_payload.license_id,
                    "license_type": license_payload.license_type,
                    "seat_count": license_payload.seat_count,
                    "evicted_lease_ids": evicted_ids,
                    "license_state": license_state,
                },
            )

        self._write_json(self.config.paths.active_license_path, imported_license.envelope.to_dict())

        # A license can verify perfectly and still be outside its term. Saying so
        # here is the difference between an admin fixing it now and every user
        # being blocked at launch with no explanation.
        warning = None
        if license_state == LICENSE_STATE_EXPIRED:
            warning = (
                f"This license ended at {license_payload.ends_at} and is already expired. "
                "It is now the active license, but every seat request will be denied "
                "until a license covering the current date is imported."
            )
        elif license_state == LICENSE_STATE_NOT_STARTED:
            warning = (
                f"This license does not start until {license_payload.starts_at}. "
                "Seat requests will be denied until then."
            )

        return LicenseImportResult(
            ok=True,
            license_id=imported_license.payload.license_id,
            imported_at=imported_at,
            evicted_lease_ids=tuple(evicted_ids),
            message="License imported successfully.",
            license_state=license_state,
            warning=warning,
        )

    def get_active_license(self, connection: sqlite3.Connection | None = None) -> ImportedLicense | None:
        """Return the imported license, if one is active in the local store."""

        if connection is not None:
            return self.repository.get_active_license(connection)
        with self.session_factory.session() as connection:
            return self.repository.get_active_license(connection)

    def validate_checkout(
        self,
        *,
        now: str,
        product: str,
        requested_feature: str | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> ImportedLicense | CheckoutDenied:
        """Validate product, term dates, and optional feature access for a checkout."""

        if connection is None:
            with self.session_factory.session() as session:
                return self.validate_checkout(
                    now=now,
                    product=product,
                    requested_feature=requested_feature,
                    connection=session,
                )

        imported_license = self.repository.get_active_license(connection)
        if imported_license is None:
            return CheckoutDenied(
                granted=False,
                reason_code="invalid_license",
                message="No active license has been imported.",
            )

        payload = imported_license.payload
        current_time = coerce_utc_datetime(now)
        starts_at = coerce_utc_datetime(payload.starts_at)
        ends_at = coerce_utc_datetime(payload.ends_at)

        if product != self.config.product or payload.product != self.config.product:
            return CheckoutDenied(
                granted=False,
                reason_code="invalid_license",
                message="The requested product does not match the imported license.",
            )
        if current_time < starts_at:
            return CheckoutDenied(
                granted=False,
                reason_code="license_not_started",
                message="The imported license term has not started yet.",
            )
        if current_time > ends_at:
            return CheckoutDenied(
                granted=False,
                reason_code="license_expired",
                message="The imported license term has already ended.",
            )
        if requested_feature is not None and requested_feature not in payload.features:
            return CheckoutDenied(
                granted=False,
                reason_code="feature_not_enabled",
                message=f"The feature '{requested_feature}' is not enabled in this license.",
            )
        return imported_license

    def get_status(self, *, now: str | None = None) -> StatusResponse:
        """Return a client-readable status summary for API and CLI wiring."""

        current_time = format_utc_datetime(now or utc_now())
        with self.session_factory.session() as connection:
            imported_license = self.repository.get_active_license(connection)
            seats_in_use = self.repository.count_active_leases(connection, now=current_time)

        inactive = StatusResponse(
            ok=False,
            product=self.config.product,
            company_name=None,
            license_type=None,
            starts_at=None,
            ends_at=None,
            seat_count=0,
            seats_in_use=0,
        )
        if imported_license is None:
            return inactive

        # `ok` means "this server can grant a seat right now". An imported but
        # out-of-term license cannot, so reporting it as ok would tell the GUI
        # connection test that everything is fine while every checkout fails.
        # The wire contract requires the license fields be omitted when not ok;
        # the admin CLI still surfaces the full term and the reason.
        if resolve_license_state(imported_license.payload, now=current_time) != LICENSE_STATE_ACTIVE:
            return inactive

        payload = imported_license.payload
        return StatusResponse(
            ok=True,
            product=payload.product,
            company_name=payload.company_name,
            license_type=payload.license_type,
            starts_at=payload.starts_at,
            ends_at=payload.ends_at,
            seat_count=payload.seat_count,
            seats_in_use=seats_in_use,
        )

    @staticmethod
    def _coerce_envelope(payload: Mapping[str, Any] | SignedLicense | str | bytes) -> SignedLicense:
        if isinstance(payload, SignedLicense):
            return payload
        if isinstance(payload, bytes):
            data = json.loads(payload.decode("utf-8"))
            return SignedLicense.model_validate(data)
        if isinstance(payload, str):
            path = Path(payload)
            text = path.read_text(encoding="utf-8") if path.exists() else payload
            return SignedLicense.model_validate(json.loads(text))
        return SignedLicense.model_validate(payload)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
