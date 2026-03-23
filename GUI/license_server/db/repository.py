"""Repository helpers for the license server SQLite store."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import json
import sqlite3
from typing import Any

from license_server.core_models import AuditEvent, ImportedLicense, LeaseRecord
from license_server.schemas import LicensePayload, ServerIdentity, SignedLicense


def _normalize_datetime(value: datetime | str) -> str:
    """Store UTC datetimes in one stable text representation for SQLite."""

    if isinstance(value, datetime):
        moment = value
    else:
        raw_value = value.strip()
        if raw_value.endswith("Z"):
            raw_value = raw_value[:-1] + "+00:00"
        moment = datetime.fromisoformat(raw_value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize_jsonable(value: Any) -> Any:
    """Convert datetimes and enums to JSON-safe values before persistence."""

    if isinstance(value, datetime):
        return _normalize_datetime(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _normalize_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_normalize_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_normalize_jsonable(item) for item in value]
    return value


class LicenseServerRepository:
    """Translate between SQLite rows and the shared schema models."""

    def get_server_identity(self, connection: sqlite3.Connection) -> ServerIdentity | None:
        row = connection.execute("SELECT * FROM server_identity WHERE id = 1").fetchone()
        if row is None:
            return None
        return ServerIdentity(
            schema_version=row["schema_version"],
            product=row["product"],
            server_id=row["server_id"],
            host_fingerprint=row["host_fingerprint"],
            hostname=row["hostname"],
            os_family=row["os_family"],
            created_at=row["created_at"],
        )

    def upsert_server_identity(self, connection: sqlite3.Connection, identity: ServerIdentity) -> None:
        identity_payload = identity.to_dict()
        connection.execute(
            """
            INSERT INTO server_identity (
                id, schema_version, product, server_id, host_fingerprint, hostname, os_family, created_at
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                schema_version = excluded.schema_version,
                product = excluded.product,
                server_id = excluded.server_id,
                host_fingerprint = excluded.host_fingerprint,
                hostname = excluded.hostname,
                os_family = excluded.os_family,
                created_at = excluded.created_at
            """,
            (
                identity_payload["schema_version"],
                identity_payload["product"],
                identity_payload["server_id"],
                identity_payload["host_fingerprint"],
                identity_payload["hostname"],
                identity_payload["os_family"],
                identity_payload["created_at"],
            ),
        )

    def get_active_license(self, connection: sqlite3.Connection) -> ImportedLicense | None:
        row = connection.execute("SELECT * FROM active_license WHERE id = 1").fetchone()
        if row is None:
            return None
        payload = LicensePayload.model_validate(json.loads(row["payload_json"]))
        envelope = SignedLicense(
            algorithm=row["algorithm"],
            key_id=row["key_id"],
            payload=payload,
            signature=row["signature"],
        )
        return ImportedLicense(envelope=envelope, imported_at=row["imported_at"])

    def replace_active_license(self, connection: sqlite3.Connection, imported_license: ImportedLicense) -> None:
        payload = imported_license.payload
        payload_dict = payload.to_dict()
        connection.execute(
            """
            INSERT INTO active_license (
                id, license_id, license_type, company_name, product, server_id, host_fingerprint,
                seat_count, features_json, starts_at, ends_at, algorithm, key_id, payload_json,
                signature, imported_at
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                license_id = excluded.license_id,
                license_type = excluded.license_type,
                company_name = excluded.company_name,
                product = excluded.product,
                server_id = excluded.server_id,
                host_fingerprint = excluded.host_fingerprint,
                seat_count = excluded.seat_count,
                features_json = excluded.features_json,
                starts_at = excluded.starts_at,
                ends_at = excluded.ends_at,
                algorithm = excluded.algorithm,
                key_id = excluded.key_id,
                payload_json = excluded.payload_json,
                signature = excluded.signature,
                imported_at = excluded.imported_at
            """,
            (
                payload_dict["license_id"],
                payload_dict["license_type"],
                payload_dict["company_name"],
                payload_dict["product"],
                payload_dict["server_id"],
                payload_dict["host_fingerprint"],
                payload_dict["seat_count"],
                json.dumps(payload_dict["features"], separators=(",", ":"), sort_keys=True),
                payload_dict["starts_at"],
                payload_dict["ends_at"],
                imported_license.envelope.algorithm,
                imported_license.envelope.key_id,
                json.dumps(payload_dict, separators=(",", ":"), sort_keys=True),
                imported_license.envelope.signature,
                _normalize_datetime(imported_license.imported_at),
            ),
        )

    def get_lease(self, connection: sqlite3.Connection, lease_id: str) -> LeaseRecord | None:
        row = connection.execute("SELECT * FROM seat_leases WHERE lease_id = ?", (lease_id,)).fetchone()
        return None if row is None else self._row_to_lease(row)

    def insert_lease(self, connection: sqlite3.Connection, lease: LeaseRecord) -> None:
        connection.execute(
            """
            INSERT INTO seat_leases (
                lease_id, machine_id, hostname, username, platform, product_version,
                checked_out_at, expires_at, released_at, release_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lease.lease_id,
                lease.machine_id,
                lease.hostname,
                lease.username,
                lease.platform,
                lease.product_version,
                lease.checked_out_at,
                lease.expires_at,
                lease.released_at,
                lease.release_reason,
            ),
        )

    def update_lease_expiry(
        self,
        connection: sqlite3.Connection,
        *,
        lease_id: str,
        machine_id: str,
        expires_at: str,
    ) -> LeaseRecord | None:
        connection.execute(
            """
            UPDATE seat_leases
            SET expires_at = ?
            WHERE lease_id = ? AND machine_id = ? AND released_at IS NULL
            """,
            (expires_at, lease_id, machine_id),
        )
        return self.get_lease(connection, lease_id)

    def release_lease(
        self,
        connection: sqlite3.Connection,
        *,
        lease_id: str,
        released_at: str,
        release_reason: str,
    ) -> LeaseRecord | None:
        current = self.get_lease(connection, lease_id)
        if current is None:
            return None
        if current.released_at is not None:
            return current
        connection.execute(
            """
            UPDATE seat_leases
            SET released_at = ?, release_reason = ?
            WHERE lease_id = ? AND released_at IS NULL
            """,
            (released_at, release_reason, lease_id),
        )
        return self.get_lease(connection, lease_id)

    def release_leases(
        self,
        connection: sqlite3.Connection,
        *,
        lease_ids: list[str],
        released_at: str,
        release_reason: str,
    ) -> list[LeaseRecord]:
        if not lease_ids:
            return []
        released: list[LeaseRecord] = []
        for lease_id in lease_ids:
            updated = self.release_lease(
                connection,
                lease_id=lease_id,
                released_at=released_at,
                release_reason=release_reason,
            )
            if updated is not None:
                released.append(updated)
        return released

    def list_active_leases(self, connection: sqlite3.Connection, *, now: str, newest_first: bool = False) -> list[LeaseRecord]:
        order_by = "checked_out_at DESC, lease_id DESC" if newest_first else "checked_out_at ASC, lease_id ASC"
        rows = connection.execute(
            f"""
            SELECT * FROM seat_leases
            WHERE released_at IS NULL AND expires_at > ?
            ORDER BY {order_by}
            """,
            (now,),
        ).fetchall()
        return [self._row_to_lease(row) for row in rows]

    def count_active_leases(self, connection: sqlite3.Connection, *, now: str) -> int:
        row = connection.execute(
            """
            SELECT COUNT(*) AS active_count
            FROM seat_leases
            WHERE released_at IS NULL AND expires_at > ?
            """,
            (now,),
        ).fetchone()
        return int(row["active_count"])

    def list_expired_unreleased_leases(self, connection: sqlite3.Connection, *, now: str) -> list[LeaseRecord]:
        rows = connection.execute(
            """
            SELECT * FROM seat_leases
            WHERE released_at IS NULL AND expires_at <= ?
            ORDER BY expires_at ASC, lease_id ASC
            """,
            (now,),
        ).fetchall()
        return [self._row_to_lease(row) for row in rows]

    def add_audit_event(
        self,
        connection: sqlite3.Connection,
        *,
        event_type: str,
        event_time: str,
        details: dict[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit_events (event_type, event_time, details_json)
            VALUES (?, ?, ?)
            """,
            (
                event_type,
                _normalize_datetime(event_time),
                json.dumps(_normalize_jsonable(details), separators=(",", ":"), sort_keys=True),
            ),
        )

    def list_recent_audit_events(self, connection: sqlite3.Connection, *, limit: int = 100) -> list[AuditEvent]:
        rows = connection.execute(
            """
            SELECT event_type, event_time, details_json
            FROM audit_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            AuditEvent(
                event_type=row["event_type"],
                event_time=row["event_time"],
                details=json.loads(row["details_json"]),
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_lease(row: sqlite3.Row) -> LeaseRecord:
        return LeaseRecord(
            lease_id=row["lease_id"],
            machine_id=row["machine_id"],
            hostname=row["hostname"],
            username=row["username"],
            platform=row["platform"],
            product_version=row["product_version"],
            checked_out_at=row["checked_out_at"],
            expires_at=row["expires_at"],
            released_at=row["released_at"],
            release_reason=row["release_reason"],
        )
