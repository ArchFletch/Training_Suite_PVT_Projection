"""Identity generation and license-request export helpers."""

from __future__ import annotations

import json
from pathlib import Path

from license_server.crypto import build_host_fingerprint, detect_hostname, generate_server_id, read_machine_token
from license_server.db import LicenseServerRepository, SQLiteSessionFactory
from license_server.schemas import LicenseRequest, ServerIdentity
from license_server.service import ServerConfig, format_utc_datetime, utc_now


class IdentityService:
    """Manage the persistent server identity and request exports."""

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

    def ensure_server_identity(self) -> ServerIdentity:
        """Load the existing server identity or create and persist one on first use."""

        with self.session_factory.session(write_lock=True) as connection:
            existing = self.repository.get_server_identity(connection)
            if existing is not None:
                return existing

            hostname = detect_hostname()
            machine_token = read_machine_token(self.config.os_family)
            created_at = format_utc_datetime(utc_now())
            identity = ServerIdentity.create(
                server_id=generate_server_id(),
                host_fingerprint=build_host_fingerprint(
                    hostname=hostname,
                    os_family=self.config.os_family,
                    machine_token=machine_token,
                ),
                hostname=hostname,
                os_family=self.config.os_family,
                created_at=created_at,
                product=self.config.product,
            )
            self.repository.upsert_server_identity(connection, identity)
            self.repository.add_audit_event(
                connection,
                event_type="server_initialized",
                event_time=created_at,
                details={
                    "server_id": identity.server_id,
                    "hostname": identity.hostname,
                    "os_family": identity.os_family,
                },
            )

        self._write_json(self.config.paths.identity_path, identity.to_dict())
        return identity

    def export_license_request(
        self,
        *,
        requested_by: str | None = None,
        output_path: Path | str | None = None,
    ) -> LicenseRequest:
        """Create and persist a vendor-facing `license_request.json` export."""

        identity = self.ensure_server_identity()
        request = LicenseRequest.from_identity(
            identity,
            generated_at=format_utc_datetime(utc_now()),
            requested_by=requested_by,
        )
        target_path = Path(output_path) if output_path is not None else self.config.paths.request_path
        self._write_json(target_path, request.to_dict())
        with self.session_factory.session(write_lock=True) as connection:
            self.repository.add_audit_event(
                connection,
                event_type="request_exported",
                event_time=request.generated_at,
                details={
                    "server_id": request.server_id,
                    "requested_by": request.requested_by,
                    "output_path": str(target_path),
                },
            )
        return request

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
