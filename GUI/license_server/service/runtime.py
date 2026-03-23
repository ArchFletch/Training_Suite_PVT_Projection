"""Bootstrap helpers that assemble the license server core services."""

from __future__ import annotations

from dataclasses import dataclass

from license_server.db import LicenseServerRepository, SQLiteSessionFactory
from license_server.service.config import ServerConfig, build_server_config, ensure_runtime_directories


@dataclass
class LicenseServerRuntime:
    """In-memory bundle of the initialized server-core collaborators."""

    config: ServerConfig
    session_factory: SQLiteSessionFactory
    repository: LicenseServerRepository
    identity_service: object
    license_service: object
    lease_service: object


def create_runtime(
    *,
    config: ServerConfig | None = None,
    runtime_root: str | None = None,
    os_family: str | None = None,
    vendor_public_key_pem: str = "",
) -> LicenseServerRuntime:
    """Create the runtime bundle consumed by future API and CLI entrypoints."""

    from license_server.services import IdentityService, LeaseService, LicenseService

    resolved_config = config or build_server_config(
        runtime_root=runtime_root,
        os_family=os_family,
        vendor_public_key_pem=vendor_public_key_pem,
    )
    ensure_runtime_directories(resolved_config.paths)

    session_factory = SQLiteSessionFactory(resolved_config.paths.database_path)
    session_factory.initialize_database()

    repository = LicenseServerRepository()
    identity_service = IdentityService(config=resolved_config, session_factory=session_factory, repository=repository)
    license_service = LicenseService(config=resolved_config, session_factory=session_factory, repository=repository)
    lease_service = LeaseService(
        config=resolved_config,
        session_factory=session_factory,
        repository=repository,
        license_service=license_service,
    )

    return LicenseServerRuntime(
        config=resolved_config,
        session_factory=session_factory,
        repository=repository,
        identity_service=identity_service,
        license_service=license_service,
        lease_service=lease_service,
    )
