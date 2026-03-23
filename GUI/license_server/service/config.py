"""Runtime path and configuration helpers for the license server."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform as runtime_platform

from license_server.schemas import PRODUCT_NAME


DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30
DEFAULT_LEASE_TTL_SECONDS = 120
DEFAULT_GRACE_SECONDS = 300


@dataclass(frozen=True)
class ServerPaths:
    """Resolved runtime locations for one server deployment."""

    runtime_root: Path
    config_dir: Path
    data_dir: Path
    logs_dir: Path
    identity_path: Path
    request_path: Path
    active_license_path: Path
    database_path: Path


@dataclass(frozen=True)
class ServerConfig:
    """Runtime settings for the core service layer."""

    product: str
    os_family: str
    vendor_public_key_pem: str
    heartbeat_interval_seconds: int
    lease_ttl_seconds: int
    grace_seconds: int
    paths: ServerPaths


def detect_os_family(platform_name: str | None = None) -> str:
    """Normalize Windows and Linux platform names to the shared schema values."""

    raw_name = (platform_name or runtime_platform.system() or os.name).strip().lower()
    if raw_name.startswith("win"):
        return "windows"
    if raw_name.startswith("linux"):
        return "linux"
    return raw_name


def resolve_repo_root(start_path: Path | None = None) -> Path | None:
    """Find the repository root by walking upward until a `.git` marker appears."""

    current_path = (start_path or Path.cwd()).resolve()
    for candidate in (current_path, *current_path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def recommended_system_runtime_root(os_family: str) -> Path:
    """Return the production runtime root for the current operating system."""

    if os_family == "windows":
        program_data = Path(os.environ.get("PROGRAMDATA", "C:/ProgramData"))
        return program_data / "MLP License Server"
    return Path("/var/lib/mlp-license-server")


def recommended_system_config_root(os_family: str, runtime_root: Path) -> Path:
    """Return the production config root for the current operating system."""

    if os_family == "linux" and runtime_root == Path("/var/lib/mlp-license-server"):
        return Path("/etc/mlp-license-server")
    return runtime_root


def resolve_server_paths(
    *,
    runtime_root: Path | str | None = None,
    os_family: str | None = None,
    start_path: Path | None = None,
) -> ServerPaths:
    """Resolve config, data, and log paths for production or local development."""

    normalized_os = detect_os_family(os_family)
    root_override = runtime_root or os.environ.get("MLP_LICENSE_SERVER_RUNTIME_DIR")
    if root_override is None:
        repo_root = resolve_repo_root(start_path)
        if repo_root is not None:
            root_path = (repo_root / "runtime" / "license_server").resolve()
        else:
            root_path = recommended_system_runtime_root(normalized_os)
    else:
        root_path = Path(root_override).expanduser()

    system_runtime_root = recommended_system_runtime_root(normalized_os)
    root_text = str(root_path).replace("\\", "/").rstrip("/")
    system_root_text = str(system_runtime_root).replace("\\", "/").rstrip("/")

    if root_text == system_root_text:
        config_dir = recommended_system_config_root(normalized_os, root_path)
        data_dir = root_path
        logs_dir = root_path / "logs"
    else:
        config_dir = root_path / "config"
        data_dir = root_path / "data"
        logs_dir = root_path / "logs"

    return ServerPaths(
        runtime_root=root_path,
        config_dir=config_dir,
        data_dir=data_dir,
        logs_dir=logs_dir,
        identity_path=data_dir / "server_identity.json",
        request_path=data_dir / "license_request.json",
        active_license_path=data_dir / "active_license.json",
        database_path=data_dir / "license.db",
    )


def ensure_runtime_directories(paths: ServerPaths) -> None:
    """Create the runtime directories needed by the core services."""

    paths.runtime_root.mkdir(parents=True, exist_ok=True)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)


def build_server_config(
    *,
    runtime_root: Path | str | None = None,
    os_family: str | None = None,
    vendor_public_key_pem: str = "",
    heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    lease_ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
    grace_seconds: int = DEFAULT_GRACE_SECONDS,
    product: str = PRODUCT_NAME,
    start_path: Path | None = None,
) -> ServerConfig:
    """Build a normalized config object for the service layer and tests."""

    normalized_os = detect_os_family(os_family)
    paths = resolve_server_paths(runtime_root=runtime_root, os_family=normalized_os, start_path=start_path)
    return ServerConfig(
        product=product,
        os_family=normalized_os,
        vendor_public_key_pem=vendor_public_key_pem,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        lease_ttl_seconds=lease_ttl_seconds,
        grace_seconds=grace_seconds,
        paths=paths,
    )
