"""Runtime path and configuration helpers for the license server."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import platform as runtime_platform
import tomllib

from license_server.schemas import PRODUCT_NAME


logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30
DEFAULT_LEASE_TTL_SECONDS = 120
DEFAULT_GRACE_SECONDS = 300

CONFIG_FILENAME = "config.toml"
CONFIG_PATH_ENV = "MLP_LICENSE_SERVER_CONFIG"


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


def recommended_system_log_root(os_family: str, runtime_root: Path) -> Path:
    """Return the production log root, matching what the installers create.

    On Linux the service bundle creates `/var/log/mlp-license-server`; keeping
    the resolved path in step with it avoids an empty directory that looks like
    the place to find logs but never receives any.
    """

    if os_family == "linux" and runtime_root == Path("/var/lib/mlp-license-server"):
        return Path("/var/log/mlp-license-server")
    return runtime_root / "logs"


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
        logs_dir = recommended_system_log_root(normalized_os, root_path)
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


def resolve_config_path(config_dir: Path) -> Path:
    """Return the config file the server should read for this deployment."""

    override = os.environ.get(CONFIG_PATH_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return config_dir / CONFIG_FILENAME


def _config_path_was_configured(config_path: Path) -> bool:
    """Report whether this path came from the operator rather than the default."""

    override = os.environ.get(CONFIG_PATH_ENV, "").strip()
    return bool(override) and Path(override).expanduser() == config_path


def load_lease_settings(config_path: Path) -> dict[str, int]:
    """Read the `[leases]` block from a deployment config file.

    A missing, unreadable, or malformed file yields no overrides so the server
    still starts on documented defaults rather than refusing to run. Every case
    warns first, except a file absent from the default location: an admin who
    tightens permissions on the config otherwise gets the defaults back with
    nothing in the journal, which is indistinguishable from the config having
    been applied.
    """

    try:
        with config_path.open("rb") as handle:
            document = tomllib.load(handle)
    except FileNotFoundError:
        # Absent from the default location is the ordinary local-dev case and stays
        # quiet. Absent from a path the operator named is a misconfiguration: both
        # service units set MLP_LICENSE_SERVER_CONFIG, so a file missing there was
        # moved, renamed, or restored away -- and re-running the installer will not
        # replace it, because that step is guarded on the file not existing.
        if _config_path_was_configured(config_path):
            logger.warning(
                "License server config %s is set by %s but does not exist; using built-in lease defaults",
                config_path,
                CONFIG_PATH_ENV,
            )
        return {}
    except OSError as error:
        logger.warning(
            "Cannot read license server config %s (%s); using built-in lease defaults", config_path, error
        )
        return {}
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        # UnicodeDecodeError is not a TOMLDecodeError: tomllib decodes the whole file
        # as UTF-8 before parsing it. Letting it escape would propagate out of
        # create_app() at import time, so a config an admin re-saved as UTF-16 from
        # Notepad would stop the service from starting at all -- strictly worse than
        # the silent-downgrade bug this warning exists to fix.
        logger.warning(
            "Cannot parse license server config %s (%s); using built-in lease defaults", config_path, error
        )
        return {}

    leases = document.get("leases")
    if leases is None:
        return {}
    if not isinstance(leases, dict):
        logger.warning(
            "Ignoring [leases] in license server config %s: expected a table, got %s; using built-in lease defaults",
            config_path,
            type(leases).__name__,
        )
        return {}

    supported = {
        "heartbeat_interval_seconds": "heartbeat_interval_seconds",
        "lease_ttl_seconds": "lease_ttl_seconds",
        "heartbeat_grace_seconds": "grace_seconds",
    }
    # A key the server does not recognize is the same silent downgrade as a bad
    # value, and easier to hit: the file spells the grace window
    # heartbeat_grace_seconds while the API and the docs call it grace_seconds.
    unknown_keys = sorted(set(leases) - set(supported))
    if unknown_keys:
        logger.warning(
            "Ignoring unrecognized key(s) in [leases] of license server config %s: %s; supported keys are %s",
            config_path,
            ", ".join(unknown_keys),
            ", ".join(sorted(supported)),
        )

    settings: dict[str, int] = {}
    for file_key, config_key in supported.items():
        value = leases.get(file_key)
        if value is None:
            continue
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            settings[config_key] = value
            continue
        logger.warning(
            "Ignoring leases.%s = %r in license server config %s: expected a positive integer; "
            "using the built-in default",
            file_key,
            value,
            config_path,
        )
    return settings


def build_server_config(
    *,
    runtime_root: Path | str | None = None,
    os_family: str | None = None,
    vendor_public_key_pem: str = "",
    heartbeat_interval_seconds: int | None = None,
    lease_ttl_seconds: int | None = None,
    grace_seconds: int | None = None,
    product: str = PRODUCT_NAME,
    start_path: Path | None = None,
) -> ServerConfig:
    """Build a normalized config object for the service layer and tests.

    Lease timings resolve as: explicit argument, then the `[leases]` block of
    the deployment config file, then the documented defaults.
    """

    normalized_os = detect_os_family(os_family)
    paths = resolve_server_paths(runtime_root=runtime_root, os_family=normalized_os, start_path=start_path)
    from_file = load_lease_settings(resolve_config_path(paths.config_dir))

    def resolve(name: str, explicit: int | None, fallback: int) -> int:
        if explicit is not None:
            return explicit
        return from_file.get(name, fallback)

    return ServerConfig(
        product=product,
        os_family=normalized_os,
        vendor_public_key_pem=vendor_public_key_pem,
        heartbeat_interval_seconds=resolve(
            "heartbeat_interval_seconds", heartbeat_interval_seconds, DEFAULT_HEARTBEAT_INTERVAL_SECONDS
        ),
        lease_ttl_seconds=resolve("lease_ttl_seconds", lease_ttl_seconds, DEFAULT_LEASE_TTL_SECONDS),
        grace_seconds=resolve("grace_seconds", grace_seconds, DEFAULT_GRACE_SECONDS),
        paths=paths,
    )
