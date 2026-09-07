"""Host identity helpers for the customer-deployed license server."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import socket
import uuid


def detect_hostname() -> str:
    """Return the current hostname in the form used by the request export.

    Case-folded on purpose. The same host can report either casing depending on
    which resolution path answers -- ``PC`` from one, ``pc`` from another -- and
    the two hash to different fingerprints, so an unfolded name made the identity
    depend on something that is not actually a property of the host.
    """

    hostname = socket.gethostname().strip() or platform.node().strip() or "unknown-host"
    return hostname.lower()


def read_machine_token(os_family: str | None = None) -> str:
    """Read a stable machine identifier for Windows or Linux if available."""

    normalized_os = (os_family or platform.system()).strip().lower()
    if normalized_os.startswith("win"):
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as registry_key:
                machine_guid, _ = winreg.QueryValueEx(registry_key, "MachineGuid")
            if isinstance(machine_guid, str) and machine_guid.strip():
                return machine_guid.strip()
        except (OSError, ImportError):
            # ImportError: winreg does not exist off-Windows; fall through to the
            # Linux machine-id files / MAC fallback instead of crashing.
            pass

    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        try:
            machine_id = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if machine_id:
            return machine_id

    return f"node-{uuid.getnode():012x}"


def generate_server_id() -> str:
    """Generate a persistent server ID that is stored once on first start."""

    return f"srv_{os.urandom(6).hex()}"


def build_host_fingerprint(*, hostname: str, os_family: str, machine_token: str | None = None) -> str:
    """Derive a deterministic fingerprint from stable host properties.

    Hostname and OS family are folded here as well as in ``detect_hostname``, so
    the fingerprint is stable no matter which caller supplies them: a stored
    identity, a re-init after a rebuild, and an ad-hoc check all have to agree or
    a valid license stops importing.
    """

    token = machine_token or read_machine_token(os_family)
    normalized_hostname = hostname.strip().lower()
    normalized_os_family = os_family.strip().lower()
    digest = hashlib.sha256(
        f"{normalized_hostname}|{normalized_os_family}|{token}".encode("utf-8")
    ).hexdigest()
    return f"host_{digest[:12]}"
