"""Host identity helpers for the customer-deployed license server."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import socket
import uuid


def detect_hostname() -> str:
    """Return the current hostname in the form used by the request export."""

    return socket.gethostname().strip() or platform.node().strip() or "unknown-host"


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
        except OSError:
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
    """Derive a deterministic fingerprint from stable host properties."""

    token = machine_token or read_machine_token(os_family)
    digest = hashlib.sha256(f"{hostname}|{os_family}|{token}".encode("utf-8")).hexdigest()
    return f"host_{digest[:12]}"
