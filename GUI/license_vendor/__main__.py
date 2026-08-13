"""Module entrypoint for `python -m license_vendor`."""

from __future__ import annotations

import sys

_LICENSE_SERVER_REQUIRED = (
    "license_vendor needs the license_server package to be importable: the request, payload, and "
    "envelope schemas it signs live in license_server.schemas.\n"
    "Run the vendor commands from the GUI/ directory, where license_server and license_vendor are "
    "sibling packages, or put that directory on PYTHONPATH. Copying license_vendor/ on its own to "
    "a vendor workstation is not enough."
)


def _is_missing_license_server(error: ImportError) -> bool:
    name = error.name or ""
    return name == "license_server" or name.startswith("license_server.")


try:
    from .cli import main
except ImportError as exc:  # pragma: no cover - depends on the vendor workstation layout
    # A vendor workstation that only has license_vendor/ used to fail here with a bare
    # ModuleNotFoundError from an import three levels down, with nothing pointing at the fix.
    if not _is_missing_license_server(exc):
        raise
    print(_LICENSE_SERVER_REQUIRED, file=sys.stderr)
    raise SystemExit(2) from exc


if __name__ == "__main__":
    raise SystemExit(main())
