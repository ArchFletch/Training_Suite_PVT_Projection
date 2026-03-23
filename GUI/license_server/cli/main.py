"""Typer entrypoints for the customer-admin CLI workflow."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError
import typer

from license_server.schemas import SignedLicense
from license_server.service import LicenseServerRuntime, build_server_config, create_runtime, format_utc_datetime, utc_now
from license_server.services import LicenseImportError


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Customer-admin commands for the on-prem MLP license server.",
)


def _runtime(
    *,
    runtime_root: Path | None,
    vendor_public_key_path: Path | None = None,
) -> LicenseServerRuntime:
    vendor_public_key_pem = _read_vendor_public_key(runtime_root=runtime_root, vendor_public_key_path=vendor_public_key_path)
    return create_runtime(
        runtime_root=str(runtime_root) if runtime_root is not None else None,
        vendor_public_key_pem=vendor_public_key_pem,
    )


def _read_vendor_public_key(
    *,
    runtime_root: Path | None,
    vendor_public_key_path: Path | None,
) -> str:
    if vendor_public_key_path is not None:
        return vendor_public_key_path.read_text(encoding="utf-8")

    config = build_server_config(runtime_root=str(runtime_root) if runtime_root is not None else None)
    default_path = config.paths.config_dir / "vendor_public_key.pem"
    if default_path.exists():
        return default_path.read_text(encoding="utf-8")
    return ""


def _dump_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _fail(message: str, *, exit_code: int = 1) -> None:
    typer.echo(json.dumps({"ok": False, "error": message}, indent=2, sort_keys=True), err=True)
    raise typer.Exit(code=exit_code)


def _build_admin_status(runtime: LicenseServerRuntime) -> dict[str, object]:
    now = format_utc_datetime(utc_now())
    status = runtime.license_service.get_status(now=now).to_dict()
    with runtime.session_factory.session() as connection:
        identity = runtime.repository.get_server_identity(connection)
        active_license = runtime.repository.get_active_license(connection)
        active_leases = runtime.repository.list_active_leases(connection, now=now)

    return {
        "ok": True,
        "runtime_root": str(runtime.config.paths.runtime_root),
        "data_dir": str(runtime.config.paths.data_dir),
        "license_loaded": active_license is not None,
        "identity": identity.to_dict() if identity is not None else None,
        "license_id": active_license.payload.license_id if active_license is not None else None,
        "product": status["product"],
        "company_name": status.get("company_name"),
        "license_type": status.get("license_type"),
        "starts_at": status.get("starts_at"),
        "ends_at": status.get("ends_at"),
        "seat_count": status["seat_count"],
        "seats_in_use": status["seats_in_use"],
        "features": list(active_license.payload.features) if active_license is not None else [],
        "active_leases": [lease.to_dict() for lease in active_leases],
    }


@app.command("init")
def init_server(
    runtime_root: Path | None = typer.Option(
        None,
        "--runtime-root",
        "--data-dir",
        help="Runtime root for the license server. The legacy --data-dir alias is still accepted.",
    ),
) -> None:
    """Create the local server identity and SQLite store if they do not already exist."""

    runtime = _runtime(runtime_root=runtime_root)
    identity = runtime.identity_service.ensure_server_identity()
    _dump_json(
        {
            "ok": True,
            "runtime_root": str(runtime.config.paths.runtime_root),
            "data_dir": str(runtime.config.paths.data_dir),
            "identity": identity.to_dict(),
        }
    )


@app.command("export-request")
def export_request(
    output_path: Path = typer.Argument(..., help="Where to write the license request JSON."),
    runtime_root: Path | None = typer.Option(
        None,
        "--runtime-root",
        "--data-dir",
        help="Runtime root for the license server. The legacy --data-dir alias is still accepted.",
    ),
    requested_by: str | None = typer.Option(None, "--requested-by", help="Optional operator name."),
) -> None:
    """Write `license_request.json` for the vendor licensing workflow."""

    runtime = _runtime(runtime_root=runtime_root)
    request_model = runtime.identity_service.export_license_request(
        requested_by=requested_by,
        output_path=output_path,
    )
    _dump_json(
        {
            "ok": True,
            "runtime_root": str(runtime.config.paths.runtime_root),
            "data_dir": str(runtime.config.paths.data_dir),
            "output_path": str(output_path),
            "request": request_model.to_dict(),
        }
    )


@app.command("import-license")
def import_license(
    license_path: Path = typer.Argument(..., exists=True, readable=True, dir_okay=False),
    runtime_root: Path | None = typer.Option(
        None,
        "--runtime-root",
        "--data-dir",
        help="Runtime root for the license server. The legacy --data-dir alias is still accepted.",
    ),
    vendor_public_key_path: Path | None = typer.Option(
        None,
        "--vendor-public-key",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Path to the vendor Ed25519 public key PEM. Defaults to <runtime-root>/config/vendor_public_key.pem when present.",
    ),
) -> None:
    """Import or replace the active customer license JSON."""

    runtime = _runtime(runtime_root=runtime_root, vendor_public_key_path=vendor_public_key_path)
    try:
        envelope = SignedLicense.model_validate_json(license_path.read_text(encoding="utf-8"))
        result = runtime.license_service.import_license(envelope)
    except ValidationError as exc:
        _fail(f"License file is not valid JSON for the expected envelope schema: {exc}")
    except LicenseImportError as exc:
        _fail(str(exc))

    imported_license = runtime.license_service.get_active_license()
    _dump_json(
        {
            "ok": result.ok,
            "runtime_root": str(runtime.config.paths.runtime_root),
            "data_dir": str(runtime.config.paths.data_dir),
            "license_id": result.license_id,
            "imported_at": result.imported_at,
            "evicted_lease_ids": list(result.evicted_lease_ids),
            "message": result.message,
            "license": imported_license.payload.to_dict() if imported_license is not None else None,
        }
    )


@app.command("show-status")
def show_status(
    runtime_root: Path | None = typer.Option(
        None,
        "--runtime-root",
        "--data-dir",
        help="Runtime root for the license server. The legacy --data-dir alias is still accepted.",
    ),
) -> None:
    """Print the current server/license summary for customer IT."""

    runtime = _runtime(runtime_root=runtime_root)
    _dump_json(_build_admin_status(runtime))


@app.command("show-audit")
def show_audit(
    runtime_root: Path | None = typer.Option(
        None,
        "--runtime-root",
        "--data-dir",
        help="Runtime root for the license server. The legacy --data-dir alias is still accepted.",
    ),
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum number of audit events to show."),
) -> None:
    """Print recent audit events from the local SQLite store."""

    runtime = _runtime(runtime_root=runtime_root)
    with runtime.session_factory.session() as connection:
        events = runtime.repository.list_recent_audit_events(connection, limit=limit)
    _dump_json({"ok": True, "events": [event.to_dict() for event in events]})


def main() -> None:
    app()


if __name__ == "__main__":
    main()
