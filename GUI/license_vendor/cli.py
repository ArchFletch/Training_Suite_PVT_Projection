"""Command-line interface for issuing evaluation and paid licenses."""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from license_server.schemas import DEFAULT_PRODUCT, LicensePayload, LicenseRequest

from .issuance_log import append_issuance_record, build_issuance_record
from .signer import (
    generate_signing_key,
    load_signing_key,
    save_signing_key,
    sign_license_payload,
)

DEFAULT_FEATURES = ("baseline", "transfer")
DEFAULT_EVALUATION_NOTES = "14-day floating POC, non-production"
DEFAULT_PAID_NOTES = "Paid floating term license"
DEFAULT_ISSUANCE_LOG = "license_vendor_issuance_log.jsonl"

# RFC 8410 fixes the Ed25519 SubjectPublicKeyInfo header, and the encoded body stays at 60
# characters, below the 64-column PEM wrap. Building the PEM here keeps the vendor tool free of
# the cryptography dependency that the pure-Python signer in signer.py already avoids.
_ED25519_SPKI_PREFIX = bytes.fromhex("302a300506032b6570032100")


def _normalize_cli_timestamp(value: str | None, *, boundary: str, default: datetime | None) -> str:
    if value is None:
        if default is None:
            raise ValueError("A timestamp is required.")
        normalized = default.astimezone(timezone.utc).replace(microsecond=0)
        return normalized.isoformat().replace("+00:00", "Z")
    candidate = value.strip()
    if "T" not in candidate:
        if boundary == "start":
            candidate = f"{candidate}T00:00:00Z"
        else:
            candidate = f"{candidate}T23:59:59Z"
    candidate = candidate[:-1] + "+00:00" if candidate.endswith("Z") else candidate
    parsed = datetime.fromisoformat(candidate)
    if parsed.tzinfo is None:
        raise ValueError("CLI timestamps must include a timezone or use a YYYY-MM-DD date.")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _derive_ends_at(starts_at: str, term_days: int) -> str:
    if term_days <= 0:
        raise ValueError("term_days must be greater than zero.")
    start_dt = datetime.fromisoformat(starts_at.replace("Z", "+00:00"))
    end_dt = start_dt + timedelta(days=term_days) - timedelta(seconds=1)
    return end_dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _default_license_id(license_type: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"lic_{license_type}_{timestamp}_{uuid4().hex[:8]}"


def _resolve_features(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_FEATURES
    return tuple(item.strip() for item in values if item.strip())


def _build_payload(args: argparse.Namespace, *, license_type: str) -> LicensePayload:
    request = LicenseRequest.from_path(args.request_file)
    starts_at = _normalize_cli_timestamp(
        args.starts_at,
        boundary="start",
        default=datetime.now(timezone.utc),
    )
    ends_at = (
        _normalize_cli_timestamp(args.ends_at, boundary="end", default=None)
        if args.ends_at
        else _derive_ends_at(starts_at, args.term_days)
    )
    notes = args.notes or (
        DEFAULT_EVALUATION_NOTES if license_type == "evaluation" else DEFAULT_PAID_NOTES
    )
    return LicensePayload(
        schema_version=request.schema_version,
        license_id=args.license_id or _default_license_id(license_type),
        license_type=license_type,
        product=args.product or request.product,
        company_name=args.company_name,
        server_id=request.server_id,
        host_fingerprint=request.host_fingerprint,
        seat_count=args.seat_count,
        features=_resolve_features(args.features),
        starts_at=starts_at,
        ends_at=ends_at,
        notes=notes,
    )


def _issue_license(args: argparse.Namespace, *, license_type: str) -> int:
    signing_key = load_signing_key(args.signing_key_file)
    payload = _build_payload(args, license_type=license_type)
    payload_dict = payload.to_dict()
    envelope = sign_license_payload(payload, signing_key)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(envelope.to_json(), encoding="utf-8")
    record = build_issuance_record(envelope, output_path=output_path)
    log_path = append_issuance_record(args.issuance_log, record)
    summary = {
        "status": "ok",
        "license_type": payload_dict["license_type"],
        "license_id": payload_dict["license_id"],
        "server_id": payload_dict["server_id"],
        "company_name": payload_dict["company_name"],
        "seat_count": payload_dict["seat_count"],
        "starts_at": payload_dict["starts_at"],
        "ends_at": payload_dict["ends_at"],
        "features": list(payload_dict["features"]),
        "key_id": envelope.key_id,
        "output_path": str(output_path),
        "issuance_log": str(log_path),
    }
    print(json.dumps(summary, ensure_ascii=True))
    return 0


def _create_key(args: argparse.Namespace) -> int:
    key = generate_signing_key(args.key_id)
    output_path = save_signing_key(args.output, key)
    summary = {
        "status": "ok",
        "algorithm": key.algorithm,
        "key_id": key.key_id,
        "created_at": key.created_at,
        "output_path": str(output_path),
        "public_key": key.public_key_base64,
    }
    print(json.dumps(summary, ensure_ascii=True))
    return 0


def _public_key_pem(public_key: bytes) -> str:
    body = base64.b64encode(_ED25519_SPKI_PREFIX + public_key).decode("ascii")
    return f"-----BEGIN PUBLIC KEY-----\n{body}\n-----END PUBLIC KEY-----\n"


def _export_public_key(args: argparse.Namespace) -> int:
    signing_key = load_signing_key(args.signing_key_file)
    output_path = Path(args.output)
    # The signing key is read before the output is written, so pointing --output at
    # it would replace the only copy of the private seed with a public PEM. The two
    # filenames differ by one word and sit in the same directory.
    if output_path.resolve() == Path(args.signing_key_file).resolve():
        raise SystemExit(
            f"Refusing to overwrite the signing key at {output_path} with the public key. "
            "Choose a different --output path, conventionally vendor_public_key.pem."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Written as bytes so the PEM keeps LF endings when it is issued on Windows and installed on
    # a Linux server.
    output_path.write_bytes(_public_key_pem(signing_key.public_key).encode("ascii"))
    summary = {
        "status": "ok",
        "algorithm": signing_key.algorithm,
        "key_id": signing_key.key_id,
        "created_at": signing_key.created_at,
        "output_path": str(output_path),
        "public_key": signing_key.public_key_base64,
    }
    print(json.dumps(summary, ensure_ascii=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the vendor CLI parser."""

    parser = argparse.ArgumentParser(
        prog="license-vendor",
        description="Issue signed evaluation and paid license files for a customer server.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_key = subparsers.add_parser("init-key", help="Create a new vendor Ed25519 signing key.")
    create_key.add_argument("--key-id", required=True, help="Short identifier for the signing key.")
    create_key.add_argument("--output", required=True, help="Path to write the vendor signing key JSON.")
    create_key.set_defaults(handler=_create_key)

    export_key = subparsers.add_parser(
        "export-public-key",
        help="Write the vendor public key as the SubjectPublicKeyInfo PEM the server verifies with.",
    )
    export_key.add_argument(
        "--signing-key-file",
        required=True,
        help="Path to the vendor signing key JSON created by init-key.",
    )
    export_key.add_argument(
        "--output",
        required=True,
        help="Path to write the vendor public key PEM sent to the customer.",
    )
    export_key.set_defaults(handler=_export_public_key)

    for command_name, license_type, default_days in (
        ("issue-eval", "evaluation", 14),
        ("issue-paid", "paid", 365),
    ):
        issue_parser = subparsers.add_parser(
            command_name,
            help=f"Issue a signed {license_type} license file.",
        )
        issue_parser.add_argument(
            "--request-file",
            required=True,
            help="Path to the customer-generated license_request.json file.",
        )
        issue_parser.add_argument(
            "--signing-key-file",
            required=True,
            help="Path to the vendor signing key JSON created by init-key.",
        )
        issue_parser.add_argument("--company-name", required=True, help="Customer company name.")
        issue_parser.add_argument(
            "--seat-count",
            type=int,
            default=1,
            help="Floating seat count to encode in the license.",
        )
        issue_parser.add_argument(
            "--feature",
            dest="features",
            action="append",
            help="Feature flag to include in the license. Repeat for multiple values.",
        )
        issue_parser.add_argument(
            "--starts-at",
            help="UTC ISO timestamp or YYYY-MM-DD date for the license start time.",
        )
        issue_parser.add_argument(
            "--ends-at",
            help="UTC ISO timestamp or YYYY-MM-DD date for the license end time.",
        )
        issue_parser.add_argument(
            "--term-days",
            type=int,
            default=default_days,
            help="Length of the license term when --ends-at is not provided.",
        )
        issue_parser.add_argument(
            "--notes",
            help="Optional human-readable notes stored in the shared payload.",
        )
        issue_parser.add_argument(
            "--license-id",
            help="Optional explicit license identifier. Defaults to a generated value.",
        )
        issue_parser.add_argument(
            "--product",
            default=None,
            help="Product name embedded in the license payload.",
        )
        issue_parser.add_argument("--output", required=True, help="Path to write the signed license JSON.")
        issue_parser.add_argument(
            "--issuance-log",
            default=DEFAULT_ISSUANCE_LOG,
            help="Path to the append-only vendor issuance log (JSON lines).",
        )
        issue_parser.set_defaults(handler=_issue_license, license_type=license_type)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the vendor CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    if handler is _issue_license:
        return handler(args, license_type=args.license_type)
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
