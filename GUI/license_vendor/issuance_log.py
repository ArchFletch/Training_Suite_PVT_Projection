"""Persistence helpers for the vendor-side issuance log."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from license_server.schemas import SignedLicenseEnvelope, canonical_payload_bytes


@dataclass(frozen=True, slots=True)
class IssuanceRecord:
    """One append-only record of a vendor-issued license."""

    issued_at: str
    output_path: str
    algorithm: str
    key_id: str
    license_id: str
    license_type: str
    company_name: str
    product: str
    server_id: str
    host_fingerprint: str
    seat_count: int
    features: tuple[str, ...]
    starts_at: str
    ends_at: str
    payload_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "issued_at": self.issued_at,
            "output_path": self.output_path,
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "license_id": self.license_id,
            "license_type": self.license_type,
            "company_name": self.company_name,
            "product": self.product,
            "server_id": self.server_id,
            "host_fingerprint": self.host_fingerprint,
            "seat_count": self.seat_count,
            "features": list(self.features),
            "starts_at": self.starts_at,
            "ends_at": self.ends_at,
            "payload_sha256": self.payload_sha256,
        }


def build_issuance_record(
    envelope: SignedLicenseEnvelope,
    *,
    output_path: str | Path,
    issued_at: str | None = None,
) -> IssuanceRecord:
    """Create a normalized record from a signed license envelope."""

    payload = envelope.payload
    payload_dict = payload.to_dict()
    record_time = issued_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    digest = hashlib.sha256(canonical_payload_bytes(payload)).hexdigest()
    return IssuanceRecord(
        issued_at=record_time,
        output_path=str(Path(output_path)),
        algorithm=envelope.algorithm,
        key_id=envelope.key_id,
        license_id=payload_dict["license_id"],
        license_type=payload_dict["license_type"],
        company_name=payload_dict["company_name"],
        product=payload_dict["product"],
        server_id=payload_dict["server_id"],
        host_fingerprint=payload_dict["host_fingerprint"],
        seat_count=payload_dict["seat_count"],
        features=tuple(payload_dict["features"]),
        starts_at=payload_dict["starts_at"],
        ends_at=payload_dict["ends_at"],
        payload_sha256=digest,
    )


def append_issuance_record(path: str | Path, record: IssuanceRecord) -> Path:
    """Append one JSON-lines record to the issuance log."""

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=True) + "\n")
    return log_path


def read_issuance_log(path: str | Path) -> list[dict[str, object]]:
    """Load an issuance log from JSON-lines storage."""

    log_path = Path(path)
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
