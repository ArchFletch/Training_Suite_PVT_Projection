"""Shared file-format schemas for server identity, requests, and licenses."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from .common import (
    PRODUCT_NAME,
    SCHEMA_VERSION,
    Base64Signature,
    FeatureCode,
    HostFingerprint,
    Hostname,
    KeyId,
    LicenseId,
    LicenseServerSchemaModel,
    LicenseType,
    NonEmptyText,
    OSFamily,
    OptionalLongText,
    OptionalShortText,
    PositiveSeatCount,
    ServerId,
    SigningAlgorithm,
    UtcDatetime,
)


class ContractDocument(LicenseServerSchemaModel):
    """Base document shape shared by persisted identity and signed payload files."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    product: Literal[PRODUCT_NAME] = PRODUCT_NAME


class ServerIdentity(ContractDocument):
    """Persistent identity created by the server on first startup."""

    server_id: ServerId
    host_fingerprint: HostFingerprint
    hostname: Hostname
    os_family: OSFamily
    created_at: UtcDatetime

    @classmethod
    def create(
        cls,
        *,
        server_id: str,
        host_fingerprint: str,
        hostname: str,
        os_family: str | OSFamily,
        created_at: datetime | str,
        product: str = PRODUCT_NAME,
    ) -> "ServerIdentity":
        """Convenience constructor used by the runtime services."""

        return cls(
            product=product,
            server_id=server_id,
            host_fingerprint=host_fingerprint,
            hostname=hostname,
            os_family=os_family,
            created_at=created_at,
        )


class LicenseRequest(ContractDocument):
    """Customer-generated request file sent to the vendor for license issuance."""

    server_id: ServerId
    host_fingerprint: HostFingerprint
    hostname: Hostname
    os_family: OSFamily
    generated_at: UtcDatetime
    requested_by: OptionalShortText | None = None

    @classmethod
    def from_identity(
        cls,
        identity: ServerIdentity,
        *,
        generated_at: datetime | str,
        requested_by: str | None = None,
    ) -> "LicenseRequest":
        """Build a request document directly from the persisted server identity."""

        return cls(
            product=identity.product,
            server_id=identity.server_id,
            host_fingerprint=identity.host_fingerprint,
            hostname=identity.hostname,
            os_family=identity.os_family,
            generated_at=generated_at,
            requested_by=requested_by,
        )


class LicensePayload(ContractDocument):
    """Signed license payload shared by evaluation and paid licenses."""

    license_id: LicenseId
    license_type: LicenseType
    company_name: NonEmptyText
    server_id: ServerId
    host_fingerprint: HostFingerprint
    seat_count: PositiveSeatCount
    features: list[FeatureCode] = Field(min_length=1)
    starts_at: UtcDatetime
    ends_at: UtcDatetime
    notes: OptionalLongText | None = None

    @classmethod
    def create(
        cls,
        *,
        license_id: str,
        license_type: str | LicenseType,
        company_name: str,
        server_id: str,
        host_fingerprint: str,
        seat_count: int,
        features: list[str] | tuple[str, ...],
        starts_at: datetime | str,
        ends_at: datetime | str,
        notes: str | None = None,
        product: str = PRODUCT_NAME,
    ) -> "LicensePayload":
        """Convenience constructor shared by vendor and service layers."""

        return cls(
            product=product,
            license_id=license_id,
            license_type=license_type,
            company_name=company_name,
            server_id=server_id,
            host_fingerprint=host_fingerprint,
            seat_count=seat_count,
            features=list(features),
            starts_at=starts_at,
            ends_at=ends_at,
            notes=notes,
        )

    @model_validator(mode="after")
    def validate_license_term(self) -> "LicensePayload":
        """Keep the payload internally consistent before signature verification."""

        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at must be after starts_at")
        if len(set(self.features)) != len(self.features):
            raise ValueError("features must not contain duplicates")
        return self


class SignedLicenseEnvelope(LicenseServerSchemaModel):
    """Outer envelope that carries a signed license payload."""

    algorithm: Literal[SigningAlgorithm.ED25519] = SigningAlgorithm.ED25519
    key_id: KeyId
    payload: LicensePayload
    signature: Base64Signature


def canonical_payload_bytes(payload: LicensePayload | dict[str, object]) -> bytes:
    """Serialize one payload into the stable JSON form used for signatures."""

    model = payload if isinstance(payload, LicensePayload) else LicensePayload.model_validate(payload)
    return json.dumps(
        model.to_dict(),
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
