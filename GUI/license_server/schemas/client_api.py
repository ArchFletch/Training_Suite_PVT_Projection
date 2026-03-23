"""Client-facing request and response contracts for the on-prem license server."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .common import (
    PRODUCT_NAME,
    ClientPlatform,
    FeatureCode,
    Hostname,
    LeaseId,
    LicenseServerSchemaModel,
    LicenseType,
    MachineId,
    NonEmptyText,
    NonNegativeInt,
    OptionalShortText,
    PositiveInt,
    PositiveSeatCount,
    ProductVersion,
    UtcDatetime,
)
from .reason_codes import DenialReasonCode


class CheckoutRequest(LicenseServerSchemaModel):
    """Payload sent by a desktop client when requesting a floating seat."""

    product: Literal[PRODUCT_NAME] = PRODUCT_NAME
    product_version: ProductVersion
    machine_id: MachineId
    hostname: Hostname
    username: OptionalShortText | None = None
    platform: ClientPlatform
    requested_feature: FeatureCode | None = None


class CheckoutGrantedResponse(LicenseServerSchemaModel):
    """Successful checkout response returned after a seat is granted."""

    granted: Literal[True] = True
    lease_id: LeaseId
    heartbeat_interval_seconds: PositiveInt
    lease_ttl_seconds: PositiveInt
    grace_seconds: PositiveInt
    expires_at: UtcDatetime
    license_type: LicenseType
    company_name: NonEmptyText
    seat_count: PositiveSeatCount
    seats_in_use: NonNegativeInt

    @model_validator(mode="after")
    def validate_seat_counts(self) -> "CheckoutGrantedResponse":
        """Ensure the summary in the response stays self-consistent."""

        if not 1 <= self.seats_in_use <= self.seat_count:
            raise ValueError("seats_in_use must be between 1 and seat_count")
        return self


class CheckoutDeniedResponse(LicenseServerSchemaModel):
    """Denied checkout response with a standardized reason code."""

    granted: Literal[False] = False
    reason_code: DenialReasonCode
    message: NonEmptyText


CheckoutResponse = Annotated[
    CheckoutGrantedResponse | CheckoutDeniedResponse,
    Field(discriminator="granted"),
]


class HeartbeatRequest(LicenseServerSchemaModel):
    """Heartbeat request used to extend an existing lease."""

    lease_id: LeaseId
    machine_id: MachineId


class HeartbeatResponse(LicenseServerSchemaModel):
    """Successful heartbeat response."""

    ok: bool
    expires_at: UtcDatetime | None = None
    reason_code: OptionalShortText | None = None
    message: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_heartbeat_outcome(self) -> "HeartbeatResponse":
        """Require either a renewed expiry or a structured failure reason."""

        if self.ok:
            if self.expires_at is None:
                raise ValueError("expires_at is required when ok is true")
            if self.reason_code is not None or self.message is not None:
                raise ValueError("reason_code and message must be omitted when ok is true")
            return self

        if self.reason_code is None or self.message is None:
            raise ValueError("reason_code and message are required when ok is false")
        return self


class ReleaseRequest(LicenseServerSchemaModel):
    """Release request sent during clean client shutdown."""

    lease_id: LeaseId


class ReleaseResponse(LicenseServerSchemaModel):
    """Successful release response."""

    ok: bool
    reason_code: OptionalShortText | None = None
    message: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_release_outcome(self) -> "ReleaseResponse":
        """Require a reason only when the release was not successful."""

        if self.ok:
            if self.reason_code is not None:
                raise ValueError("reason_code must be omitted when ok is true")
            return self

        if self.reason_code is None or self.message is None:
            raise ValueError("reason_code and message are required when ok is false")
        return self


class StatusResponse(LicenseServerSchemaModel):
    """Client-readable summary exposed by the status endpoint."""

    ok: bool = True
    product: Literal[PRODUCT_NAME] = PRODUCT_NAME
    company_name: NonEmptyText | None = None
    license_type: LicenseType | None = None
    starts_at: UtcDatetime | None = None
    ends_at: UtcDatetime | None = None
    seat_count: NonNegativeInt = 0
    seats_in_use: NonNegativeInt

    @model_validator(mode="after")
    def validate_status_counts(self) -> "StatusResponse":
        """Prevent impossible seat summaries from leaving the API layer."""

        if self.ok:
            if None in (self.company_name, self.license_type, self.starts_at, self.ends_at):
                raise ValueError("active license fields are required when ok is true")
            if self.seat_count < 1:
                raise ValueError("seat_count must be at least 1 when ok is true")
            if self.ends_at <= self.starts_at:
                raise ValueError("ends_at must be after starts_at")
        elif any(value is not None for value in (self.company_name, self.license_type, self.starts_at, self.ends_at)):
            raise ValueError("inactive status must omit active license fields")

        if self.seats_in_use > self.seat_count:
            raise ValueError("seats_in_use must be less than or equal to seat_count")
        return self
