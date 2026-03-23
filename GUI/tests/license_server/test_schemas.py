"""Focused contract tests for the on-prem license-server schema layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from license_server.schemas import (
    CHECKOUT_DENIAL_REASON_CODES,
    CheckoutDeniedResponse,
    CheckoutGrantedResponse,
    CheckoutRequest,
    CheckoutResponse,
    DenialReasonCode,
    HeartbeatRequest,
    HeartbeatResponse,
    LicenseRequest,
    ReleaseRequest,
    ReleaseResponse,
    ServerIdentity,
    SignedLicenseEnvelope,
    StatusResponse,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures"

VALID_MODEL_FIXTURES = [
    ("server_identity.json", ServerIdentity),
    ("license_request.json", LicenseRequest),
    ("license_envelope.json", SignedLicenseEnvelope),
    ("checkout_request.json", CheckoutRequest),
    ("checkout_granted_response.json", CheckoutGrantedResponse),
    ("checkout_denied_response.json", CheckoutDeniedResponse),
    ("checkout_request_with_feature.json", CheckoutRequest),
    ("heartbeat_request.json", HeartbeatRequest),
    ("heartbeat_response.json", HeartbeatResponse),
    ("heartbeat_failed_response.json", HeartbeatResponse),
    ("release_request.json", ReleaseRequest),
    ("release_response.json", ReleaseResponse),
    ("release_failed_response.json", ReleaseResponse),
    ("status_response.json", StatusResponse),
    ("status_response_unlicensed.json", StatusResponse),
]

INVALID_MODEL_FIXTURES = [
    ("server_identity_naive_created_at.json", ServerIdentity),
    ("license_request_missing_server_id.json", LicenseRequest),
    ("license_envelope_bad_signature.json", SignedLicenseEnvelope),
    ("license_envelope_end_before_start.json", SignedLicenseEnvelope),
    ("checkout_request_bad_platform.json", CheckoutRequest),
    ("heartbeat_response_missing_reason.json", HeartbeatResponse),
    ("release_response_missing_reason.json", ReleaseResponse),
    ("status_response_active_missing_company.json", StatusResponse),
    ("status_response_overbooked.json", StatusResponse),
]

CHECKOUT_RESPONSE_ADAPTER = TypeAdapter(CheckoutResponse)


def _load_fixture(*parts: str) -> dict[str, object]:
    path = FIXTURE_ROOT.joinpath(*parts)
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("fixture_name", "model_cls"), VALID_MODEL_FIXTURES)
def test_valid_fixtures_round_trip_through_models(fixture_name: str, model_cls: type) -> None:
    payload = _load_fixture("valid", fixture_name)

    model = model_cls.model_validate(payload)
    serialized = json.loads(model.model_dump_json(exclude_none=True))

    reparsed = model_cls.model_validate(serialized)
    assert reparsed == model


@pytest.mark.parametrize(
    ("fixture_name", "expected_type"),
    [
        ("checkout_granted_response.json", CheckoutGrantedResponse),
        ("checkout_denied_response.json", CheckoutDeniedResponse),
    ],
)
def test_checkout_response_union_uses_granted_discriminator(
    fixture_name: str,
    expected_type: type,
) -> None:
    payload = _load_fixture("valid", fixture_name)

    parsed = CHECKOUT_RESPONSE_ADAPTER.validate_python(payload)

    assert isinstance(parsed, expected_type)


@pytest.mark.parametrize(("fixture_name", "model_cls"), INVALID_MODEL_FIXTURES)
def test_invalid_model_fixtures_raise_validation_error(fixture_name: str, model_cls: type) -> None:
    payload = _load_fixture("invalid", fixture_name)

    with pytest.raises(ValidationError):
        model_cls.model_validate(payload)


def test_invalid_checkout_denial_reason_code_is_rejected() -> None:
    payload = _load_fixture("invalid", "checkout_denied_unknown_reason_code.json")

    with pytest.raises(ValidationError):
        CHECKOUT_RESPONSE_ADAPTER.validate_python(payload)


def test_checkout_denial_reason_codes_match_the_mvp_contract() -> None:
    expected_codes = (
        "all_seats_in_use",
        "license_expired",
        "license_not_started",
        "feature_not_enabled",
        "server_binding_mismatch",
        "invalid_license",
    )

    assert CHECKOUT_DENIAL_REASON_CODES == expected_codes
    assert tuple(reason.value for reason in DenialReasonCode) == expected_codes
