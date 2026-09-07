"""Focused tests for the license server service layer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
import pytest

from license_server.crypto import sign_license_payload
from license_server.schemas import CheckoutRequest, HeartbeatRequest, LicensePayload, ReleaseRequest, SignedLicense
from license_server.service import build_server_config, resolve_server_paths
from license_server.service.runtime import create_runtime


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


@pytest.fixture
def signing_keys() -> tuple[str, str]:
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    return private_pem, public_pem


@pytest.fixture
def runtime(tmp_path, monkeypatch: pytest.MonkeyPatch, signing_keys: tuple[str, str]):
    private_pem, public_pem = signing_keys
    config = build_server_config(runtime_root=tmp_path / "runtime", os_family="windows", vendor_public_key_pem=public_pem)
    server_runtime = create_runtime(config=config)

    created_at = _utc(2026, 3, 13, 12, 0, 0)
    monkeypatch.setattr("license_server.services.identity_service.utc_now", lambda: created_at)
    identity = server_runtime.identity_service.ensure_server_identity()

    return {
        "private_pem": private_pem,
        "public_pem": public_pem,
        "runtime": server_runtime,
        "identity": identity,
    }


def _make_signed_license(
    runtime_bundle,
    *,
    seat_count: int,
    starts_at: datetime,
    ends_at: datetime,
    license_id: str,
    license_type: str = "evaluation",
    company_name: str = "Acme Design House",
    key_id: str = "2026-01",
) -> SignedLicense:
    payload = LicensePayload.create(
        license_id=license_id,
        license_type=license_type,
        company_name=company_name,
        server_id=runtime_bundle["identity"].server_id,
        host_fingerprint=runtime_bundle["identity"].host_fingerprint,
        seat_count=seat_count,
        features=("baseline", "transfer"),
        starts_at=starts_at,
        ends_at=ends_at,
    )
    signature = sign_license_payload(payload, private_key_pem=runtime_bundle["private_pem"])
    return SignedLicense(
        algorithm="Ed25519",
        key_id=key_id,
        payload=payload,
        signature=signature,
    )


def _checkout_request(machine_id: str, hostname: str) -> CheckoutRequest:
    return CheckoutRequest(
        product="Surrogate Model Training Suite",
        product_version="0.1.0",
        machine_id=machine_id,
        hostname=hostname,
        username="jdoe",
        platform="windows",
    )


def test_resolve_server_paths_supports_windows_and_linux_layouts(tmp_path) -> None:
    windows_paths = resolve_server_paths(runtime_root=tmp_path / "windows-runtime", os_family="windows")
    linux_paths = resolve_server_paths(runtime_root="/var/lib/mlp-license-server", os_family="linux")

    assert windows_paths.data_dir == tmp_path / "windows-runtime" / "data"
    assert windows_paths.database_path == tmp_path / "windows-runtime" / "data" / "license.db"
    assert str(linux_paths.config_dir).replace("\\", "/").endswith("/etc/mlp-license-server")
    assert str(linux_paths.data_dir).replace("\\", "/").endswith("/var/lib/mlp-license-server")


def test_checkout_enforces_license_term_window(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    server_runtime = runtime["runtime"]

    not_started_import_time = _utc(2026, 3, 13, 12, 5, 0)
    monkeypatch.setattr("license_server.services.license_service.utc_now", lambda: not_started_import_time)
    future_license = _make_signed_license(
        runtime,
        seat_count=1,
        starts_at=_utc(2026, 3, 14, 0, 0, 0),
        ends_at=_utc(2026, 3, 28, 0, 0, 0),
        license_id="lic_future",
    )
    server_runtime.license_service.import_license(future_license)

    monkeypatch.setattr("license_server.services.lease_service.utc_now", lambda: _utc(2026, 3, 13, 13, 0, 0))
    not_started = server_runtime.lease_service.checkout(_checkout_request("cli_future", "eda-win-17"))
    assert not_started.granted is False
    assert not_started.reason_code == "license_not_started"

    expired_import_time = _utc(2026, 3, 29, 0, 0, 0)
    monkeypatch.setattr("license_server.services.license_service.utc_now", lambda: expired_import_time)
    expired_license = _make_signed_license(
        runtime,
        seat_count=1,
        starts_at=_utc(2026, 3, 1, 0, 0, 0),
        ends_at=_utc(2026, 3, 10, 0, 0, 0),
        license_id="lic_expired",
    )
    server_runtime.license_service.import_license(expired_license)

    monkeypatch.setattr("license_server.services.lease_service.utc_now", lambda: _utc(2026, 3, 29, 0, 1, 0))
    expired = server_runtime.lease_service.checkout(_checkout_request("cli_expired", "eda-win-18"))
    assert expired.granted is False
    assert expired.reason_code == "license_expired"


def test_checkout_denies_when_all_seats_are_in_use(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    server_runtime = runtime["runtime"]

    monkeypatch.setattr("license_server.services.license_service.utc_now", lambda: _utc(2026, 3, 13, 12, 5, 0))
    license_envelope = _make_signed_license(
        runtime,
        seat_count=1,
        starts_at=_utc(2026, 3, 13, 0, 0, 0),
        ends_at=_utc(2026, 4, 13, 0, 0, 0),
        license_id="lic_one_seat",
    )
    server_runtime.license_service.import_license(license_envelope)

    checkout_time = _utc(2026, 3, 13, 12, 10, 0)
    monkeypatch.setattr("license_server.services.lease_service.utc_now", lambda: checkout_time)
    first = server_runtime.lease_service.checkout(_checkout_request("cli_1", "eda-win-17"))
    second = server_runtime.lease_service.checkout(_checkout_request("cli_2", "eda-win-18"))

    assert first.granted is True
    assert second.granted is False
    assert second.reason_code == "all_seats_in_use"


def test_heartbeat_timeout_releases_seat_for_new_checkout(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    server_runtime = runtime["runtime"]

    monkeypatch.setattr("license_server.services.license_service.utc_now", lambda: _utc(2026, 3, 13, 12, 5, 0))
    license_envelope = _make_signed_license(
        runtime,
        seat_count=1,
        starts_at=_utc(2026, 3, 13, 0, 0, 0),
        ends_at=_utc(2026, 4, 13, 0, 0, 0),
        license_id="lic_timeout",
    )
    server_runtime.license_service.import_license(license_envelope)

    start_time = _utc(2026, 3, 13, 12, 10, 0)
    monkeypatch.setattr("license_server.services.lease_service.utc_now", lambda: start_time)
    checkout = server_runtime.lease_service.checkout(_checkout_request("cli_timeout", "eda-win-17"))
    assert checkout.granted is True

    heartbeat_time = start_time + timedelta(seconds=60)
    monkeypatch.setattr("license_server.services.lease_service.utc_now", lambda: heartbeat_time)
    heartbeat = server_runtime.lease_service.heartbeat(
        HeartbeatRequest(lease_id=checkout.lease_id, machine_id="cli_timeout")
    )
    assert heartbeat.ok is True

    cleanup_time = heartbeat_time + timedelta(seconds=121)
    timed_out = server_runtime.lease_service.cleanup_expired_leases(now=cleanup_time)
    assert [lease.lease_id for lease in timed_out] == [checkout.lease_id]
    assert timed_out[0].release_reason == "heartbeat_timeout"

    monkeypatch.setattr("license_server.services.lease_service.utc_now", lambda: cleanup_time)
    replacement = server_runtime.lease_service.checkout(_checkout_request("cli_timeout_2", "eda-win-18"))
    assert replacement.granted is True
    assert replacement.lease_id != checkout.lease_id


def test_license_downgrade_immediately_evicts_excess_leases(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    server_runtime = runtime["runtime"]

    monkeypatch.setattr("license_server.services.license_service.utc_now", lambda: _utc(2026, 3, 13, 12, 5, 0))
    initial_license = _make_signed_license(
        runtime,
        seat_count=2,
        starts_at=_utc(2026, 3, 13, 0, 0, 0),
        ends_at=_utc(2026, 4, 13, 0, 0, 0),
        license_id="lic_two_seats",
    )
    server_runtime.license_service.import_license(initial_license)

    monkeypatch.setattr("license_server.services.lease_service.utc_now", lambda: _utc(2026, 3, 13, 12, 10, 0))
    first = server_runtime.lease_service.checkout(_checkout_request("cli_a", "eda-win-17"))

    monkeypatch.setattr("license_server.services.lease_service.utc_now", lambda: _utc(2026, 3, 13, 12, 11, 0))
    second = server_runtime.lease_service.checkout(_checkout_request("cli_b", "eda-win-18"))
    assert first.granted is True
    assert second.granted is True

    monkeypatch.setattr("license_server.services.license_service.utc_now", lambda: _utc(2026, 3, 13, 12, 11, 30))
    downgraded_license = _make_signed_license(
        runtime,
        seat_count=1,
        starts_at=_utc(2026, 3, 13, 0, 0, 0),
        ends_at=_utc(2026, 4, 13, 0, 0, 0),
        license_id="lic_one_seat_after_downgrade",
    )
    result = server_runtime.license_service.import_license(downgraded_license)

    assert result.ok is True
    assert result.evicted_lease_ids == (second.lease_id,)

    active_leases = server_runtime.lease_service.list_active_leases(now=_utc(2026, 3, 13, 12, 11, 31))
    assert [lease.lease_id for lease in active_leases] == [first.lease_id]

    monkeypatch.setattr("license_server.services.lease_service.utc_now", lambda: _utc(2026, 3, 13, 12, 11, 35))
    evicted_heartbeat = server_runtime.lease_service.heartbeat(
        HeartbeatRequest(lease_id=second.lease_id, machine_id="cli_b")
    )
    assert evicted_heartbeat.ok is False
    assert evicted_heartbeat.reason_code == "invalid_lease"


def test_release_requires_the_machine_that_owns_the_lease(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    server_runtime = runtime["runtime"]

    monkeypatch.setattr("license_server.services.license_service.utc_now", lambda: _utc(2026, 3, 13, 12, 5, 0))
    license_envelope = _make_signed_license(
        runtime,
        seat_count=1,
        starts_at=_utc(2026, 3, 13, 0, 0, 0),
        ends_at=_utc(2026, 4, 13, 0, 0, 0),
        license_id="lic_release_binding",
    )
    server_runtime.license_service.import_license(license_envelope)

    checkout_time = _utc(2026, 3, 13, 12, 10, 0)
    monkeypatch.setattr("license_server.services.lease_service.utc_now", lambda: checkout_time)
    checkout = server_runtime.lease_service.checkout(_checkout_request("cli_owner", "eda-win-17"))
    assert checkout.granted is True

    stolen = server_runtime.lease_service.release(
        ReleaseRequest(lease_id=checkout.lease_id, machine_id="cli_thief")
    )
    assert stolen.ok is False
    assert stolen.reason_code == "machine_mismatch"

    still_held = server_runtime.lease_service.list_active_leases(now=checkout_time)
    assert [lease.lease_id for lease in still_held] == [checkout.lease_id]

    released = server_runtime.lease_service.release(
        ReleaseRequest(lease_id=checkout.lease_id, machine_id="cli_owner")
    )
    assert released.ok is True
    assert server_runtime.lease_service.list_active_leases(now=checkout_time) == []


def test_release_rejects_an_unknown_lease(runtime) -> None:
    server_runtime = runtime["runtime"]

    missing = server_runtime.lease_service.release(
        ReleaseRequest(lease_id="lease_does_not_exist", machine_id="cli_owner")
    )

    assert missing.ok is False
    assert missing.reason_code == "invalid_lease"


def _audit_events(server_runtime) -> list[dict]:
    import json as _json

    with server_runtime.session_factory.session() as connection:
        rows = connection.execute(
            "SELECT event_type, details_json FROM audit_events ORDER BY id"
        ).fetchall()
    return [{"event_type": r["event_type"], **_json.loads(r["details_json"])} for r in rows]


def test_a_second_checkout_from_one_machine_reuses_its_seat(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    """A floating seat is per machine, not per process. Two GUI windows on one
    workstation used to take two seats, so one engineer could lock a colleague out
    of a two-seat licence."""
    server_runtime = runtime["runtime"]
    monkeypatch.setattr("license_server.services.license_service.utc_now", lambda: _utc(2026, 3, 13, 12, 5, 0))
    server_runtime.license_service.import_license(
        _make_signed_license(
            runtime, seat_count=2, starts_at=_utc(2026, 3, 13), ends_at=_utc(2026, 4, 13), license_id="lic_two_seats"
        )
    )
    monkeypatch.setattr("license_server.services.lease_service.utc_now", lambda: _utc(2026, 3, 13, 12, 10, 0))

    first = server_runtime.lease_service.checkout(_checkout_request("gui_same", "eda-win-17"))
    second = server_runtime.lease_service.checkout(_checkout_request("gui_same", "eda-win-17"))

    assert first.granted is True and second.granted is True
    assert second.lease_id == first.lease_id, "a second window took a second lease"
    assert second.seats_in_use == 1

    # A genuinely different machine still consumes the second seat.
    other = server_runtime.lease_service.checkout(_checkout_request("gui_other", "eda-win-18"))
    assert other.granted is True
    assert other.lease_id != first.lease_id
    assert other.seats_in_use == 2

    # And the third machine is refused, so dedup did not inflate the seat pool.
    third = server_runtime.lease_service.checkout(_checkout_request("gui_third", "eda-win-19"))
    assert third.granted is False and third.reason_code == "all_seats_in_use"

    reused = [e for e in _audit_events(server_runtime) if e["event_type"] == "checkout_reused"]
    assert len(reused) == 1 and reused[0]["machine_id"] == "gui_same"


def test_reusing_a_seat_extends_the_lease_rather_than_leaving_it_to_expire(
    runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_runtime = runtime["runtime"]
    monkeypatch.setattr("license_server.services.license_service.utc_now", lambda: _utc(2026, 3, 13, 12, 5, 0))
    server_runtime.license_service.import_license(
        _make_signed_license(
            runtime, seat_count=2, starts_at=_utc(2026, 3, 13), ends_at=_utc(2026, 4, 13), license_id="lic_renew"
        )
    )
    monkeypatch.setattr("license_server.services.lease_service.utc_now", lambda: _utc(2026, 3, 13, 12, 10, 0))
    first = server_runtime.lease_service.checkout(_checkout_request("gui_same", "eda-win-17"))

    # Still inside the 120 s lease TTL: the same machine must get the SAME lease
    # back, with its expiry pushed out. (Past the TTL a fresh lease is correct,
    # which the heartbeat-timeout test already covers.)
    monkeypatch.setattr("license_server.services.lease_service.utc_now", lambda: _utc(2026, 3, 13, 12, 11, 0))
    again = server_runtime.lease_service.checkout(_checkout_request("gui_same", "eda-win-17"))

    assert again.lease_id == first.lease_id
    assert again.expires_at > first.expires_at, "the reused lease kept its old expiry"


@pytest.mark.parametrize(
    "reason_code, requested_feature, license_window",
    [
        ("feature_not_enabled", "quantum_solver", (_utc(2026, 3, 13), _utc(2026, 4, 13))),
        ("license_expired", None, (_utc(2026, 1, 1), _utc(2026, 2, 1))),
        ("license_not_started", None, (_utc(2026, 6, 1), _utc(2026, 7, 1))),
    ],
)
def test_license_layer_refusals_are_written_to_the_audit_log(
    runtime, monkeypatch: pytest.MonkeyPatch, reason_code, requested_feature, license_window
) -> None:
    """validate_checkout returned these straight to the caller and wrote nothing, so
    an audit could not answer 'who was turned away, and why'. Only all_seats_in_use
    was ever recorded."""
    server_runtime = runtime["runtime"]
    monkeypatch.setattr("license_server.services.license_service.utc_now", lambda: _utc(2026, 3, 13, 12, 5, 0))
    starts_at, ends_at = license_window
    server_runtime.license_service.import_license(
        _make_signed_license(
            runtime, seat_count=2, starts_at=starts_at, ends_at=ends_at, license_id=f"lic_{reason_code}"
        )
    )
    monkeypatch.setattr("license_server.services.lease_service.utc_now", lambda: _utc(2026, 3, 13, 12, 10, 0))

    request = _checkout_request("gui_denied", "eda-win-17")
    if requested_feature is not None:
        request = request.model_copy(update={"requested_feature": requested_feature})
    denied = server_runtime.lease_service.checkout(request)

    assert denied.granted is False and denied.reason_code == reason_code
    recorded = [e for e in _audit_events(server_runtime) if e["event_type"] == "checkout_denied"]
    assert len(recorded) == 1, f"{reason_code} left no audit trail"
    assert recorded[0]["reason_code"] == reason_code
    assert recorded[0]["machine_id"] == "gui_denied"
    assert recorded[0]["requested_feature"] == requested_feature


def test_a_checkout_with_no_license_at_all_is_audited(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    server_runtime = runtime["runtime"]
    monkeypatch.setattr("license_server.services.lease_service.utc_now", lambda: _utc(2026, 3, 13, 12, 10, 0))

    denied = server_runtime.lease_service.checkout(_checkout_request("gui_nolic", "eda-win-17"))

    assert denied.granted is False and denied.reason_code == "invalid_license"
    recorded = [e for e in _audit_events(server_runtime) if e["event_type"] == "checkout_denied"]
    assert len(recorded) == 1 and recorded[0]["reason_code"] == "invalid_license"
