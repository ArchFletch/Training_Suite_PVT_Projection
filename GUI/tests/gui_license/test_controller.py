from __future__ import annotations

from datetime import datetime, timedelta, timezone

from xfmr_v2.licensing import (
    LicenseCheckoutResult,
    LicenseConnectionError,
    LicenseHeartbeatResult,
    LicenseLeaseController,
    LicenseReleaseResult,
    LicenseStatus,
)


class _FrozenClock:
    def __init__(self, start: datetime) -> None:
        self.value = start

    def now(self) -> datetime:
        return self.value

    def advance(self, *, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class _FakeLicenseClient:
    def __init__(self, server: "_FakeServer") -> None:
        self.server = server

    def get_status(self) -> LicenseStatus:
        return self.server.status

    def checkout(self, identity) -> LicenseCheckoutResult:
        self.server.checkout_calls.append(identity)
        return self.server.checkout_result

    def heartbeat(self, *, lease_id: str, machine_id: str) -> LicenseHeartbeatResult:
        self.server.heartbeat_calls.append((lease_id, machine_id))
        if self.server.heartbeat_error is not None:
            raise self.server.heartbeat_error
        return self.server.heartbeat_result

    def release(self, *, lease_id: str) -> LicenseReleaseResult:
        self.server.release_calls.append(lease_id)
        return self.server.release_result


class _FakeServer:
    def __init__(self) -> None:
        self.status = LicenseStatus(ok=True, company_name="Acme", license_type="evaluation", seat_count=2, seats_in_use=1)
        self.checkout_result = LicenseCheckoutResult(
            granted=True,
            lease_id="lease_001",
            heartbeat_interval_seconds=999.0,
            lease_ttl_seconds=120.0,
            grace_seconds=5.0,
            expires_at=datetime(2026, 3, 13, 23, 15, tzinfo=timezone.utc),
            company_name="Acme",
            seat_count=2,
            seats_in_use=1,
        )
        self.heartbeat_result = LicenseHeartbeatResult(
            ok=True,
            expires_at=datetime(2026, 3, 13, 23, 16, tzinfo=timezone.utc),
        )
        self.release_result = LicenseReleaseResult(ok=True)
        self.heartbeat_error: Exception | None = None
        self.checkout_calls: list[object] = []
        self.heartbeat_calls: list[tuple[str, str]] = []
        self.release_calls: list[str] = []

    def factory(self, server_url: str, timeout_seconds: float) -> _FakeLicenseClient:
        assert server_url == "http://license-host:27850"
        assert timeout_seconds == 5.0
        return _FakeLicenseClient(self)


def test_controller_enters_grace_then_requires_reconnect_after_heartbeat_loss() -> None:
    clock = _FrozenClock(datetime(2026, 3, 13, 23, 0, tzinfo=timezone.utc))
    server = _FakeServer()
    controller = LicenseLeaseController(client_factory=server.factory, now=clock.now)

    checked_out = controller.connect_and_checkout("http://license-host:27850")

    assert checked_out.phase == "checked_out"
    assert checked_out.can_start_runs is True

    server.heartbeat_error = LicenseConnectionError("network down")
    warning = controller.send_heartbeat_once()

    assert warning.phase == "heartbeat_warning"
    assert warning.heartbeat_failures == 1
    assert warning.can_start_runs is False

    clock.advance(seconds=6.0)
    required = controller.send_heartbeat_once()

    assert required.phase == "license_required"
    assert required.heartbeat_failures == 2
    assert required.can_start_runs is False

    controller.shutdown()


def test_controller_releases_active_lease_cleanly() -> None:
    clock = _FrozenClock(datetime(2026, 3, 13, 23, 0, tzinfo=timezone.utc))
    server = _FakeServer()
    controller = LicenseLeaseController(client_factory=server.factory, now=clock.now)

    controller.connect_and_checkout("http://license-host:27850")
    released = controller.release()

    assert released.phase == "released"
    assert released.lease_id is None
    assert server.release_calls == ["lease_001"]
