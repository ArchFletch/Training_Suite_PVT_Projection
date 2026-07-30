"""Lease-state controller and heartbeat loop for the desktop GUI."""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Callable

from .client import LicenseClientError, LicenseHttpClient
from .models import LicenseIdentity, LicenseLeaseState, LicenseStatus, normalize_server_url


StateCallback = Callable[[LicenseLeaseState], None]
ClientFactory = Callable[[str, float], LicenseHttpClient]


class LicenseLeaseController:
    """Owns one GUI lease state and the background heartbeat lifecycle."""

    def __init__(
        self,
        *,
        identity: LicenseIdentity | None = None,
        timeout_seconds: float = 5.0,
        client_factory: ClientFactory | None = None,
        state_callback: StateCallback | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.identity = identity or LicenseIdentity()
        self.timeout_seconds = float(timeout_seconds)
        self._client_factory = client_factory or (lambda server_url, timeout: LicenseHttpClient(server_url, timeout_seconds=timeout))
        self._state_callback = state_callback
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._state = LicenseLeaseState(machine_id=self.identity.machine_id)

    @property
    def state(self) -> LicenseLeaseState:
        with self._lock:
            return self._state

    def clear_configuration(self) -> LicenseLeaseState:
        """Stop heartbeats and reset back to the unconfigured state."""

        self._release_active_lease(update_state=False)
        state = LicenseLeaseState(machine_id=self.identity.machine_id)
        self._set_state(state)
        return state

    def connect_and_checkout(self, server_url: str) -> LicenseLeaseState:
        """Validate server connectivity, request a seat, and start heartbeats."""

        normalized_url = normalize_server_url(server_url)
        if not normalized_url:
            return self.clear_configuration()

        self._release_active_lease(update_state=False)
        self._set_state(
            LicenseLeaseState(
                phase="checking",
                badge_text="Checking",
                server_url=normalized_url,
                machine_id=self.identity.machine_id,
                message="Contacting the license server and requesting a seat...",
            )
        )

        client = self._make_client(normalized_url)
        try:
            status = client.get_status()
        except LicenseClientError as exc:
            state = LicenseLeaseState(
                phase="error",
                badge_text="Error",
                server_url=normalized_url,
                machine_id=self.identity.machine_id,
                message=str(exc),
                last_error=str(exc),
            )
            self._set_state(state)
            return state

        try:
            checkout = client.checkout(self.identity)
        except LicenseClientError as exc:
            state = LicenseLeaseState(
                phase="error",
                badge_text="Error",
                server_url=normalized_url,
                machine_id=self.identity.machine_id,
                message=str(exc),
                last_status=status,
                last_error=str(exc),
            )
            self._set_state(state)
            return state

        if not checkout.granted or not checkout.lease_id:
            message = checkout.message or "The license server denied the checkout request."
            state = LicenseLeaseState(
                phase="checkout_denied",
                badge_text="Denied",
                server_url=normalized_url,
                machine_id=self.identity.machine_id,
                message=message,
                last_status=status,
                last_error=message,
            )
            self._set_state(state)
            return state

        state = LicenseLeaseState(
            phase="checked_out",
            badge_text="Checked Out",
            server_url=normalized_url,
            machine_id=self.identity.machine_id,
            message=_checked_out_message(status=status, seat_count=checkout.seat_count, seats_in_use=checkout.seats_in_use),
            last_status=_merge_status(status, seat_count=checkout.seat_count, seats_in_use=checkout.seats_in_use),
            lease_id=checkout.lease_id,
            expires_at=checkout.expires_at,
            heartbeat_interval_seconds=checkout.heartbeat_interval_seconds or 30.0,
            lease_ttl_seconds=checkout.lease_ttl_seconds or 120.0,
            grace_seconds=checkout.grace_seconds or 300.0,
            last_heartbeat_at=self._now(),
        )
        self._set_state(state)
        self._start_heartbeat_thread()
        return state

    def send_heartbeat_once(self) -> LicenseLeaseState:
        """Send one heartbeat immediately and return the updated lease state."""

        snapshot = self.state
        if not snapshot.server_url or not snapshot.lease_id:
            return snapshot

        client = self._make_client(snapshot.server_url)
        try:
            result = client.heartbeat(lease_id=snapshot.lease_id, machine_id=snapshot.machine_id or self.identity.machine_id)
        except LicenseClientError as exc:
            return self._handle_heartbeat_failure(snapshot, str(exc))

        if not result.ok:
            # The server answered and rejected the lease (expired, evicted, or
            # machine mismatch). Unlike a network failure this is authoritative —
            # the seat is gone and may already be in use elsewhere — so fail
            # closed immediately instead of entering the grace window.
            reason = result.message or result.reason_code or "the lease is no longer valid"
            state = replace(
                snapshot,
                phase="license_required",
                badge_text="License Required",
                message=(
                    f"The license server rejected the seat lease ({reason}). "
                    "Finish the current run, then reconnect before starting another."
                ),
                lease_id=None,
                expires_at=None,
                heartbeat_failures=snapshot.heartbeat_failures + 1,
                last_error=reason,
            )
            return self._set_state_for_lease(state, snapshot)

        now = self._now()
        state = replace(
            snapshot,
            phase="checked_out",
            badge_text="Checked Out",
            message=_checked_out_message(status=snapshot.last_status),
            expires_at=result.expires_at or snapshot.expires_at,
            last_heartbeat_at=now,
            heartbeat_failures=0,
            grace_deadline=None,
            last_error=None,
        )
        return self._set_state_for_lease(state, snapshot)

    def release(self) -> LicenseLeaseState:
        """Stop heartbeats and release the current seat when possible."""

        return self._release_active_lease(update_state=True)

    def shutdown(self) -> None:
        """Best-effort cleanup used by GUI shutdown code."""

        try:
            self.release()
        except Exception:
            return

    def _start_heartbeat_thread(self) -> None:
        self._stop_heartbeat_thread()
        stop_event = threading.Event()
        self._heartbeat_stop = stop_event
        # The loop holds its own stop event: reading self._heartbeat_stop from the
        # loop would let an old thread (still in a slow HTTP call when its 1 s
        # join times out) latch onto the replacement event and run forever
        # alongside the new thread.
        thread = threading.Thread(
            target=self._heartbeat_loop, args=(stop_event,), name="license-heartbeat", daemon=True
        )
        self._heartbeat_thread = thread
        thread.start()

    def _heartbeat_loop(self, stop_event: threading.Event) -> None:
        while True:
            snapshot = self.state
            if not snapshot.lease_id:
                return
            wait_seconds = float(snapshot.heartbeat_interval_seconds or 30.0)
            if stop_event.wait(wait_seconds):
                return
            updated = self.send_heartbeat_once()
            if updated.phase == "license_required":
                stop_event.set()
                return

    def _handle_heartbeat_failure(self, snapshot: LicenseLeaseState, error_message: str) -> LicenseLeaseState:
        now = self._now()
        failures = snapshot.heartbeat_failures + 1
        grace_seconds = float(snapshot.grace_seconds or 0.0)
        grace_deadline = snapshot.grace_deadline or (now + timedelta(seconds=grace_seconds) if grace_seconds > 0 else now)

        if grace_seconds <= 0 or now >= grace_deadline:
            state = replace(
                snapshot,
                phase="license_required",
                badge_text="License Required",
                message=(
                    "License heartbeats failed past the grace window. "
                    "Finish the current run, then reconnect before starting another."
                ),
                heartbeat_failures=failures,
                grace_deadline=grace_deadline,
                last_error=error_message,
            )
            return self._set_state_for_lease(state, snapshot)

        remaining_seconds = max(int((grace_deadline - now).total_seconds()), 0)
        state = replace(
            snapshot,
            phase="heartbeat_warning",
            badge_text="Grace",
            message=(
                f"Heartbeat failed ({failures}). The current run can continue, "
                f"but new runs are blocked until the seat is healthy again. Grace expires in {remaining_seconds}s."
            ),
            heartbeat_failures=failures,
            grace_deadline=grace_deadline,
            last_error=error_message,
        )
        return self._set_state_for_lease(state, snapshot)

    def _release_active_lease(self, *, update_state: bool) -> LicenseLeaseState:
        snapshot = self.state
        self._stop_heartbeat_thread()

        released_ok = True
        release_error: str | None = None
        if snapshot.server_url and snapshot.lease_id:
            try:
                result = self._make_client(snapshot.server_url).release(lease_id=snapshot.lease_id)
                released_ok = bool(result.ok)
                if not result.ok:
                    release_error = "The license server did not confirm the release."
            except LicenseClientError as exc:
                released_ok = False
                release_error = str(exc)

        if not update_state:
            return snapshot

        if not snapshot.server_url:
            state = LicenseLeaseState(machine_id=self.identity.machine_id)
        elif released_ok:
            state = replace(
                snapshot,
                phase="released",
                badge_text="Released",
                message="The license seat was released cleanly.",
                lease_id=None,
                expires_at=None,
                heartbeat_failures=0,
                grace_deadline=None,
                last_error=None,
            )
        else:
            state = replace(
                snapshot,
                phase="released",
                badge_text="Released",
                message=f"Release was attempted, but the server could not confirm it: {release_error}",
                lease_id=None,
                expires_at=None,
                heartbeat_failures=0,
                grace_deadline=None,
                last_error=release_error,
            )
        self._set_state(state)
        return state

    def _stop_heartbeat_thread(self) -> None:
        thread = self._heartbeat_thread
        self._heartbeat_thread = None
        self._heartbeat_stop.set()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _make_client(self, server_url: str) -> LicenseHttpClient:
        return self._client_factory(server_url, self.timeout_seconds)

    def _set_state(self, state: LicenseLeaseState) -> None:
        callback = self._state_callback
        with self._lock:
            self._state = state
        if callback is not None:
            callback(state)

    def _set_state_for_lease(self, state: LicenseLeaseState, snapshot: LicenseLeaseState) -> LicenseLeaseState:
        """Apply a heartbeat result only if the lease it was sent for is still current.

        A heartbeat is a snapshot-read, network call, write-back sequence with no
        lock held across the call. If a release, reconnect, or clear happened while
        the request was in flight, writing the result back would resurrect the old
        lease state — so it is discarded instead.
        """
        callback = self._state_callback
        with self._lock:
            if (
                self._state.lease_id != snapshot.lease_id
                or self._state.server_url != snapshot.server_url
            ):
                return self._state
            self._state = state
        if callback is not None:
            callback(state)
        return state


def _checked_out_message(
    *,
    status: LicenseStatus | None,
    seat_count: int | None = None,
    seats_in_use: int | None = None,
) -> str:
    effective_seat_count = seat_count if seat_count is not None else (status.seat_count if status else None)
    effective_seats_in_use = seats_in_use if seats_in_use is not None else (status.seats_in_use if status else None)
    if effective_seat_count is not None and effective_seats_in_use is not None:
        return f"Seat granted. {effective_seats_in_use}/{effective_seat_count} seats are in use."
    return "Seat granted. Background heartbeats are active."


def _merge_status(status: LicenseStatus, *, seat_count: int | None, seats_in_use: int | None) -> LicenseStatus:
    return replace(
        status,
        seat_count=seat_count if seat_count is not None else status.seat_count,
        seats_in_use=seats_in_use if seats_in_use is not None else status.seats_in_use,
    )
