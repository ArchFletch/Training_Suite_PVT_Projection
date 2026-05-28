from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QMessageBox

import xfmr_v2.gui_window as gui_window_module
from xfmr_v2.gui_workers import ImmediateTaskExecutor
from xfmr_v2.licensing import LicenseLeaseState, LicenseStatus


class _StubLicenseController:
    def __init__(self, *, state_callback=None, checkout_state: LicenseLeaseState | None = None) -> None:
        self._state_callback = state_callback
        self._checkout_state = checkout_state or LicenseLeaseState()
        self.state = LicenseLeaseState()
        self.shutdown_calls = 0

    def connect_and_checkout(self, server_url: str) -> LicenseLeaseState:
        self.state = self._checkout_state if self._checkout_state.server_url else LicenseLeaseState(
            phase=self._checkout_state.phase,
            badge_text=self._checkout_state.badge_text,
            server_url=server_url,
            message=self._checkout_state.message,
            last_status=self._checkout_state.last_status,
            lease_id=self._checkout_state.lease_id,
            machine_id=self._checkout_state.machine_id,
            expires_at=self._checkout_state.expires_at,
            heartbeat_interval_seconds=self._checkout_state.heartbeat_interval_seconds,
            lease_ttl_seconds=self._checkout_state.lease_ttl_seconds,
            grace_seconds=self._checkout_state.grace_seconds,
            grace_deadline=self._checkout_state.grace_deadline,
            last_heartbeat_at=self._checkout_state.last_heartbeat_at,
            heartbeat_failures=self._checkout_state.heartbeat_failures,
            last_error=self._checkout_state.last_error,
        )
        if self._state_callback is not None:
            self._state_callback(self.state)
        return self.state

    def clear_configuration(self) -> LicenseLeaseState:
        self.state = LicenseLeaseState()
        if self._state_callback is not None:
            self._state_callback(self.state)
        return self.state

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _fake_list_available_devices():
    return [
        {"id": "cuda:0", "label": "cuda:0 — Synthetic GPU (16.0 GB)"},
        {"id": "cpu", "label": "cpu — CPU"},
    ]


def _build_window(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    *,
    checkout_state: LicenseLeaseState | None = None,
    saved_session: dict[str, object] | None = None,
    status_result: LicenseStatus | None = None,
):
    controller_instances: list[_StubLicenseController] = []

    def controller_factory(*, state_callback=None):
        controller = _StubLicenseController(state_callback=state_callback, checkout_state=checkout_state)
        controller_instances.append(controller)
        return controller

    monkeypatch.setattr(gui_window_module, "list_available_devices", _fake_list_available_devices)
    monkeypatch.setattr(gui_window_module, "load_last_session", lambda: saved_session)
    monkeypatch.setattr(gui_window_module, "save_last_session", lambda payload: None)
    monkeypatch.setattr(gui_window_module.QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(gui_window_module, "LicenseLeaseController", controller_factory)
    if status_result is not None:
        monkeypatch.setattr(gui_window_module, "test_license_connection", lambda server_url: status_result)

    window = gui_window_module.MlpTrainingStudio(executor=ImmediateTaskExecutor())
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(20)
    return window, controller_instances


def test_saved_license_url_checkout_denial_blocks_training(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    denied_state = LicenseLeaseState(
        phase="checkout_denied",
        badge_text="Denied",
        server_url="http://license-host:27850",
        message="All floating seats are currently in use.",
    )
    window, _ = _build_window(
        qtbot,
        monkeypatch,
        checkout_state=denied_state,
        saved_session={"licensing": {"server_url": "http://license-host:27850"}},
    )

    assert window.license_server_url_edit.text() == "http://license-host:27850"
    assert window.license_seat_state_badge.text() == "Denied"
    assert "currently in use" in window.license_status_text.text()
    assert not window.start_baseline_button.isEnabled()
    assert not window.start_transfer_button.isEnabled()


def test_saved_license_url_checkout_success_keeps_training_enabled(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    status = LicenseStatus(ok=True, company_name="Acme", license_type="evaluation", seat_count=2, seats_in_use=1)
    checked_out_state = LicenseLeaseState(
        phase="checked_out",
        badge_text="Checked Out",
        server_url="http://license-host:27850",
        message="Seat granted. 1/2 seats are in use.",
        last_status=status,
        lease_id="lease_001",
    )
    window, _ = _build_window(
        qtbot,
        monkeypatch,
        checkout_state=checked_out_state,
        saved_session={"licensing": {"server_url": "http://license-host:27850"}},
    )

    assert window.license_seat_state_badge.text() == "Checked Out"
    assert window.license_company_value.text() == "Acme"
    assert window.start_baseline_button.isEnabled()
    assert window.start_transfer_button.isEnabled()


def test_license_connection_test_updates_summary_without_checkout(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    status = LicenseStatus(ok=True, company_name="Acme", license_type="evaluation", seat_count=2, seats_in_use=1)
    window, _ = _build_window(qtbot, monkeypatch, status_result=status)

    window.license_server_url_edit.setText("http://license-host:27850")
    window.run_license_connection_test()

    assert window.license_server_status_badge.text() == "Connected"
    assert window.license_company_value.text() == "Acme"
    assert window.license_seat_usage_value.text() == "1/2 seats in use"
    assert window.collect_config_payload()["licensing"]["server_url"] == "http://license-host:27850"
