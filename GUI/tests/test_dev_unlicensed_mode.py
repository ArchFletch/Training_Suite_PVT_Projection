"""The development escape from the licence gate, and its containment.

Licensing fails closed on purpose, so a bypass is only safe if a packaged build
cannot honour it. These tests pin both halves: the switch works from a source
checkout, and it is inert everywhere else.
"""

from __future__ import annotations

import sys

import pytest
from PySide6.QtWidgets import QMessageBox

from xfmr_v2 import gui_window as gui_window_module
from xfmr_v2.gui_window import DEV_UNLICENSED_ENV, dev_unlicensed_mode
from xfmr_v2.gui_workers import ImmediateTaskExecutor


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_truthy_values_enable_the_bypass_in_a_source_checkout(value: str) -> None:
    assert dev_unlicensed_mode({DEV_UNLICENSED_ENV: value}) is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe", " "])
def test_anything_else_leaves_the_gate_closed(value: str) -> None:
    assert dev_unlicensed_mode({DEV_UNLICENSED_ENV: value}) is False


def test_unset_leaves_the_gate_closed() -> None:
    assert dev_unlicensed_mode({}) is False


def test_a_packaged_build_ignores_the_bypass() -> None:
    """The containment that makes this safe to ship.

    Shipping fail-closed was a deliberate decision. A switch a customer could set
    would undo it, so the packaged runtime refuses it no matter how it is spelled.
    """
    for value in ("1", "true", "yes", "on"):
        assert (
            dev_unlicensed_mode({DEV_UNLICENSED_ENV: value, "MLP_APP_PATH_MODE": "packaged"})
            is False
        ), f"a packaged build honoured {DEV_UNLICENSED_ENV}={value}"


def test_a_frozen_build_ignores_the_bypass_even_when_told_it_is_a_source_checkout(monkeypatch) -> None:
    """The hole the packaged-build test above did not cover.

    It only ever proved that MODE=packaged closes the gate. The path policy used
    to honour MODE=source ahead of the freeze markers, so a customer with two
    environment variables reopened the gate in a shipped build. The freeze
    markers must win.
    """
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    for value in ("1", "true", "yes", "on"):
        assert (
            dev_unlicensed_mode({DEV_UNLICENSED_ENV: value, "MLP_APP_PATH_MODE": "source"})
            is False
        ), f"a frozen build honoured {DEV_UNLICENSED_ENV}={value} via MLP_APP_PATH_MODE=source"


def _window(qtbot, monkeypatch):
    monkeypatch.setattr(
        gui_window_module,
        "list_available_devices",
        lambda: [{"id": "cpu", "label": "cpu — CPU"}],
    )
    monkeypatch.setattr(gui_window_module, "load_last_session", lambda: None)
    monkeypatch.setattr(gui_window_module, "save_last_session", lambda payload: None)
    monkeypatch.setattr(
        gui_window_module.QMessageBox, "warning",
        lambda *a, **k: QMessageBox.StandardButton.Ok,
    )
    window = gui_window_module.MlpTrainingStudio(executor=ImmediateTaskExecutor())
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(20)
    return window


def test_the_bypass_opens_the_run_gate_and_labels_the_session(qtbot, monkeypatch) -> None:
    monkeypatch.setenv(DEV_UNLICENSED_ENV, "1")
    window = _window(qtbot, monkeypatch)

    assert window._license_allows_new_runs()
    assert window.start_baseline_button.isEnabled()
    assert window.start_baseline_button.toolTip() == ""
    # A bypassed session must never be mistakable for a licensed one.
    assert "licence check disabled" in window.windowTitle()
    assert window.license_seat_state_badge.text() == "Dev Bypass"
    assert DEV_UNLICENSED_ENV in window.license_status_text.text()
    assert "licence check is disabled" in window.run_log_text_edit.toPlainText()


def test_without_the_bypass_the_gate_stays_closed(qtbot, monkeypatch) -> None:
    monkeypatch.delenv(DEV_UNLICENSED_ENV, raising=False)
    window = _window(qtbot, monkeypatch)

    assert not window._license_allows_new_runs()
    assert not window.start_baseline_button.isEnabled()
    assert "licence check disabled" not in window.windowTitle()
    assert window.license_seat_state_badge.text() != "Dev Bypass"


def test_a_packaged_build_keeps_the_gate_closed_with_the_switch_set(qtbot, monkeypatch) -> None:
    monkeypatch.setenv(DEV_UNLICENSED_ENV, "1")
    monkeypatch.setenv("MLP_APP_PATH_MODE", "packaged")
    window = _window(qtbot, monkeypatch)

    assert not window._license_allows_new_runs()
    assert not window.start_baseline_button.isEnabled()
    assert "licence check disabled" not in window.windowTitle()


def test_a_frozen_build_keeps_the_gate_closed_when_told_it_is_a_source_checkout(qtbot, monkeypatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv(DEV_UNLICENSED_ENV, "1")
    monkeypatch.setenv("MLP_APP_PATH_MODE", "source")
    window = _window(qtbot, monkeypatch)

    assert not window._license_allows_new_runs()
    assert not window.start_baseline_button.isEnabled()
    assert "licence check disabled" not in window.windowTitle()
    assert window.license_seat_state_badge.text() != "Dev Bypass"
