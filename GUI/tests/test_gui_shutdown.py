"""Shutdown paths that have to release the floating seat.

Closing the window has always released it. A signal did not: Qt's event loop
sits in C++, so a Python-level handler only runs once the interpreter regains
control, and the default SIGTERM disposition kills the process first. An IT
`kill`, a logout, or a shutdown script therefore stranded a seat for the full
lease TTL. These tests drive the real handler, so they send a real signal to the
test process -- which is safe only because the handler replaces the disposition
that would otherwise end the run.
"""

from __future__ import annotations

import os
import signal

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

import xfmr_v2.gui_window as gui_window_module
from xfmr_v2.app_paths import APP_NAME, APP_SLUG
from xfmr_v2.gui_workers import ImmediateTaskExecutor


@pytest.fixture
def restored_signal_state():
    """Undo the process-global changes install_termination_handler makes.

    `set_wakeup_fd` and the SIGTERM/SIGINT dispositions outlive the test. Left
    installed, a later test's signal would fire a notifier reading from a closed
    socket, and pytest's own Ctrl-C handling would be gone.
    """

    previous_handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    notifiers = []
    yield notifiers
    for notifier in notifiers:
        if notifier is not None:
            notifier.setEnabled(False)
    signal.set_wakeup_fd(-1)
    for number, handler in previous_handlers.items():
        signal.signal(number, handler)


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr(
        gui_window_module, "list_available_devices", lambda: [{"id": "cpu", "label": "cpu — CPU"}]
    )
    monkeypatch.setattr(gui_window_module, "load_last_session", lambda: None)
    monkeypatch.setattr(gui_window_module, "save_last_session", lambda payload: None)
    monkeypatch.setattr(
        gui_window_module.QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok
    )
    widget = gui_window_module.MlpTrainingStudio(executor=ImmediateTaskExecutor())
    qtbot.addWidget(widget)
    widget.show()
    qtbot.wait(20)
    return widget


def test_sigterm_runs_the_same_shutdown_as_closing_the_window(
    qtbot, monkeypatch, window, restored_signal_state
) -> None:
    """The seat release lives in closeEvent, so the signal has to reach it."""

    released: list[str] = []
    monkeypatch.setattr(window.license_controller, "shutdown", lambda: released.append("shutdown"))

    notifier = gui_window_module.install_termination_handler(QApplication.instance(), window)
    assert notifier is not None, "no handler installed; sending SIGTERM would kill the test run"
    restored_signal_state.append(notifier)

    os.kill(os.getpid(), signal.SIGTERM)
    qtbot.waitUntil(lambda: bool(released), timeout=5000)

    assert released == ["shutdown"]
    assert not window.isVisible()


def test_sigint_takes_the_same_path(qtbot, monkeypatch, window, restored_signal_state) -> None:
    """Ctrl-C in a terminal-launched session should not strand a seat either."""

    released: list[str] = []
    monkeypatch.setattr(window.license_controller, "shutdown", lambda: released.append("shutdown"))

    notifier = gui_window_module.install_termination_handler(QApplication.instance(), window)
    assert notifier is not None
    restored_signal_state.append(notifier)

    os.kill(os.getpid(), signal.SIGINT)
    qtbot.waitUntil(lambda: bool(released), timeout=5000)

    assert released == ["shutdown"]


def test_a_signal_arriving_during_a_deferred_close_does_not_re_enter(
    qtbot, monkeypatch, window, restored_signal_state
) -> None:
    """A task still running defers the close. A second signal at that point must
    not start the close again -- the window is already stopping the task, and
    re-entering closeEvent is how the close deadlock was reachable before."""

    closes: list[str] = []
    monkeypatch.setattr(window, "close", lambda: closes.append("close"))

    notifier = gui_window_module.install_termination_handler(QApplication.instance(), window)
    assert notifier is not None
    restored_signal_state.append(notifier)

    os.kill(os.getpid(), signal.SIGTERM)
    qtbot.waitUntil(lambda: bool(closes), timeout=5000)
    os.kill(os.getpid(), signal.SIGTERM)
    qtbot.wait(300)

    assert closes == ["close"], "the second signal re-entered the close path"


def test_the_handler_declines_rather_than_half_installs_off_the_main_thread(monkeypatch) -> None:
    """signal.signal is main-thread only. A refusal has to leave the previous
    behaviour intact, not a live notifier with no signals wired to it."""

    def refuse(*args, **kwargs):
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(signal, "set_wakeup_fd", refuse)
    assert gui_window_module.install_termination_handler(QApplication.instance(), object()) is None


# --------------------------------------------------------------------------- #
# desktop identity
# --------------------------------------------------------------------------- #


def test_the_application_identity_matches_the_desktop_entry(qtbot) -> None:
    """Qt builds WM_CLASS from the desktop file name, and the launcher entry's
    StartupWMClass is what associates the running window with it. Verified
    against a real X server: WM_CLASS came out ("gui_window.py", "Python") and
    is now ("mlp-training-studio", "mlp-training-studio")."""

    app = QApplication.instance()
    gui_window_module.apply_application_identity(app)

    assert app.applicationName() == APP_SLUG
    assert app.applicationDisplayName() == APP_NAME
    assert app.desktopFileName() == APP_SLUG


def test_the_desktop_entry_declares_the_matching_startup_wm_class() -> None:
    from pathlib import Path

    entry = Path(__file__).resolve().parents[1] / "packaging" / "linux" / "mlp-training-studio.desktop.in"
    lines = [line.strip() for line in entry.read_text(encoding="utf-8").splitlines()]
    assert f"StartupWMClass={APP_SLUG}" in lines
    # The desktop file's own basename is the other half of the association.
    assert entry.name == f"{APP_SLUG}.desktop.in"
