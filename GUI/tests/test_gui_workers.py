"""Tests for the Qt task executor helpers.

These checks focus on lifecycle guarantees that are easy to regress when the GUI
threading model changes: returning results, surfacing errors, and shutting down
background threads cleanly.
"""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QTimer
from PySide6.QtWidgets import QApplication

from xfmr_v2.gui_workers import QtTaskExecutor


def _app() -> QCoreApplication:
    """Return one reusable Qt application for the executor test.

    This must be a QApplication, not a bare QCoreApplication: the instance is a
    process-wide singleton that is never torn down, and a QCoreApplication cannot
    host widgets. Creating one here used to make every GUI test that ran later in
    the same process abort with SIGABRT -- hidden only because this file sorted
    after test_gui_window.py.
    """

    app = QCoreApplication.instance()
    return app if app is not None else QApplication([])


def test_qt_task_executor_stops_thread_after_completion() -> None:
    """The production executor should stop its worker thread after one quick task."""

    app = _app()
    executor = QtTaskExecutor()
    state: dict[str, object] = {"timed_out": False}
    handle = None

    def task(*, progress_callback=None, should_stop=None):
        # This task intentionally does almost nothing; the test is about the thread
        # lifecycle around it rather than any particular business logic.
        return {"ok": True}

    def on_result(result):
        state["result"] = result

    def on_error(message: str, traceback_text: str) -> None:
        state["error"] = (message, traceback_text)

    def on_finished() -> None:
        if handle is not None and handle.thread is not None:
            state["thread_running_on_finish"] = handle.thread.isRunning()
        state["finished"] = True

    handle = executor.start(
        task,
        kwargs={},
        on_progress=lambda payload: None,
        on_result=on_result,
        on_error=on_error,
        on_finished=on_finished,
    )
    QTimer.singleShot(5000, lambda: (state.__setitem__("timed_out", True), app.quit()))
    # Pump events until the task reports finished instead of app.exec(): Qt 6 posts a
    # Quit event when the last top-level window closes, and under pytest no loop is
    # running to consume it, so any earlier GUI test that closed its window would make
    # exec() here return before the worker had run at all.
    import time as _time

    deadline = _time.monotonic() + 5.0
    while "finished" not in state and _time.monotonic() < deadline:
        app.processEvents()
        _time.sleep(0.005)

    assert state["timed_out"] is False
    assert state["result"] == {"ok": True}
    assert "error" not in state
    assert state["finished"] is True
    assert state["thread_running_on_finish"] is False
