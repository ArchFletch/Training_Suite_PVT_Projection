"""Qt task runners used by the GUI.

The GUI needs to run long operations such as search and training outside the
main thread while still receiving progress events. This module provides a small
executor abstraction with two implementations:

- `QtTaskExecutor` for the real application
- `ImmediateTaskExecutor` for deterministic GUI tests
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from PySide6.QtCore import QObject, QMetaObject, QThread, Qt, Signal, Slot

from .progress import RunCancelled


TaskFunction = Callable[..., Any]
ProgressHandler = Callable[[dict[str, Any]], None]
ResultHandler = Callable[[Any], None]
ErrorHandler = Callable[[str, str], None]
FinishedHandler = Callable[[], None]


@dataclass
class TaskHandle:
    """Light wrapper around one running task.

    The GUI keeps this handle so it can request cancellation and inspect the
    backing thread during shutdown or tests.
    """

    stop: Callable[[], None]
    thread: QThread | None = None
    runner: QObject | None = None
    completed: bool = False


class _TaskSignals(QObject):
    """Qt signal bundle shared by both production and test task execution paths."""

    progress = Signal(object)
    result = Signal(object)
    error = Signal(str, str)
    finished = Signal()


class _TaskRunner(QObject):
    """Small QObject wrapper that adapts a plain Python function into a Qt worker."""

    def __init__(self, function: TaskFunction, kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.function = function
        self.kwargs = kwargs
        self.signals = _TaskSignals(self)
        self._stop_event = threading.Event()

    @Slot()
    def run(self) -> None:
        # Inject the stop/progress hooks here so call sites can pass normal Python
        # functions without knowing anything about Qt signals or threads.
        try:
            result = self.function(
                **self.kwargs,
                progress_callback=self.signals.progress.emit,
                should_stop=self._stop_event.is_set,
            )
        except RunCancelled:
            # A user-requested stop is a normal outcome, not an error: report it
            # the same way runner-level stops are reported.
            self.signals.result.emit({"status": "stopped"})
        except Exception as exc:  # pragma: no cover - exercised through GUI tests indirectly
            self.signals.error.emit(str(exc), traceback.format_exc())
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

    def stop(self) -> None:
        # A thread-safe event is enough for the current workloads because the long
        # loops periodically poll `should_stop` and unwind on their own.
        self._stop_event.set()


class QtTaskExecutor:
    """Real task executor that runs one function on a dedicated `QThread`."""

    def start(
        self,
        function: TaskFunction,
        *,
        kwargs: dict[str, Any],
        on_progress: ProgressHandler,
        on_result: ResultHandler,
        on_error: ErrorHandler,
        on_finished: FinishedHandler,
    ) -> TaskHandle:
        # Each submitted task receives a dedicated `QThread`. That keeps the
        # lifecycle simple and avoids cross-run state leaking between jobs.
        thread = QThread()
        runner = _TaskRunner(function, kwargs)
        runner.moveToThread(thread)
        runner.signals.progress.connect(on_progress)
        runner.signals.result.connect(on_result)
        runner.signals.error.connect(on_error)
        runner.signals.finished.connect(thread.quit)
        runner.signals.finished.connect(runner.deleteLater)
        thread.finished.connect(on_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        QMetaObject.invokeMethod(runner, "run", Qt.ConnectionType.QueuedConnection)

        def stop() -> None:
            runner.stop()

        return TaskHandle(stop=stop, thread=thread, runner=runner, completed=False)


class ImmediateTaskExecutor:
    """Test executor that runs tasks synchronously in the calling thread."""

    def start(
        self,
        function: TaskFunction,
        *,
        kwargs: dict[str, Any],
        on_progress: ProgressHandler,
        on_result: ResultHandler,
        on_error: ErrorHandler,
        on_finished: FinishedHandler,
    ) -> TaskHandle:
        # The synchronous executor mirrors the production callback contract so the
        # GUI tests can exercise the same code paths without real background threads.
        stop_event = threading.Event()
        try:
            result = function(
                **kwargs,
                progress_callback=on_progress,
                should_stop=stop_event.is_set,
            )
        except RunCancelled:
            on_result({"status": "stopped"})
        except Exception as exc:  # pragma: no cover - intentionally mirrors production code
            on_error(str(exc), traceback.format_exc())
        else:
            on_result(result)
        finally:
            on_finished()
        return TaskHandle(stop=stop_event.set, thread=None, runner=None, completed=True)
