"""Shared progress event helpers for long-running GUI-facing operations.

The rest of the codebase performs long-running work such as:
- reading large datasets
- training models
- running multi-trial searches

Those workflows optionally report structured progress dictionaries so a GUI,
worker thread, or notebook can react without parsing free-form log strings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable

ProgressCallback = Callable[[dict[str, Any]], None]
StopChecker = Callable[[], bool]
# These aliases keep the long-running modules readable by naming the callback
# contracts once instead of repeating the full `Callable[...]` spellings.


class RunCancelled(RuntimeError):
    """Raised when a long-running operation is asked to stop.

    Using a dedicated exception type makes it possible to distinguish an expected
    user stop request from an actual training or data-processing failure.
    """


@dataclass(slots=True)
class ProgressEvent:
    """Structured event payload suitable for GUI worker bridges.

    Using a dataclass here keeps event construction explicit and consistent.
    The fields map directly to what downstream consumers expect to receive.
    """

    # `event` is the machine-readable event name, for example `started` or `epoch_end`.
    event: str
    # `phase` groups events by workflow, such as `baseline`, `transfer`, or `search`.
    phase: str
    # `message` is optional human-readable text that a GUI can display directly.
    message: str | None = None
    # `data` holds all workflow-specific fields without forcing every event to share
    # the same schema.
    data: dict[str, Any] = field(default_factory=dict)
    # Timestamping here avoids duplicating "what time is it?" logic throughout the repo.
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        # Convert the dataclass into a plain dictionary because callbacks and JSON
        # serializers usually work with built-in Python containers.
        payload = asdict(self)
        if self.message is None:
            payload.pop("message")
        if not self.data:
            # Drop empty optional fields so the emitted payload stays compact.
            payload.pop("data")
        return payload


def emit_progress(
    callback: ProgressCallback | None,
    event: str,
    phase: str,
    message: str | None = None,
    **data: Any,
) -> dict[str, Any]:
    """Build and emit one progress event when a callback is present."""
    # Centralizing event construction in one helper keeps every caller consistent.
    # It also makes it easy to reuse the same payload whether or not a callback exists.
    payload = ProgressEvent(event=event, phase=phase, message=message, data=data).to_dict()
    if callback is not None:
        callback(payload)
    return payload


def request_stop(should_stop: StopChecker | None) -> None:
    """Raise ``RunCancelled`` when a stop was requested."""
    # Callers sprinkle this check inside long loops. Raising an exception lets the
    # code unwind naturally and collect a partial summary instead of crashing abruptly.
    if should_stop is not None and should_stop():
        raise RunCancelled("Run cancelled by user.")
