"""Atomic JSON writes for the GUI's small per-user state files.

Deliberately free of heavy imports: the licence client config must stay
importable without torch or Qt.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Replace non-finite floats with None so the result is valid JSON.

    ``json.dumps`` emits the bare tokens ``NaN``, ``Infinity`` and ``-Infinity``
    for those values. RFC 8259 has no such literals: Python's own strict parser
    rejects them, and jq silently reads Infinity as 1.797e308 -- so a "pick the
    best run by best_val_loss" script scores a diverged run as a finite loss
    instead of failing. A diverged or untrained metric is genuinely "no value",
    which is what null means.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def dumps_json(payload: Any, *, indent: int = 2) -> str:
    """``json.dumps`` that cannot emit NaN/Infinity (see :func:`json_safe`)."""

    return json.dumps(json_safe(payload), indent=indent, allow_nan=False)


def write_json_atomically(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as JSON so the file is always either the old or the new content.

    A plain ``write_text`` truncates the target before writing, so a crash, kill
    or full disk part-way through leaves a fragment -- and a fragment where
    ``last_session.json`` should be stopped the GUI from starting. Here the JSON
    goes to a sibling temp file that replaces the target only once complete and
    flushed; whatever fails, the previous file survives intact.
    """

    # Serialize first so an unserializable payload fails before the filesystem is touched.
    text = dumps_json(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Same directory as the target: os.replace is only atomic within one filesystem.
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    return target
