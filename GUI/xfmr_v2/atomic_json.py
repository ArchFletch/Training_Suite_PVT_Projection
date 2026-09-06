"""Atomic JSON writes for the GUI's small per-user state files.

Deliberately free of heavy imports: the licence client config must stay
importable without torch or Qt.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomically(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as JSON so the file is always either the old or the new content.

    A plain ``write_text`` truncates the target before writing, so a crash, kill
    or full disk part-way through leaves a fragment -- and a fragment where
    ``last_session.json`` should be stopped the GUI from starting. Here the JSON
    goes to a sibling temp file that replaces the target only once complete and
    flushed; whatever fails, the previous file survives intact.
    """

    # Serialize first so an unserializable payload fails before the filesystem is touched.
    text = json.dumps(payload, indent=2)
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
