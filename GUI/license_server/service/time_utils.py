"""UTC time helpers shared by the license server core."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current time as an aware UTC datetime."""

    return datetime.now(timezone.utc)


def coerce_utc_datetime(value: datetime | str) -> datetime:
    """Convert a datetime or ISO-8601 string into an aware UTC datetime."""

    if isinstance(value, datetime):
        moment = value
    else:
        raw_value = value.strip()
        if raw_value.endswith("Z"):
            raw_value = raw_value[:-1] + "+00:00"
        moment = datetime.fromisoformat(raw_value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def format_utc_datetime(value: datetime | str) -> str:
    """Canonicalize a datetime into the `YYYY-MM-DDTHH:MM:SSZ` format."""

    return coerce_utc_datetime(value).isoformat(timespec="seconds").replace("+00:00", "Z")
