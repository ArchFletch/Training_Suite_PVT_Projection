"""SQLite schema definitions for the license server core."""

from __future__ import annotations


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS server_identity (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        schema_version INTEGER NOT NULL,
        product TEXT NOT NULL,
        server_id TEXT NOT NULL,
        host_fingerprint TEXT NOT NULL,
        hostname TEXT NOT NULL,
        os_family TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS active_license (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        license_id TEXT NOT NULL,
        license_type TEXT NOT NULL,
        company_name TEXT NOT NULL,
        product TEXT NOT NULL,
        server_id TEXT NOT NULL,
        host_fingerprint TEXT NOT NULL,
        seat_count INTEGER NOT NULL CHECK (seat_count > 0),
        features_json TEXT NOT NULL,
        starts_at TEXT NOT NULL,
        ends_at TEXT NOT NULL,
        algorithm TEXT NOT NULL,
        key_id TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        signature TEXT NOT NULL,
        imported_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS seat_leases (
        lease_id TEXT PRIMARY KEY,
        machine_id TEXT NOT NULL,
        hostname TEXT NOT NULL,
        username TEXT,
        platform TEXT NOT NULL,
        product_version TEXT NOT NULL,
        checked_out_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        released_at TEXT,
        release_reason TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        event_time TEXT NOT NULL,
        details_json TEXT NOT NULL
    )
    """,
)
