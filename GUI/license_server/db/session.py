"""SQLite connection and transaction helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator

from .models import SCHEMA_STATEMENTS


class SQLiteSessionFactory:
    """Create short-lived SQLite sessions with predictable transaction behavior."""

    def __init__(self, database_path: Path | str, *, timeout_seconds: float = 30.0) -> None:
        self.database_path = Path(database_path)
        self.timeout_seconds = timeout_seconds

    def initialize_database(self) -> None:
        """Create the SQLite file and required tables if they do not exist yet."""

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.session(write_lock=True) as connection:
            for statement in SCHEMA_STATEMENTS:
                connection.execute(statement)

    @contextmanager
    def session(self, *, write_lock: bool = False) -> Iterator[sqlite3.Connection]:
        """Yield a transaction-scoped SQLite connection."""

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.database_path,
            timeout=self.timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("BEGIN IMMEDIATE" if write_lock else "BEGIN")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
