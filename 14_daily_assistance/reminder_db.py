"""SQLite database storage for daily assistance and medication reminders."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
import sqlite3
import time
from typing import Generator, List, Optional

logger = logging.getLogger("NAYRA-ReminderDB")


@dataclass
class ReminderRecord:
    id: int
    title: str
    due_timestamp: float
    category: str = "general"
    is_completed: bool = False
    repeat_interval_sec: float = 0.0
    created_at: float = 0.0


class ReminderDB:
    """Persistent SQLite store for scheduled reminders."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            db_path = Path(__file__).resolve().parents[1] / "data" / "reminders.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.was_recovered: bool = False
        self.last_corrupted_backup_path: Optional[Path] = None
        self.recovery_message: Optional[str] = None
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding an active connection with transaction rollback and cleanup."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        try:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.DatabaseError:
                pass
            yield conn
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reminders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        due_timestamp REAL NOT NULL,
                        category TEXT DEFAULT 'general',
                        is_completed INTEGER DEFAULT 0,
                        repeat_interval_sec REAL DEFAULT 0,
                        created_at REAL NOT NULL
                    )
                    """
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_due ON reminders(due_timestamp, is_completed)")
                conn.commit()
        except sqlite3.DatabaseError as exc:
            logger.error(f"Database corruption or initialization failure detected: {exc}. Attempting recovery backup.")
            backup_path = self.db_path.with_suffix(f".corrupt.{int(time.time())}.bak")
            try:
                if self.db_path.exists():
                    self.db_path.rename(backup_path)
                    logger.warning(f"Corrupted database backed up to {backup_path}")
            except Exception as e:
                logger.error(f"Failed to backup corrupt db file: {e}")
            self.was_recovered = True
            self.last_corrupted_backup_path = backup_path
            self.recovery_message = f"Database corrupted ({exc}). Automatic backup saved at {backup_path.name}."
            # Recreate clean DB
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reminders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        due_timestamp REAL NOT NULL,
                        category TEXT DEFAULT 'general',
                        is_completed INTEGER DEFAULT 0,
                        repeat_interval_sec REAL DEFAULT 0,
                        created_at REAL NOT NULL
                    )
                    """
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_due ON reminders(due_timestamp, is_completed)")
                conn.commit()

    def checkpoint(self) -> bool:
        """Explicitly flush WAL entries into the primary database file."""
        try:
            with self._get_connection() as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                return True
        except sqlite3.DatabaseError as exc:
            logger.error(f"WAL checkpoint failed: {exc}")
            return False

    def get_recovery_status(self) -> dict[str, Any]:
        """Return audit information about database recovery state."""
        return {
            "was_recovered": self.was_recovered,
            "backup_path": str(self.last_corrupted_backup_path) if self.last_corrupted_backup_path else None,
            "message": self.recovery_message,
        }

    def add_reminder(
        self,
        title: str,
        due_timestamp: float,
        category: str = "general",
        repeat_interval_sec: float = 0.0,
    ) -> int:
        """Insert a new reminder with duplicate check and validation."""
        cleaned_title = title.strip()
        if not cleaned_title:
            raise ValueError("Reminder title cannot be empty or whitespace.")
        if due_timestamp <= 0:
            raise ValueError(f"Invalid due_timestamp: {due_timestamp}")

        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Duplicate prevention: check for identical pending reminder within 60s
            cursor.execute(
                """
                SELECT id FROM reminders
                WHERE title = ? AND is_completed = 0 AND ABS(due_timestamp - ?) < 60.0
                """,
                (cleaned_title, due_timestamp),
            )
            existing = cursor.fetchone()
            if existing:
                logger.info(f"Duplicate reminder detected for '{cleaned_title}'. Returning existing ID #{existing[0]}.")
                return existing[0]

            cursor.execute(
                """
                INSERT INTO reminders (title, due_timestamp, category, is_completed, repeat_interval_sec, created_at)
                VALUES (?, ?, ?, 0, ?, ?)
                """,
                (cleaned_title, due_timestamp, category, repeat_interval_sec, now),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_due_reminders(self, now: Optional[float] = None) -> List[ReminderRecord]:
        """Fetch all pending reminders whose due time has arrived."""
        if now is None:
            now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, due_timestamp, category, is_completed, repeat_interval_sec, created_at
                FROM reminders
                WHERE is_completed = 0 AND due_timestamp <= ?
                ORDER BY due_timestamp ASC
                """,
                (now,),
            )
            rows = cursor.fetchall()
            return [
                ReminderRecord(
                    id=row[0],
                    title=row[1],
                    due_timestamp=row[2],
                    category=row[3],
                    is_completed=bool(row[4]),
                    repeat_interval_sec=row[5],
                    created_at=row[6],
                )
                for row in rows
            ]

    def list_active(self) -> List[ReminderRecord]:
        """List all uncompleted reminders."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, due_timestamp, category, is_completed, repeat_interval_sec, created_at
                FROM reminders
                WHERE is_completed = 0
                ORDER BY due_timestamp ASC
                """
            )
            rows = cursor.fetchall()
            return [
                ReminderRecord(
                    id=row[0],
                    title=row[1],
                    due_timestamp=row[2],
                    category=row[3],
                    is_completed=bool(row[4]),
                    repeat_interval_sec=row[5],
                    created_at=row[6],
                )
                for row in rows
            ]

    def mark_completed(self, reminder_id: int) -> bool:
        """Mark a reminder as completed by user or reschedule if repeating."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT repeat_interval_sec, due_timestamp FROM reminders WHERE id = ?", (reminder_id,))
            row = cursor.fetchone()
            if not row:
                return False
            repeat_sec, due_ts = row[0], row[1]
            if repeat_sec > 0:
                next_due = due_ts + repeat_sec
                cursor.execute(
                    "UPDATE reminders SET due_timestamp = ?, is_completed = 0 WHERE id = ?",
                    (next_due, reminder_id),
                )
            else:
                cursor.execute("UPDATE reminders SET is_completed = 1 WHERE id = ?", (reminder_id,))
            conn.commit()
            return True

    def mark_announced(self, reminder_id: int, re_alert_delay_sec: float = 300.0) -> bool:
        """Record that a reminder was spoken. Postpones next alert without marking medication taken."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT repeat_interval_sec FROM reminders WHERE id = ?", (reminder_id,))
            row = cursor.fetchone()
            if not row:
                return False
            repeat_sec = row[0]
            delay = repeat_sec if repeat_sec > 0 else re_alert_delay_sec
            next_due = time.time() + delay
            cursor.execute("UPDATE reminders SET due_timestamp = ? WHERE id = ?", (next_due, reminder_id))
            conn.commit()
            return True

    def snooze(self, reminder_id: int, snooze_seconds: float = 600.0) -> bool:
        """Postpone a reminder by snooze_seconds."""
        now = time.time()
        new_due = now + snooze_seconds
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE reminders SET due_timestamp = ?, is_completed = 0 WHERE id = ?",
                (new_due, reminder_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_reminder(self, reminder_id: int) -> bool:
        """Delete a reminder by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
            conn.commit()
            return cursor.rowcount > 0
