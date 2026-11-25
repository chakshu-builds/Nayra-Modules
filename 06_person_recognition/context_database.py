"""Personal Context Database for NAYRA Module 06.

Stores normalized user-provided contextual information linked to biometric profile IDs
in a dedicated SQLite database (data/person_context.db).
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from config.settings import PersonRecognitionSettings, get_default_person_recognition_settings
from recognition_types import FactItem, PersonContext, PreferenceItem

if TYPE_CHECKING:
    from face_database import FaceDatabase

logger = logging.getLogger(__name__)


class PersonContextDatabase:
    """Manages user-provided relational and biographical context linked to biometric profile IDs."""

    def __init__(self, settings: PersonRecognitionSettings | None = None, db_path: str | Path | None = None) -> None:
        self.settings = settings or get_default_person_recognition_settings()
        self.db_path = Path(db_path or self.settings.context_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with WAL mode and foreign key constraints enabled."""
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            return conn
        except sqlite3.DatabaseError as err:
            logger.error("Database connection failed or file is corrupted (%s). Attempting recovery...", err)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            import gc
            gc.collect()
            self._handle_corrupted_db()
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            return conn

    def _handle_corrupted_db(self) -> None:
        """Move corrupted database aside or truncate and start a fresh instance."""
        if self.db_path.exists():
            backup_path = self.db_path.with_name(f"{self.db_path.stem}_corrupted_{int(time.time())}.db")
            moved = False
            try:
                shutil.move(str(self.db_path), str(backup_path))
                logger.warning("Moved corrupted context database to %s", backup_path)
                moved = True
            except Exception as e:
                logger.warning("Failed to move corrupted database (%s). Truncating file directly.", e)

            if not moved:
                try:
                    self.db_path.unlink(missing_ok=True)
                except Exception:
                    try:
                        self.db_path.write_bytes(b"")
                    except Exception as wr_err:
                        logger.error("Failed to truncate corrupted database: %s", wr_err)

    def _init_db(self) -> None:
        """Initialize tables, foreign key cascades, and indexes."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS person_context (
                    profile_id TEXT PRIMARY KEY,
                    relationship TEXT NOT NULL DEFAULT '',
                    occupation TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS person_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(profile_id) REFERENCES person_context(profile_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS person_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id TEXT NOT NULL,
                    preference TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(profile_id) REFERENCES person_context(profile_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_facts_profile_id ON person_facts(profile_id);
                CREATE INDEX IF NOT EXISTS idx_preferences_profile_id ON person_preferences(profile_id);
                """
            )

    def _verify_profile_exists(self, profile_id: str, face_db: FaceDatabase | None) -> None:
        """Validate that the referenced profile_id exists in the biometric database."""
        if face_db is not None:
            if face_db.get_profile(profile_id) is None:
                raise ValueError(
                    f"Profile ID '{profile_id}' does not exist in FaceDatabase. "
                    f"Context can only be linked to an enrolled familiar person."
                )

    def set_context(
        self,
        profile_id: str,
        relationship: str = "",
        occupation: str = "",
        description: str = "",
        notes: str = "",
        face_db: FaceDatabase | None = None,
    ) -> PersonContext:
        """Create or update primary context attributes for an enrolled person."""
        clean_id = profile_id.strip()
        if not clean_id:
            raise ValueError("profile_id cannot be empty.")

        self._verify_profile_exists(clean_id, face_db)

        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO person_context (profile_id, relationship, occupation, description, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    relationship = excluded.relationship,
                    occupation = excluded.occupation,
                    description = excluded.description,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at;
                """,
                (
                    clean_id,
                    relationship.strip(),
                    occupation.strip(),
                    description.strip(),
                    notes.strip(),
                    now,
                    now,
                ),
            )

        context = self.get_context(clean_id, face_db=face_db)
        if context is None:
            raise RuntimeError(f"Failed to retrieve context after saving for {clean_id}")
        return context

    def update_profile_details(
        self,
        profile_id: str,
        relationship: str | None = None,
        occupation: str | None = None,
        description: str | None = None,
        notes: str | None = None,
        face_db: FaceDatabase | None = None,
    ) -> PersonContext:
        """Partially update non-null details of an existing person context."""
        clean_id = profile_id.strip()
        self._verify_profile_exists(clean_id, face_db)

        existing = self.get_context(clean_id, face_db=face_db)
        if existing is None:
            return self.set_context(
                profile_id=clean_id,
                relationship=relationship or "",
                occupation=occupation or "",
                description=description or "",
                notes=notes or "",
                face_db=face_db,
            )

        new_rel = relationship.strip() if relationship is not None else existing.relationship
        new_occ = occupation.strip() if occupation is not None else existing.occupation
        new_desc = description.strip() if description is not None else existing.description
        new_notes = notes.strip() if notes is not None else existing.notes

        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE person_context
                SET relationship = ?, occupation = ?, description = ?, notes = ?, updated_at = ?
                WHERE profile_id = ?;
                """,
                (new_rel, new_occ, new_desc, new_notes, now, clean_id),
            )

        res = self.get_context(clean_id, face_db=face_db)
        if res is None:
            raise RuntimeError(f"Failed to fetch updated context for {clean_id}")
        return res

    def get_context(self, profile_id: str, face_db: FaceDatabase | None = None) -> PersonContext | None:
        """Retrieve full personal context including facts and preferences for a profile_id."""
        clean_id = profile_id.strip()

        # If face_db is supplied and profile has been deleted from biometric DB, clean up orphan
        if face_db is not None and face_db.get_profile(clean_id) is None:
            self.delete_context(clean_id)
            return None

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM person_context WHERE profile_id = ?;",
                (clean_id,),
            ).fetchone()

            if not row:
                return None

            facts_rows = conn.execute(
                "SELECT id, fact, category, created_at FROM person_facts WHERE profile_id = ? ORDER BY id ASC;",
                (clean_id,),
            ).fetchall()

            facts = [
                FactItem(
                    id=f["id"],
                    fact=f["fact"],
                    category=f["category"],
                    created_at=f["created_at"],
                )
                for f in facts_rows
            ]

            pref_rows = conn.execute(
                "SELECT id, preference, category, created_at FROM person_preferences WHERE profile_id = ? ORDER BY id ASC;",
                (clean_id,),
            ).fetchall()

            preferences = [
                PreferenceItem(
                    id=p["id"],
                    preference=p["preference"],
                    category=p["category"],
                    created_at=p["created_at"],
                )
                for p in pref_rows
            ]

            return PersonContext(
                profile_id=row["profile_id"],
                relationship=row["relationship"],
                occupation=row["occupation"],
                description=row["description"],
                notes=row["notes"],
                facts=facts,
                preferences=preferences,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def add_fact(
        self,
        profile_id: str,
        fact: str,
        category: str = "general",
        face_db: FaceDatabase | None = None,
    ) -> FactItem:
        """Add an important fact about an enrolled person."""
        clean_id = profile_id.strip()
        clean_fact = fact.strip()
        if not clean_fact:
            raise ValueError("Fact text cannot be empty.")

        self._verify_profile_exists(clean_id, face_db)

        # Ensure parent person_context entry exists
        if self.get_context(clean_id, face_db=face_db) is None:
            self.set_context(clean_id, face_db=face_db)

        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO person_facts (profile_id, fact, category, created_at)
                VALUES (?, ?, ?, ?);
                """,
                (clean_id, clean_fact, category.strip(), now),
            )
            fact_id = cursor.lastrowid
            conn.execute(
                "UPDATE person_context SET updated_at = ? WHERE profile_id = ?;",
                (now, clean_id),
            )

        if fact_id is None:
            raise RuntimeError("Failed to insert fact record.")

        return FactItem(id=fact_id, fact=clean_fact, category=category.strip(), created_at=now)

    def remove_fact(self, fact_id: int) -> bool:
        """Remove a single fact by its ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM person_facts WHERE id = ?;", (fact_id,))
            return cursor.rowcount > 0

    def add_preference(
        self,
        profile_id: str,
        preference: str,
        category: str = "general",
        face_db: FaceDatabase | None = None,
    ) -> PreferenceItem:
        """Add a personal preference for an enrolled person."""
        clean_id = profile_id.strip()
        clean_pref = preference.strip()
        if not clean_pref:
            raise ValueError("Preference text cannot be empty.")

        self._verify_profile_exists(clean_id, face_db)

        # Ensure parent person_context entry exists
        if self.get_context(clean_id, face_db=face_db) is None:
            self.set_context(clean_id, face_db=face_db)

        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO person_preferences (profile_id, preference, category, created_at)
                VALUES (?, ?, ?, ?);
                """,
                (clean_id, clean_pref, category.strip(), now),
            )
            pref_id = cursor.lastrowid
            conn.execute(
                "UPDATE person_context SET updated_at = ? WHERE profile_id = ?;",
                (now, clean_id),
            )

        if pref_id is None:
            raise RuntimeError("Failed to insert preference record.")

        return PreferenceItem(id=pref_id, preference=clean_pref, category=category.strip(), created_at=now)

    def remove_preference(self, preference_id: int) -> bool:
        """Remove a single preference by its ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM person_preferences WHERE id = ?;", (preference_id,))
            return cursor.rowcount > 0

    def list_all_contexts(self) -> list[PersonContext]:
        """List all stored person contexts with facts and preferences."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT profile_id FROM person_context ORDER BY created_at ASC;").fetchall()
            contexts: list[PersonContext] = []
            for r in rows:
                c = self.get_context(r["profile_id"])
                if c is not None:
                    contexts.append(c)
            return contexts

    def delete_context(self, profile_id: str) -> bool:
        """Delete all contextual information, facts, and preferences for a profile_id."""
        clean_id = profile_id.strip()
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM person_context WHERE profile_id = ?;", (clean_id,))
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info("Deleted personal context for profile ID %s", clean_id)
            return deleted

    def prune_orphans(self, valid_profile_ids: set[str]) -> int:
        """Prune any context records referencing profile IDs that no longer exist in FaceDatabase."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT profile_id FROM person_context;").fetchall()
            orphans = [r["profile_id"] for r in rows if r["profile_id"] not in valid_profile_ids]
            if orphans:
                conn.executemany(
                    "DELETE FROM person_context WHERE profile_id = ?;",
                    [(pid,) for pid in orphans],
                )
                logger.info("Pruned %d orphaned person context records.", len(orphans))
            return len(orphans)

    def sync_with_face_db(self, face_db: FaceDatabase) -> int:
        """Convenience method to prune context records for any profile deleted from biometric database."""
        valid_ids = {p.profile_id for p in face_db.list_profiles()}
        return self.prune_orphans(valid_ids)
