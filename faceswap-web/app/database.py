from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FaceRecord:
    id: str
    name: str
    original_image_path: str
    aligned_image_path: str
    embedding_path: str
    created_at: str
    updated_at: str


class Database:
    def __init__(self, path: Path, timeout_seconds: float = 30.0) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self._write_lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.timeout_seconds,
            isolation_level="DEFERRED",
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _to_record(row: sqlite3.Row | None) -> FaceRecord | None:
        if row is None:
            return None
        return FaceRecord(**dict(row))

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock, self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS faces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL CHECK(length(name) BETWEEN 1 AND 80),
                    original_image_path TEXT NOT NULL,
                    aligned_image_path TEXT NOT NULL,
                    embedding_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_faces_created_at ON faces(created_at DESC)"
            )

    def list_faces(self) -> list[FaceRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM faces ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [FaceRecord(**dict(row)) for row in rows]

    def get_face(self, face_id: str) -> FaceRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM faces WHERE id = ?", (face_id,)
            ).fetchone()
        return self._to_record(row)

    def create_face(self, record: FaceRecord) -> FaceRecord:
        with self._write_lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO faces (
                    id, name, original_image_path, aligned_image_path,
                    embedding_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.name,
                    record.original_image_path,
                    record.aligned_image_path,
                    record.embedding_path,
                    record.created_at,
                    record.updated_at,
                ),
            )
        return record

    def rename_face(self, face_id: str, name: str, updated_at: str) -> FaceRecord | None:
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE faces SET name = ?, updated_at = ? WHERE id = ?",
                (name, updated_at, face_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_face(face_id)

    def delete_face(self, face_id: str) -> FaceRecord | None:
        with self._write_lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM faces WHERE id = ?", (face_id,)
            ).fetchone()
            if row is None:
                return None
            connection.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        return self._to_record(row)
