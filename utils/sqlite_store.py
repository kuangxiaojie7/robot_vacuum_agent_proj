from __future__ import annotations

import csv
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tools import get_abs_path


class SQLiteStore:
    def __init__(self):
        self.db_path = Path(get_abs_path(agent_conf.get("sqlite_db_path", "data/app.db")))
        self.external_data_path = Path(get_abs_path(agent_conf["external_data_path"]))
        self.history_limit = int(agent_conf.get("conversation_history_limit", 20))
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self):
        """Close every SQLite connection after its transaction completes."""
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _initialize(self) -> None:
        if self._initialized:
            return

        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_records (
                    user_id TEXT NOT NULL,
                    month TEXT NOT NULL,
                    feature TEXT NOT NULL,
                    efficiency TEXT NOT NULL,
                    consumables TEXT NOT NULL,
                    comparison TEXT NOT NULL,
                    PRIMARY KEY (user_id, month)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                ON messages (conversation_id, id)
                """
            )
            conn.commit()

        self._sync_usage_records()
        self._initialized = True

    def _sync_usage_records(self) -> None:
        if not self.external_data_path.exists():
            logger.warning(f"[sqlite_store] 外部记录文件不存在: {self.external_data_path}")
            return

        with self._connection() as conn:
            count = conn.execute("SELECT COUNT(1) FROM usage_records").fetchone()[0]
            if count > 0:
                return

            with open(self.external_data_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = [
                    (
                        str(row.get("用户ID", "")).strip(),
                        str(row.get("时间", "")).strip(),
                        str(row.get("特征", "")).strip(),
                        str(row.get("清洁效率", "")).strip(),
                        str(row.get("耗材", "")).strip(),
                        str(row.get("对比", "")).strip(),
                    )
                    for row in reader
                    if str(row.get("用户ID", "")).strip() and str(row.get("时间", "")).strip()
                ]

            conn.executemany(
                """
                INSERT OR REPLACE INTO usage_records (
                    user_id, month, feature, efficiency, consumables, comparison
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def get_usage_record(self, user_id: str, month: str) -> dict[str, str] | None:
        self._initialize()
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT feature, efficiency, consumables, comparison
                FROM usage_records
                WHERE user_id = ? AND month = ?
                """,
                (user_id, month),
            ).fetchone()

        if not row:
            return None

        return {
            "特征": row["feature"],
            "效率": row["efficiency"],
            "耗材": row["consumables"],
            "对比": row["comparison"],
        }

    def ensure_conversation(self, conversation_id: str | None = None) -> str:
        self._initialize()
        conversation_id = (conversation_id or "").strip() or str(uuid.uuid4())
        now = datetime.now().isoformat(timespec="seconds")
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO conversations (conversation_id, created_at, updated_at)
                VALUES (?, ?, ?)
                """,
                (conversation_id, now, now),
            )
            conn.commit()
        return conversation_id

    def list_messages(self, conversation_id: str, limit: int | None = None) -> list[dict[str, str]]:
        self._initialize()
        limit = limit or self.history_limit
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM (
                    SELECT role, content, id
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (conversation_id, limit),
            ).fetchall()

        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def seed_messages(self, conversation_id: str, messages: list[dict[str, str]]) -> None:
        self._initialize()
        if not messages:
            return

        with self._connection() as conn:
            existing = conn.execute(
                "SELECT COUNT(1) FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0]
            if existing > 0:
                return

            now = datetime.now().isoformat(timespec="seconds")
            conn.executemany(
                """
                INSERT INTO messages (conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (conversation_id, msg["role"], str(msg["content"]), now)
                    for msg in messages
                    if msg.get("role") and msg.get("content") is not None
                ],
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )
            conn.commit()

    def append_message(self, conversation_id: str, role: str, content: str) -> None:
        self._initialize()
        now = datetime.now().isoformat(timespec="seconds")
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, role, content, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )
            conn.commit()


sqlite_store = SQLiteStore()
