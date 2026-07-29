import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import main as api_main
from utils.request_context import get_request_user_city, get_request_user_id
from utils.sqlite_store import SQLiteStore


class FakeAgent:
    def __init__(self):
        self.calls = []

    def execute(self, query: str, history=None, context=None):
        self.calls.append({"query": query, "history": history or []})
        return {
            "answer": f"已回答：{query}",
            "sources": ["[1] 故障排除.txt：测试证据"],
            "latency_ms": 1.0,
            "tool_call_total": 0,
            "tool_call_success": 0,
            "tool_call_failed": 0,
            "tool_calls": [],
            "tool_call_failed_names": [],
        }


class ApiChatTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        csv_path = root / "records.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["用户ID", "时间", "特征", "清洁效率", "耗材", "对比"])
            writer.writeheader()

        self.store = SQLiteStore()
        self.store.db_path = root / "app.db"
        self.store.external_data_path = csv_path
        self.store._initialized = False
        self.fake_agent = FakeAgent()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_chat_reuses_persisted_history_by_conversation_id(self):
        with patch.object(api_main, "sqlite_store", self.store), patch.object(api_main, "agent", self.fake_agent):
            client = TestClient(api_main.app)
            first = client.post(
                "/chat",
                json={"query": "第一轮问题", "user_id": "1001", "city": "合肥"},
            )
            self.assertEqual(first.status_code, 200)
            conversation_id = first.json()["conversation_id"]
            self.assertEqual(first.json()["sources"], ["[1] 故障排除.txt：测试证据"])

            second = client.post(
                "/chat",
                json={"query": "第二轮追问", "conversation_id": conversation_id},
            )
            self.assertEqual(second.status_code, 200)
            self.assertEqual(second.json()["conversation_id"], conversation_id)
            self.assertEqual(
                self.fake_agent.calls[1]["history"],
                [
                    {"role": "user", "content": "第一轮问题"},
                    {"role": "assistant", "content": "已回答：第一轮问题"},
                ],
            )

        self.assertIsNone(get_request_user_id())
        self.assertIsNone(get_request_user_city())

