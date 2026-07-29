import csv
import tempfile
import unittest
from pathlib import Path

from utils.sqlite_store import SQLiteStore


class SQLiteStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        csv_path = root / "records.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["用户ID", "时间", "特征", "清洁效率", "耗材", "对比"])
            writer.writeheader()
            writer.writerow({
                "用户ID": "1001",
                "时间": "2025-01",
                "特征": "宠物毛发",
                "清洁效率": "90%",
                "耗材": "滤网正常",
                "对比": "较上月提升",
            })

        self.store = SQLiteStore()
        self.store.db_path = root / "app.db"
        self.store.external_data_path = csv_path
        self.store.history_limit = 2
        self.store._initialized = False

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_usage_record_and_conversation_messages_are_persisted(self):
        record = self.store.get_usage_record("1001", "2025-01")
        self.assertEqual(record["特征"], "宠物毛发")

        conversation_id = self.store.ensure_conversation("conversation-test")
        self.store.append_message(conversation_id, "user", "第一条问题")
        self.store.append_message(conversation_id, "assistant", "第一条回答")
        self.store.append_message(conversation_id, "user", "第二条问题")

        # history_limit=2 means only the latest two messages enter the next Agent request.
        history = self.store.list_messages(conversation_id)
        self.assertEqual(
            history,
            [
                {"role": "assistant", "content": "第一条回答"},
                {"role": "user", "content": "第二条问题"},
            ],
        )

