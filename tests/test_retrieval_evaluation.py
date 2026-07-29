import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from evaluation import run_eval


class FakeVectorStore:
    retrieval_mode = "vector"
    top_k = 3


class FakeRag:
    def __init__(self):
        self.vector_store = FakeVectorStore()

    def set_retrieval_mode(self, mode):
        self.vector_store.retrieval_mode = mode

    def retrieve_docs(self, _query):
        return [Document(page_content="滤网需要定期清理", metadata={"source": "/tmp/维护保养.txt"})]


class EvidenceRag(FakeRag):
    def retrieve_docs(self, _query):
        return [
            Document(page_content="这是一段不相关说明", metadata={"source": "/tmp/故障排除.txt"}),
            Document(page_content="过滤部件需要定期清理", metadata={"source": "/tmp/维护保养.txt"}),
        ]


class RetrievalEvaluationTests(unittest.TestCase):
    def test_retrieval_only_mode_does_not_need_agent_model_calls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "samples.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "rag_001",
                        "type": "rag",
                        "query": "滤网多久清理一次？",
                        "expected_tools": ["rag_summarize"],
                        "gold_sources": ["维护保养.txt"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "output"
            with patch.object(run_eval, "OUTPUT_DIR", output_dir), patch.object(run_eval, "rag", FakeRag()):
                report = run_eval.run_retrieval_evaluation(
                    retrieval_mode="hybrid",
                    output_tag="hybrid",
                    dataset_path=dataset_path,
                )

            self.assertEqual(report["evaluation_scope"], "retrieval_only")
            self.assertEqual(report["dataset_path"], str(dataset_path))
            self.assertEqual(report["metrics"]["top_k_hit_rate"], 100.0)
            self.assertEqual(report["metrics"]["avg_recall_at_k"], 100.0)
            self.assertTrue((output_dir / "hybrid_report.json").exists())

    def test_evidence_level_mode_reports_rank_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "evidence_samples.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "evidence_001",
                        "type": "rag",
                        "query": "过滤部件多久清理一次？",
                        "expected_tools": ["rag_summarize"],
                        "gold_evidence": [
                            {"source": "维护保养.txt", "anchor": "过滤部件需要定期清理"}
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "output"
            with patch.object(run_eval, "OUTPUT_DIR", output_dir), patch.object(run_eval, "rag", EvidenceRag()):
                report = run_eval.run_retrieval_evaluation(
                    output_tag="evidence",
                    dataset_path=dataset_path,
                )

            metrics = report["metrics"]
            self.assertEqual(report["evaluation_protocol"], "evidence_level_v2")
            self.assertEqual(metrics["evidence_hit_at_1"], 0.0)
            self.assertEqual(metrics["evidence_hit_at_k"], 100.0)
            self.assertEqual(metrics["avg_evidence_mrr_at_k"], 0.5)
            self.assertEqual(metrics["avg_evidence_recall_at_k"], 100.0)
