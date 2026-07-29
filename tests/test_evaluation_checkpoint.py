import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import run_eval


class FakeVectorStore:
    def __init__(self):
        self.retrieval_mode = "hybrid"
        self.top_k = 3


class FakeRag:
    def __init__(self):
        self.vector_store = FakeVectorStore()

    def set_retrieval_mode(self, mode):
        self.vector_store.retrieval_mode = mode


class FakeAgent:
    def __init__(self, calls, interrupt_on=None):
        self.calls = calls
        self.interrupt_on = interrupt_on

    def execute(self, query, history=None):
        self.calls.append(query)
        if query == self.interrupt_on:
            raise KeyboardInterrupt()
        return {
            "answer": f"回答：{query}",
            "latency_ms": 1.0,
            "tool_call_total": 0,
            "tool_call_success": 0,
            "tool_call_failed": 0,
            "tool_calls": [],
            "tool_call_failed_names": [],
        }


class EvaluationCheckpointTests(unittest.TestCase):
    def test_interrupted_run_resumes_without_repeating_completed_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "samples.jsonl"
            dataset_path.write_text(
                "\n".join(
                    json.dumps(sample, ensure_ascii=False)
                    for sample in (
                        {
                            "id": "sample_001",
                            "type": "general",
                            "query": "第一题",
                            "expected_keywords": ["第一题"],
                        },
                        {
                            "id": "sample_002",
                            "type": "general",
                            "query": "第二题",
                            "expected_keywords": ["第二题"],
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "output"
            first_calls = []
            second_calls = []

            with (
                patch.object(run_eval, "OUTPUT_DIR", output_dir),
                patch.object(run_eval, "rag", FakeRag()),
                patch.object(run_eval, "ReactAgent", return_value=FakeAgent(first_calls, interrupt_on="第二题")),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_eval.run_evaluation(
                        retrieval_mode="hybrid",
                        output_tag="resume_demo",
                        enable_judge=False,
                        dataset_path=dataset_path,
                    )

            checkpoint_path = output_dir / "resume_demo_checkpoint.jsonl"
            checkpoint_rows = [
                json.loads(line)
                for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(first_calls, ["第一题", "第二题"])
            self.assertEqual([row["id"] for row in checkpoint_rows], ["sample_001"])

            with (
                patch.object(run_eval, "OUTPUT_DIR", output_dir),
                patch.object(run_eval, "rag", FakeRag()),
                patch.object(run_eval, "ReactAgent", return_value=FakeAgent(second_calls)),
            ):
                report = run_eval.run_evaluation(
                    retrieval_mode="hybrid",
                    output_tag="resume_demo",
                    enable_judge=False,
                    dataset_path=dataset_path,
                    resume=True,
                )

            self.assertEqual(second_calls, ["第二题"])
            self.assertEqual(report["total_samples"], 2)
            self.assertEqual(report["metrics"]["answer_accuracy"], 100.0)
            self.assertTrue((output_dir / "resume_demo_report.json").exists())
            self.assertTrue((output_dir / "resume_demo_details.csv").exists())

