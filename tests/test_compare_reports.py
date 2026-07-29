import json
import tempfile
import unittest
from pathlib import Path

from evaluation.compare_reports import calc_latency_change, compare_reports, format_latency_line


class CompareReportsTests(unittest.TestCase):
    def test_latency_increase_is_not_reported_as_a_reduction(self):
        change = calc_latency_change(100.0, 120.0)
        line = format_latency_line(change, label="检索时延")

        self.assertIn("增加 20.0 ms", line)
        self.assertNotIn("降低", line)

    def test_evidence_level_reports_are_compared_with_rank_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_path = root / "vector.json"
            optimized_path = root / "hybrid.json"
            common = {
                "evaluation_scope": "retrieval_only",
                "evaluation_protocol": "evidence_level_v2",
                "total_samples": 100,
                "dataset_path": "/tmp/v2.jsonl",
                "detail_path": "",
            }
            baseline_path.write_text(
                json.dumps(
                    {
                        **common,
                        "metrics": {
                            "evidence_hit_at_1": 50.0,
                            "evidence_hit_at_k": 80.0,
                            "avg_evidence_mrr_at_k": 0.6,
                            "avg_evidence_recall_at_k": 80.0,
                            "avg_retrieval_latency_ms": 100.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            optimized_path.write_text(
                json.dumps(
                    {
                        **common,
                        "metrics": {
                            "evidence_hit_at_1": 60.0,
                            "evidence_hit_at_k": 90.0,
                            "avg_evidence_mrr_at_k": 0.7,
                            "avg_evidence_recall_at_k": 90.0,
                            "avg_retrieval_latency_ms": 110.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = compare_reports(baseline_path, optimized_path)

            self.assertEqual(result["evaluation_scope"], "evidence_level_v2")
            self.assertEqual(result["metrics"]["evidence_hit_at_1"]["absolute_change"], 10.0)
            self.assertEqual(result["metrics"]["avg_evidence_mrr_at_k"]["absolute_change"], 0.1)


if __name__ == "__main__":
    unittest.main()
