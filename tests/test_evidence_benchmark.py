import unittest

from evaluation.build_retrieval_benchmark import normalize_query
from evaluation.build_retrieval_benchmark_v2 import build_benchmark_v2, normalize_evidence_text


class EvidenceBenchmarkTests(unittest.TestCase):
    def test_v2_benchmark_is_frozen_by_query_and_has_exact_evidence_labels(self):
        dev_rows, test_rows = build_benchmark_v2()
        dev_queries = {normalize_query(row["query"]) for row in dev_rows}
        test_queries = {normalize_query(row["query"]) for row in test_rows}

        self.assertEqual(len(dev_rows), 100)
        self.assertEqual(len(test_rows), 500)
        self.assertFalse(dev_queries & test_queries)
        self.assertTrue(all(row["benchmark_version"] == "evidence_v2" for row in dev_rows + test_rows))
        self.assertTrue(all(row["query"] != row["source_question"] for row in dev_rows + test_rows))
        self.assertTrue(
            all(
                len(row["gold_evidence"]) == 1
                and row["gold_evidence"][0]["source"]
                and normalize_evidence_text(row["gold_evidence"][0]["anchor"])
                for row in dev_rows + test_rows
            )
        )
