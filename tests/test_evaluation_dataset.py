import json
import unittest
from pathlib import Path


class EvaluationDatasetTests(unittest.TestCase):
    @staticmethod
    def load_jsonl(filename: str) -> list[dict]:
        dataset_path = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / filename
        return [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def is_retrieval_sample(sample: dict) -> bool:
        return sample.get("type") == "rag" or "rag_summarize" in sample.get("expected_tools", [])

    def test_end_to_end_set_keeps_one_hundred_fifty_samples_and_contains_context_dependent_turns(self):
        samples = self.load_jsonl("qa_samples.jsonl")
        multi_turn_samples = [sample for sample in samples if sample.get("type") == "multi_turn"]

        self.assertEqual(len(samples), 150)
        self.assertEqual(len(multi_turn_samples), 24)
        self.assertTrue(all(sample.get("history") for sample in multi_turn_samples))
        self.assertTrue(all(sample.get("expected_tools") for sample in multi_turn_samples))

    def test_end_to_end_retrieval_samples_are_unique_and_have_gold_sources(self):
        samples = self.load_jsonl("qa_samples.jsonl")
        retrieval_samples = [sample for sample in samples if self.is_retrieval_sample(sample)]
        queries = [sample["query"] for sample in retrieval_samples]

        self.assertEqual(len(retrieval_samples), 78)
        self.assertEqual(len(queries), len(set(queries)))
        self.assertTrue(all(sample.get("gold_sources") for sample in retrieval_samples))

    def test_retrieval_dev_and_test_sets_are_source_labeled_and_isolated(self):
        dev_samples = self.load_jsonl("retrieval_dev.jsonl")
        test_samples = self.load_jsonl("retrieval_test.jsonl")
        dev_queries = [sample["query"] for sample in dev_samples]
        test_queries = [sample["query"] for sample in test_samples]

        self.assertEqual(len(dev_samples), 100)
        self.assertEqual(len(test_samples), 500)
        self.assertTrue(all(sample.get("type") == "rag" for sample in dev_samples))
        self.assertTrue(all(sample.get("type") == "rag" for sample in test_samples))
        self.assertEqual(len(dev_queries), len(set(dev_queries)))
        self.assertEqual(len(test_queries), len(set(test_queries)))
        self.assertTrue(all(sample.get("gold_sources") for sample in dev_samples))
        self.assertTrue(all(sample.get("gold_sources") for sample in test_samples))
        self.assertTrue(all(sample.get("source_question") for sample in dev_samples + test_samples))
        self.assertTrue(all(sample["query"] != sample["source_question"] for sample in dev_samples + test_samples))
        self.assertGreater(sum(len(sample["gold_sources"]) > 1 for sample in test_samples), 0)
        self.assertFalse(set(test_queries) & set(dev_queries))
        self.assertEqual(
            {sample.get("source_group") for sample in test_samples},
            {"vacuum_faq", "mop_faq", "fault", "purchase", "technical_faq"},
        )

    def test_evidence_level_v2_sets_are_separate_and_anchor_labeled(self):
        dev_samples = self.load_jsonl("retrieval_v2_dev.jsonl")
        test_samples = self.load_jsonl("retrieval_v2_test.jsonl")
        dev_queries = {sample["query"] for sample in dev_samples}
        test_queries = {sample["query"] for sample in test_samples}

        self.assertEqual(len(dev_samples), 100)
        self.assertEqual(len(test_samples), 500)
        self.assertFalse(dev_queries & test_queries)
        self.assertTrue(all(sample.get("benchmark_version") == "evidence_v2" for sample in dev_samples + test_samples))
        self.assertTrue(
            all(
                len(sample.get("gold_evidence", [])) == 1
                and sample["gold_evidence"][0].get("source")
                and sample["gold_evidence"][0].get("anchor")
                for sample in dev_samples + test_samples
            )
        )

    def test_end_to_end_retrieval_queries_do_not_leak_into_retrieval_benchmark(self):
        end_to_end_queries = {
            sample["query"]
            for sample in self.load_jsonl("qa_samples.jsonl")
            if self.is_retrieval_sample(sample)
        }
        benchmark_queries = {
            sample["query"]
            for filename in ("retrieval_dev.jsonl", "retrieval_test.jsonl")
            for sample in self.load_jsonl(filename)
        }

        self.assertFalse(end_to_end_queries & benchmark_queries)
