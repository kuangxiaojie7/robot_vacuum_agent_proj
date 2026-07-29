"""Build a frozen, evidence-level retrieval benchmark.

The original benchmark measures whether a relevant *file* appears in Top-K.
This v2 benchmark keeps the same local knowledge scope but labels the exact
source section that should be retrieved. It is intentionally stored in new
files so the original source-level benchmark remains reproducible.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from evaluation.build_retrieval_benchmark import (
    OUTPUT_DIR,
    RetrievalCandidate,
    choose_evenly,
    collect_candidates,
    normalize_query,
)


V2_WORDING_REPLACEMENTS = (
    ("扫地机器人", "清扫设备"),
    ("机器人", "机器"),
    ("APP", "手机端"),
    ("WiFi", "家里网络"),
    ("开机", "刚启动"),
    ("回充", "自己回基座充电"),
    ("充电座", "充电底座"),
    ("建图", "生成房间地图"),
    ("地图错乱", "房间图混乱"),
    ("漏扫", "有地方没扫到"),
    ("边刷", "侧边的小刷子"),
    ("主刷", "底部滚刷"),
    ("滤网", "过滤部件"),
    ("尘盒", "装灰的盒子"),
    ("水箱", "储水盒"),
    ("拖布", "擦地布"),
    ("避障", "躲开家具杂物"),
    ("传感器", "感应部件"),
    ("无法", "没法"),
    ("如何", "怎么"),
    ("为什么", "是什么原因"),
)


def normalize_evidence_text(text: str) -> str:
    """Remove formatting and punctuation before matching evidence anchors."""
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", text or "").lower()


def evidence_anchor_for(candidate: RetrievalCandidate) -> str:
    """Return a heading/phenomenon that identifies one exact source section."""
    if candidate.source_group == "fault":
        return candidate.query.removesuffix("，应该如何排查和处理？")
    if candidate.source_group == "purchase":
        return candidate.query.removeprefix("选购扫地机器人时，").removesuffix("需要关注什么？")
    return candidate.query.rstrip("？?")


def rewrite_topic(text: str) -> str:
    rewritten = text
    for original, replacement in V2_WORDING_REPLACEMENTS:
        rewritten = rewritten.replace(original, replacement)
    return rewritten


def v2_user_query(candidate: RetrievalCandidate, index: int) -> str:
    """Use scenario prompts rather than document-title-shaped questions."""
    source_question = candidate.query.rstrip("？?")

    if candidate.source_group == "fault":
        symptom = rewrite_topic(evidence_anchor_for(candidate).removeprefix("机器人"))
        templates = (
            "家里的清扫设备出现{symptom}，我想先自己排查，应该从哪里开始？",
            "机器最近{symptom}，不急着报修的话先检查哪些地方？",
            "遇到{symptom}这种情况，通常是什么部位导致的，怎么处理更稳妥？",
            "设备出现{symptom}，有没有按步骤自查的方法？",
        )
        return templates[index % len(templates)].format(symptom=symptom)

    if candidate.source_group == "purchase":
        topic = rewrite_topic(evidence_anchor_for(candidate))
        templates = (
            "准备买清扫设备，家里最在意{topic}，看参数时该怎么选？",
            "挑选机器时如果重点是{topic}，哪些配置更值得优先比较？",
            "我不太懂参数，和{topic}有关的指标应该重点看什么？",
            "以{topic}为主要需求，选购时有什么容易忽略的点？",
        )
        return templates[index % len(templates)].format(topic=topic)

    topic = rewrite_topic(source_question)
    templates = (
        "家里这台设备碰到这个状况：{topic}。我先该怎么处理？",
        "日常使用中遇到“{topic}”，一般优先检查哪些地方？",
        "机器出现{topic}的情况，能给一个实际可执行的处理建议吗？",
        "不用太专业地说，{topic}通常是哪里出了问题，怎么排查？",
    )
    return templates[index % len(templates)].format(topic=topic)


def build_rows(candidates: list[RetrievalCandidate], prefix: str) -> list[dict]:
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        anchor = evidence_anchor_for(candidate)
        row = {
            "id": f"{prefix}_{index:03d}",
            "type": "rag",
            "query": v2_user_query(candidate, index),
            "expected_tools": ["rag_summarize"],
            "gold_sources": [candidate.source],
            "gold_evidence": [{"source": candidate.source, "anchor": anchor}],
            "source_group": candidate.source_group,
            "scenario": candidate.scenario,
            "source_question": candidate.query,
            "query_style": "scenario_paraphrase_evidence_v2",
            "benchmark_version": "evidence_v2",
        }
        if normalize_query(row["query"]) == normalize_query(row["source_question"]):
            raise AssertionError(f"Query leaks source heading: {row['id']}")
        rows.append(row)
    return rows


def build_benchmark_v2() -> tuple[list[dict], list[dict]]:
    """Create a fixed 100/500 split with no query overlap inside v2."""
    groups = collect_candidates()
    quotas = {
        "vacuum_faq": {"test": 85, "dev": 15},
        "mop_faq": {"test": 85, "dev": 15},
        "fault": {"test": 140, "dev": 30},
        "purchase": {"test": 140, "dev": 30},
        "technical_faq": {"test": 50, "dev": 10},
    }

    dev_candidates: list[RetrievalCandidate] = []
    test_candidates: list[RetrievalCandidate] = []
    for group_name, quota in quotas.items():
        candidates = groups[group_name]
        # Offset the deterministic picks relative to v1, then split by the
        # evidence source question so v2 dev/test remain disjoint.
        rotated = candidates[len(candidates) // 3 :] + candidates[: len(candidates) // 3]
        dev_group = choose_evenly(rotated, quota["dev"])
        dev_questions = {normalize_query(candidate.query) for candidate in dev_group}
        remaining = [
            candidate for candidate in rotated if normalize_query(candidate.query) not in dev_questions
        ]
        test_group = choose_evenly(remaining, quota["test"])
        dev_candidates.extend(dev_group)
        test_candidates.extend(test_group)

    dev_rows = build_rows(dev_candidates, "retrieval_v2_dev")
    test_rows = build_rows(test_candidates, "retrieval_v2_test")
    dev_queries = {normalize_query(row["query"]) for row in dev_rows}
    test_queries = {normalize_query(row["query"]) for row in test_rows}
    if len(dev_rows) != 100 or len(test_rows) != 500:
        raise AssertionError("Unexpected v2 benchmark size")
    if len(dev_queries) != len(dev_rows) or len(test_queries) != len(test_rows):
        raise AssertionError("Duplicate v2 queries detected")
    if dev_queries & test_queries:
        raise AssertionError("v2 dev/test query overlap detected")
    return dev_rows, test_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    dev_rows, test_rows = build_benchmark_v2()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_DIR / "retrieval_v2_dev.jsonl", dev_rows)
    write_jsonl(OUTPUT_DIR / "retrieval_v2_test.jsonl", test_rows)
    print("Generated frozen evidence-level benchmark: v2_dev=100, v2_test=500")


if __name__ == "__main__":
    main()
