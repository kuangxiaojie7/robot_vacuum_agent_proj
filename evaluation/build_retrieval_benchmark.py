"""Build source-grounded retrieval dev/test datasets from local knowledge documents.

Queries are rewritten as user-facing scenarios instead of copying source
headings verbatim. Each row preserves ``source_question`` for auditability and
uses one or more verified source files as its gold evidence labels.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "datasets"

# These component topics are documented by more than one local file. They are
# intentionally conservative: unlisted topics keep only their primary source.
MULTI_SOURCE_RULES = (
    (
        ("滤网", "边刷", "主刷", "尘盒", "吸力", "集尘"),
        ["扫地机器人100问2.txt", "故障排除.txt", "维护保养.txt"],
    ),
    (
        ("水箱", "拖布", "清洁仓", "污水", "出水", "拖地"),
        ["扫拖一体机器人100问.txt", "故障排除.txt", "维护保养.txt"],
    ),
    (
        ("充电", "回充", "WiFi", "地图", "漏扫", "避障", "传感器", "APP"),
        ["扫地机器人100问2.txt", "故障排除.txt"],
    ),
    (
        ("地毯", "宠物", "续航", "越障", "噪音", "导航"),
        ["扫地机器人100问2.txt", "选购指南.txt"],
    ),
)

# Replace common document-heading wording with the phrasing users are more
# likely to use in a support conversation. The source heading is retained in
# each JSONL row for review, while the evaluated query uses this paraphrase.
USER_WORDING_REPLACEMENTS = (
    ("首次使用", "第一次启用"),
    ("扫地机器人", "机器人"),
    ("APP", "手机应用"),
    ("WiFi", "无线网络"),
    ("找不到充电座", "回不了充电基站"),
    ("开机后", "启动后"),
    ("不移动", "不走动"),
    ("建图", "生成房间地图"),
    ("地图错乱", "房间地图混乱"),
    ("漏扫", "有区域没扫到"),
    ("边刷", "侧边刷"),
    ("主刷", "滚刷"),
    ("滤网", "过滤网"),
    ("尘盒", "集尘盒"),
    ("集尘袋", "集尘耗材袋"),
    ("水箱", "储水箱"),
    ("拖布", "擦地布"),
    ("水渍、水痕", "水印和残水"),
    ("地毯", "绒毯区域"),
    ("避障", "躲避家具"),
    ("传感器", "感应组件"),
    ("怎么处理", "该先排查哪里"),
    ("怎么办", "该先排查哪里"),
    ("如何", "怎样"),
    ("为什么", "什么原因会导致"),
    ("多久", "间隔多长时间"),
    ("无法", "没法"),
    ("需要做什么", "要准备什么"),
)


@dataclass(frozen=True)
class RetrievalCandidate:
    query: str
    source: str
    source_group: str
    scenario: str


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", "", query).strip("？?。！!")


def extract_faq_candidates(path: Path, source_group: str) -> list[RetrievalCandidate]:
    candidates = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\d+\.\s+\*\*(.+?)\*\*", line.strip())
        if match:
            candidates.append(
                RetrievalCandidate(
                    query=match.group(1).replace("**", ""),
                    source=path.name,
                    source_group=source_group,
                    scenario="faq",
                )
            )
    return candidates


def extract_fault_candidates(path: Path) -> list[RetrievalCandidate]:
    candidates = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\d+\.\s+故障现象：([^；]+)", line.strip())
        if match:
            candidates.append(
                RetrievalCandidate(
                    query=f"{match.group(1)}，应该如何排查和处理？",
                    source=path.name,
                    source_group="fault",
                    scenario="fault_diagnosis",
                )
            )
    return candidates


def extract_purchase_candidates(path: Path) -> list[RetrievalCandidate]:
    candidates = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\d+\.\s+([^：]+)：", line.strip())
        if match:
            candidates.append(
                RetrievalCandidate(
                    query=f"选购扫地机器人时，{match.group(1)}需要关注什么？",
                    source=path.name,
                    source_group="purchase",
                    scenario="purchase_advice",
                )
            )
    return candidates


def extract_pdf_faq_candidates(path: Path) -> list[RetrievalCandidate]:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    candidates = []
    for line in text.splitlines():
        match = re.match(r"^\d+\.\s+\*\*(.+?)\*\*", line.strip())
        if match:
            candidates.append(
                RetrievalCandidate(
                    query=match.group(1).replace("**", ""),
                    source=path.name,
                    source_group="technical_faq",
                    scenario="faq",
                )
            )
    return candidates


def collect_candidates() -> dict[str, list[RetrievalCandidate]]:
    groups = {
        "vacuum_faq": extract_faq_candidates(DATA_DIR / "扫地机器人100问2.txt", "vacuum_faq"),
        "mop_faq": extract_faq_candidates(DATA_DIR / "扫拖一体机器人100问.txt", "mop_faq"),
        "fault": extract_fault_candidates(DATA_DIR / "故障排除.txt"),
        "purchase": extract_purchase_candidates(DATA_DIR / "选购指南.txt"),
        "technical_faq": extract_pdf_faq_candidates(DATA_DIR / "扫地机器人100问.pdf"),
    }

    seen_queries = set()
    deduplicated = defaultdict(list)
    for group_name, candidates in groups.items():
        for candidate in candidates:
            normalized = normalize_query(candidate.query)
            if not normalized or normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            deduplicated[group_name].append(candidate)
    return dict(deduplicated)


def choose_evenly(candidates: list[RetrievalCandidate], count: int) -> list[RetrievalCandidate]:
    if count > len(candidates):
        raise ValueError(f"Need {count} candidates but only have {len(candidates)}")
    if count == 0:
        return []
    if count == len(candidates):
        return list(candidates)

    indexes = {
        round(index * (len(candidates) - 1) / (count - 1))
        for index in range(count)
    }
    selected = [candidate for index, candidate in enumerate(candidates) if index in indexes]
    if len(selected) != count:
        raise AssertionError("Even selection produced an unexpected candidate count")
    return selected


def user_facing_query(candidate: RetrievalCandidate, index: int) -> str:
    source_question = candidate.query.rstrip("？?")
    natural_question = source_question
    for original, replacement in USER_WORDING_REPLACEMENTS:
        natural_question = natural_question.replace(original, replacement)

    if candidate.source_group == "fault":
        phenomenon = source_question.removesuffix("，应该如何排查和处理？")
        phenomenon = phenomenon.removeprefix("机器人")
        templates = (
            "我家的机器人最近{phenomenon}，联系售后前可以先检查什么？",
            "使用中出现{phenomenon}的情况，通常该从哪些部位排查？",
        )
        return templates[index % len(templates)].format(phenomenon=phenomenon)

    if candidate.source_group == "purchase":
        topic = source_question.removeprefix("选购扫地机器人时，").removesuffix("需要关注什么")
        templates = (
            "准备给家里买机器人，我特别在意{topic}，挑选参数时应该看什么？",
            "选购设备时如果主要关注{topic}，有哪些配置更值得优先比较？",
        )
        return templates[index % len(templates)].format(topic=topic)

    if candidate.source_group == "technical_faq":
        templates = (
            "作为普通用户，我想弄清楚：{question}，实际会影响哪些使用体验？",
            "能从实际使用角度解释一下吗：{question}？",
        )
        return templates[index % len(templates)].format(question=natural_question)

    if candidate.source_group == "mop_faq":
        templates = (
            "我在用扫拖一体机时想咨询：{question}，该怎么设置或处理？",
            "家里扫拖时遇到这个情况：{question}，有什么实用建议？",
        )
        return templates[index % len(templates)].format(question=natural_question)

    templates = (
        "日常使用机器人时，我想了解：{question}，应该注意什么？",
        "家里的机器人遇到这个问题：{question}，通常该怎样处理？",
    )
    return templates[index % len(templates)].format(question=natural_question)


def gold_sources_for(candidate: RetrievalCandidate) -> list[str]:
    sources = [candidate.source]
    for keywords, related_sources in MULTI_SOURCE_RULES:
        if any(keyword in candidate.query for keyword in keywords):
            sources.extend(related_sources)

    seen = set()
    ordered_sources = []
    for source in sources:
        if source in seen:
            continue
        seen.add(source)
        ordered_sources.append(source)
    return ordered_sources


def build_rows(candidates: list[RetrievalCandidate], prefix: str) -> list[dict]:
    return [
        {
            "id": f"{prefix}_{index:03d}",
            "type": "rag",
            "query": user_facing_query(candidate, index),
            "expected_tools": ["rag_summarize"],
            "gold_sources": gold_sources_for(candidate),
            "source_group": candidate.source_group,
            "scenario": candidate.scenario,
            "source_question": candidate.query,
            "query_style": "rule_based_user_paraphrase",
        }
        for index, candidate in enumerate(candidates, start=1)
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def build_benchmark() -> tuple[list[dict], list[dict]]:
    groups = collect_candidates()
    quotas = {
        "vacuum_faq": {"test": 85, "dev": 15},
        "mop_faq": {"test": 85, "dev": 15},
        "fault": {"test": 140, "dev": 30},
        "purchase": {"test": 140, "dev": 30},
        "technical_faq": {"test": 50, "dev": 10},
    }

    test_candidates = []
    dev_candidates = []
    for group_name, quota in quotas.items():
        candidates = groups[group_name]
        dev_group = choose_evenly(candidates, quota["dev"])
        dev_normalized = {normalize_query(candidate.query) for candidate in dev_group}
        remaining = [
            candidate
            for candidate in candidates
            if normalize_query(candidate.query) not in dev_normalized
        ]
        test_group = choose_evenly(remaining, quota["test"])
        test_candidates.extend(test_group)
        dev_candidates.extend(dev_group)

    test_rows = build_rows(test_candidates, "retrieval_test")
    dev_rows = build_rows(dev_candidates, "retrieval_dev")
    test_queries = {normalize_query(row["query"]) for row in test_rows}
    dev_queries = {normalize_query(row["query"]) for row in dev_rows}
    if len(test_rows) != 500 or len(dev_rows) != 100:
        raise AssertionError("Unexpected benchmark size")
    if len(test_queries) != len(test_rows) or len(dev_queries) != len(dev_rows):
        raise AssertionError("Duplicate queries detected in a retrieval split")
    if test_queries & dev_queries:
        raise AssertionError("Retrieval dev and test datasets overlap")
    return dev_rows, test_rows


def main() -> None:
    dev_rows, test_rows = build_benchmark()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_DIR / "retrieval_dev.jsonl", dev_rows)
    write_jsonl(OUTPUT_DIR / "retrieval_test.jsonl", test_rows)
    print("Generated retrieval benchmark: dev=100, test=500")


if __name__ == "__main__":
    main()
