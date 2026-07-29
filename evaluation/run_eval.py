import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.react_agent import ReactAgent
from agent.tools.agent_tools import clear_user_context, rag, set_user_context
from model.factory import judge_model

DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "qa_samples.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "output"
REPORT_PATH = OUTPUT_DIR / "latest_report.json"
DETAIL_PATH = OUTPUT_DIR / "latest_details.csv"

RAG_GOLD_SOURCE_MAP = {
    "吸力变弱": ["故障排除.txt", "选购指南.txt"],
    "漏扫": ["扫地机器人100问2.txt", "故障排除.txt"],
    "拖布多久清洗": ["扫拖一体机器人100问.txt", "维护保养.txt"],
    "边刷多久更换": ["维护保养.txt", "故障排除.txt"],
    "地毯场景": ["选购指南.txt", "维护保养.txt"],
}


class JudgeResult(BaseModel):
    score: int = Field(..., ge=1, le=5)
    passed: bool
    reason: str


JUDGE_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=JudgeResult)
JUDGE_PROMPT = PromptTemplate.from_template(
    """
你是一个评测助手，需要判断智能客服回答是否满足要求。

请根据以下维度评分：
1. 是否回答了用户问题
2. 内容是否基本正确，是否存在明显幻觉
3. 是否覆盖了关键要点
4. 表达是否清晰、可直接给用户使用

评分规则：
- 5分：完整、准确、清晰
- 4分：基本正确，只有轻微缺失
- 3分：部分正确，但有明显缺失
- 2分：大部分不满足要求
- 1分：答非所问或明显错误

通过规则：
- score >= 4 时，passed=true
- 否则 passed=false

样本类型：{sample_type}
会话历史：{history}
用户问题：{query}
模型回答：{answer}
期望关键词：{expected_keywords}
预期工具：{expected_tools}
标准来源：{gold_sources}
实际检索来源：{retrieved_sources}

请只输出 JSON，不要输出额外解释。
{format_instructions}
    """
)
JUDGE_CHAIN = (JUDGE_PROMPT | judge_model | StrOutputParser()) if judge_model is not None else None


def load_samples(dataset_path: Path) -> list[dict]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"评测集不存在: {dataset_path}")

    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(normalize_sample(json.loads(line)))
    return rows


def normalize_sample(sample: dict) -> dict:
    normalized = dict(sample)
    expected_tools = normalized.get("expected_tools", [])
    history = normalized.get("history") or []

    if "type" not in normalized:
        if history:
            normalized["type"] = "multi_turn"
        elif "rag_summarize" in expected_tools:
            normalized["type"] = "rag"
        elif expected_tools:
            normalized["type"] = "tool"
        else:
            normalized["type"] = "general"

    normalized["history"] = history
    normalized["expected_keywords"] = normalized.get("expected_keywords", [])
    normalized["expected_tools"] = expected_tools
    normalized["gold_sources"] = normalized.get("gold_sources") or infer_gold_sources(
        normalized.get("query", "")
    )

    if normalized["type"] == "rag" and "retrieval_expected_keywords" not in normalized:
        query = str(normalized.get("query", "")).replace("？", "").replace("?", "")
        query_hint = query.replace("扫地机器人", "").strip()
        normalized["retrieval_expected_keywords"] = [query_hint[:4] or query[:4]]

    return normalized


def evaluate_answer(answer: str, expected_keywords: list[str]) -> bool:
    # Citations are appended by the RAG layer and should not make keyword accuracy look better.
    answer = str(answer or "")
    for marker in ("【检索来源】", "参考来源："):
        answer = answer.split(marker, 1)[0]
    if not expected_keywords:
        return bool(answer.strip())
    return all(keyword in answer for keyword in expected_keywords)


def evaluate_expected_tools(called_tools: list[str], expected_tools: list[str]) -> bool:
    if not expected_tools:
        return True
    return all(tool_name in called_tools for tool_name in expected_tools)


def infer_gold_sources(query: str) -> list[str]:
    query = str(query or "")
    for key, sources in RAG_GOLD_SOURCE_MAP.items():
        if key in query:
            return list(sources)
    return []


def unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def extract_retrieved_sources(retrieved_docs: list) -> list[str]:
    sources = []
    for doc in retrieved_docs:
        metadata = getattr(doc, "metadata", {}) or {}
        raw_source = str(metadata.get("source", "")).strip()
        if raw_source:
            sources.append(Path(raw_source).name)
    return unique_in_order(sources)


def evaluate_retrieval_hit(retrieved_docs: list, expected_keywords: list[str]) -> bool:
    if not expected_keywords:
        return False

    combined_text = "\n".join(getattr(doc, "page_content", "") for doc in retrieved_docs)
    return all(keyword in combined_text for keyword in expected_keywords)


def evaluate_retrieval_by_sources(
    retrieved_docs: list,
    gold_sources: list[str],
) -> tuple[bool, float, str]:
    if not gold_sources:
        return False, 0.0, ""

    retrieved_sources = extract_retrieved_sources(retrieved_docs)
    gold_set = set(gold_sources)
    hit = any(source in gold_set for source in retrieved_sources)
    recall = round(
        len(set(retrieved_sources) & gold_set) / len(gold_set) * 100,
        2,
    )
    return hit, recall, "|".join(retrieved_sources)


def retrieve_for_eval(
    query: str,
    expected_keywords: list[str],
    gold_sources: list[str],
) -> tuple[bool, float, float, str, str]:
    start = perf_counter()
    try:
        docs = rag.retrieve_docs(query)
        latency_ms = round((perf_counter() - start) * 1000, 2)
        if gold_sources:
            hit, recall_at_k, retrieved_sources = evaluate_retrieval_by_sources(docs, gold_sources)
        else:
            hit = evaluate_retrieval_hit(docs, expected_keywords)
            recall_at_k = 100.0 if hit else 0.0
            retrieved_sources = "|".join(extract_retrieved_sources(docs))
        return hit, recall_at_k, latency_ms, "", retrieved_sources
    except Exception as e:
        latency_ms = round((perf_counter() - start) * 1000, 2)
        return False, 0.0, latency_ms, str(e)[:200], ""


def normalize_evidence_text(text: str) -> str:
    """Normalize formatting before matching a labeled source-section anchor."""
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", str(text or "")).lower()


def evaluate_evidence_by_rank(
    retrieved_docs: list,
    gold_evidence: list[dict],
) -> tuple[bool, bool, float, float, str]:
    """Calculate exact-evidence Hit@1, Hit@K, MRR@K, and Recall@K."""
    expected = {
        (str(item.get("source", "")).strip(), normalize_evidence_text(item.get("anchor", "")))
        for item in gold_evidence
        if str(item.get("source", "")).strip() and normalize_evidence_text(item.get("anchor", ""))
    }
    if not expected:
        return False, False, 0.0, 0.0, ""

    matched: set[tuple[str, str]] = set()
    first_rank: int | None = None
    for rank, document in enumerate(retrieved_docs, start=1):
        metadata = getattr(document, "metadata", {}) or {}
        source = Path(str(metadata.get("source", ""))).name
        content = normalize_evidence_text(getattr(document, "page_content", ""))
        rank_matches = {
            evidence
            for evidence in expected
            if evidence[0] == source and evidence[1] in content
        }
        if rank_matches and first_rank is None:
            first_rank = rank
        matched.update(rank_matches)

    hit_at_k = first_rank is not None
    hit_at_1 = first_rank == 1
    mrr_at_k = round(1.0 / first_rank, 4) if first_rank else 0.0
    recall_at_k = round(len(matched) / len(expected) * 100, 2)
    return hit_at_1, hit_at_k, mrr_at_k, recall_at_k, "|".join(extract_retrieved_sources(retrieved_docs))


def retrieve_evidence_for_eval(
    query: str,
    gold_evidence: list[dict],
) -> tuple[bool, bool, float, float, float, str, str]:
    start = perf_counter()
    try:
        docs = rag.retrieve_docs(query)
        latency_ms = round((perf_counter() - start) * 1000, 2)
        hit_at_1, hit_at_k, mrr_at_k, recall_at_k, retrieved_sources = evaluate_evidence_by_rank(
            docs,
            gold_evidence,
        )
        return hit_at_1, hit_at_k, mrr_at_k, recall_at_k, latency_ms, "", retrieved_sources
    except Exception as e:
        latency_ms = round((perf_counter() - start) * 1000, 2)
        return False, False, 0.0, 0.0, latency_ms, str(e)[:200], ""


def is_retrieval_sample(sample: dict) -> bool:
    return sample.get("type") == "rag" or "rag_summarize" in sample.get("expected_tools", [])


def format_history(history: list[dict]) -> str:
    if not history:
        return "无"
    return "\n".join(
        f"{message.get('role', 'unknown')}：{message.get('content', '')}"
        for message in history
    )


def should_run_judge(sample: dict) -> bool:
    if JUDGE_CHAIN is None:
        return False

    if sample.get("type") == "rag":
        return True
    if sample.get("history"):
        return True
    return "fill_context_for_report" in sample.get("expected_tools", [])


def judge_answer(sample: dict, answer: str, retrieved_sources: str) -> tuple[str, str, str]:
    if JUDGE_CHAIN is None or not answer.strip():
        return "", "", ""

    input_dict = {
        "sample_type": sample.get("type", "general"),
        "history": format_history(sample.get("history", [])),
        "query": sample.get("query", ""),
        "answer": answer,
        "expected_keywords": "、".join(sample.get("expected_keywords", [])) or "无",
        "expected_tools": "、".join(sample.get("expected_tools", [])) or "无",
        "gold_sources": "、".join(sample.get("gold_sources", [])) or "无",
        "retrieved_sources": retrieved_sources or "无",
        "format_instructions": JUDGE_OUTPUT_PARSER.get_format_instructions(),
    }

    try:
        raw_output = JUDGE_CHAIN.invoke(input_dict)
        parsed = JUDGE_OUTPUT_PARSER.parse(raw_output)
    except Exception as e:
        return "", "", f"judge输出解析失败: {str(e)[:160]}"

    return str(parsed.score), str(int(parsed.passed)), parsed.reason[:200]


def output_paths(output_tag: str) -> tuple[Path, Path]:
    tag = re.sub(r"[^a-zA-Z0-9_-]", "_", output_tag.strip() or "latest")
    if tag == "latest":
        return REPORT_PATH, DETAIL_PATH
    return OUTPUT_DIR / f"{tag}_report.json", OUTPUT_DIR / f"{tag}_details.csv"


def checkpoint_paths(output_tag: str) -> tuple[Path, Path]:
    tag = re.sub(r"[^a-zA-Z0-9_-]", "_", output_tag.strip() or "latest")
    return (
        OUTPUT_DIR / f"{tag}_checkpoint.jsonl",
        OUTPUT_DIR / f"{tag}_checkpoint_meta.json",
    )


def _checkpoint_metadata(
    dataset_path: Path,
    retrieval_mode: str,
    enable_judge: bool,
    sample_count: int,
) -> dict:
    return {
        "schema_version": 1,
        "dataset_path": str(dataset_path.resolve()),
        "retrieval_mode": retrieval_mode,
        "enable_judge": enable_judge,
        "sample_count": sample_count,
    }


def _load_checkpoint_details(checkpoint_path: Path) -> list[dict]:
    details = []
    seen_ids = set()
    if not checkpoint_path.exists():
        return details

    lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            detail = json.loads(line)
        except json.JSONDecodeError as error:
            if line_number == len(lines):
                print(f"忽略 checkpoint 最后一条不完整记录：{checkpoint_path}")
                break
            raise ValueError(f"checkpoint 第 {line_number} 行不是合法 JSON") from error
        sample_id = str(detail.get("id", "")).strip()
        if not sample_id:
            raise ValueError(f"checkpoint 第 {line_number} 行缺少样本 id")
        if sample_id in seen_ids:
            raise ValueError(f"checkpoint 中存在重复样本 id：{sample_id}")
        seen_ids.add(sample_id)
        details.append(detail)
    return details


def prepare_checkpoint(
    output_tag: str,
    metadata: dict,
    resume: bool,
) -> tuple[Path, list[dict]]:
    checkpoint_path, metadata_path = checkpoint_paths(output_tag)
    checkpoint_exists = checkpoint_path.exists() or metadata_path.exists()

    if resume:
        if not checkpoint_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"未找到可恢复的 checkpoint：{checkpoint_path}。请去掉 --resume 或使用正确的 --output-tag。"
            )
        with open(metadata_path, "r", encoding="utf-8") as file:
            saved_metadata = json.load(file)
        if saved_metadata != metadata:
            raise ValueError(
                "checkpoint 与本次评测配置不一致；请保持数据集、检索模式和 --skip-judge 参数不变，"
                "或使用新的 --output-tag。"
            )
        return checkpoint_path, _load_checkpoint_details(checkpoint_path)

    if checkpoint_exists:
        raise FileExistsError(
            f"checkpoint 已存在：{checkpoint_path}。请使用 --resume 继续，或更换 --output-tag。"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    checkpoint_path.touch()
    return checkpoint_path, []


def append_checkpoint_detail(checkpoint_path: Path, detail: dict) -> None:
    with open(checkpoint_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(detail, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def run_retrieval_evaluation(
    retrieval_mode: str | None = None,
    output_tag: str = "latest",
    dataset_path: Path | None = None,
):
    """Evaluate retrieval metrics only, without any Agent or judge-model invocation."""
    active_dataset_path = dataset_path or DATASET_PATH
    if retrieval_mode:
        rag.set_retrieval_mode(retrieval_mode)
    active_retrieval_mode = rag.vector_store.retrieval_mode
    report_path, detail_path = output_paths(output_tag)
    samples = [sample for sample in load_samples(active_dataset_path) if is_retrieval_sample(sample)]
    details = []
    evidence_level = bool(samples) and all(sample.get("gold_evidence") for sample in samples)

    for sample in samples:
        if evidence_level:
            hit_at_1, hit_at_k, mrr_at_k, recall_at_k, latency_ms, retrieval_error, retrieved_sources = (
                retrieve_evidence_for_eval(sample["query"], sample["gold_evidence"])
            )
            details.append(
                {
                    "id": sample.get("id", ""),
                    "type": sample.get("type", "general"),
                    "query": sample["query"],
                    "evidence_hit_at_1": int(hit_at_1),
                    "evidence_hit_at_k": int(hit_at_k),
                    "evidence_mrr_at_k": mrr_at_k,
                    "evidence_recall_at_k": recall_at_k,
                    "retrieval_latency_ms": latency_ms,
                    "retrieved_sources": retrieved_sources,
                    "retrieval_error": retrieval_error,
                }
            )
            continue

        hit, recall_at_k, latency_ms, retrieval_error, retrieved_sources = retrieve_for_eval(
            sample["query"],
            sample.get("retrieval_expected_keywords", []),
            sample.get("gold_sources", []),
        )
        details.append(
            {
                "id": sample.get("id", ""),
                "type": sample.get("type", "general"),
                "query": sample["query"],
                "retrieval_hit": int(hit),
                "retrieval_recall_at_k": recall_at_k,
                "retrieval_latency_ms": latency_ms,
                "retrieved_sources": retrieved_sources,
                "retrieval_error": retrieval_error,
            }
        )

    total_samples = len(details)
    avg_retrieval_latency_ms = round(
        sum(item["retrieval_latency_ms"] for item in details) / total_samples,
        2,
    ) if total_samples else 0.0
    if evidence_level:
        hit_at_1_rate = round(sum(item["evidence_hit_at_1"] for item in details) / total_samples * 100, 2) if total_samples else 0.0
        hit_at_k_rate = round(sum(item["evidence_hit_at_k"] for item in details) / total_samples * 100, 2) if total_samples else 0.0
        avg_mrr_at_k = round(sum(item["evidence_mrr_at_k"] for item in details) / total_samples, 4) if total_samples else 0.0
        avg_evidence_recall_at_k = round(sum(item["evidence_recall_at_k"] for item in details) / total_samples, 2) if total_samples else 0.0
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_path": str(active_dataset_path),
            "evaluation_scope": "retrieval_only",
            "evaluation_protocol": "evidence_level_v2",
            "retrieval_mode": active_retrieval_mode,
            "retrieval_k": rag.vector_store.top_k,
            "total_samples": total_samples,
            "model_unavailable": False,
            "metrics": {
                "evidence_hit_at_1": hit_at_1_rate,
                "evidence_hit_at_k": hit_at_k_rate,
                "avg_evidence_mrr_at_k": avg_mrr_at_k,
                "avg_evidence_recall_at_k": avg_evidence_recall_at_k,
                "avg_retrieval_latency_ms": avg_retrieval_latency_ms,
            },
            "breakdown": {
                "retrieval_samples": total_samples,
                "evidence_hit_at_1_count": sum(item["evidence_hit_at_1"] for item in details),
                "evidence_hit_at_k_count": sum(item["evidence_hit_at_k"] for item in details),
            },
            "detail_path": str(detail_path),
        }
    else:
        top_k_hit_rate = round(sum(item["retrieval_hit"] for item in details) / total_samples * 100, 2) if total_samples else 0.0
        avg_recall_at_k = round(
            sum(item["retrieval_recall_at_k"] for item in details) / total_samples,
            2,
        ) if total_samples else 0.0
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_path": str(active_dataset_path),
            "evaluation_scope": "retrieval_only",
            "evaluation_protocol": "source_level_v1",
            "retrieval_mode": active_retrieval_mode,
            "retrieval_k": rag.vector_store.top_k,
            "total_samples": total_samples,
            "model_unavailable": False,
            "metrics": {
                "top_k_hit_rate": top_k_hit_rate,
                "avg_recall_at_k": avg_recall_at_k,
                "avg_retrieval_latency_ms": avg_retrieval_latency_ms,
            },
            "breakdown": {
                "retrieval_samples": total_samples,
                "retrieval_hit_count": sum(item["retrieval_hit"] for item in details),
            },
            "detail_path": str(detail_path),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    with open(detail_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(details[0].keys()) if details else [])
        if details:
            writer.writeheader()
            writer.writerows(details)

    print("Retrieval-only evaluation completed.")
    print(f"Retrieval mode: {active_retrieval_mode} (k={rag.vector_store.top_k})")
    print(f"Total retrieval samples: {total_samples}")
    if evidence_level:
        print(f"Evidence Hit@1: {report['metrics']['evidence_hit_at_1']}%")
        print(f"Evidence Hit@{rag.vector_store.top_k}: {report['metrics']['evidence_hit_at_k']}%")
        print(f"Average Evidence MRR@{rag.vector_store.top_k}: {report['metrics']['avg_evidence_mrr_at_k']}")
        print(f"Average Evidence Recall@{rag.vector_store.top_k}: {report['metrics']['avg_evidence_recall_at_k']}%")
    else:
        print(f"Top-K hit rate: {report['metrics']['top_k_hit_rate']}%")
        print(f"Average Recall@K: {report['metrics']['avg_recall_at_k']}%")
    print(f"Average retrieval latency: {avg_retrieval_latency_ms} ms")
    print(f"Report: {report_path}")
    print(f"Details: {detail_path}")
    return report


def run_evaluation(
    retrieval_mode: str | None = None,
    output_tag: str = "latest",
    enable_judge: bool = True,
    dataset_path: Path | None = None,
    resume: bool = False,
):
    active_dataset_path = dataset_path or DATASET_PATH
    samples = load_samples(active_dataset_path)
    if retrieval_mode:
        rag.set_retrieval_mode(retrieval_mode)
    active_retrieval_mode = rag.vector_store.retrieval_mode
    report_path, detail_path = output_paths(output_tag)
    sample_ids = [str(sample.get("id", "")).strip() for sample in samples]
    if not all(sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("端到端评测集中的每条样本必须具有唯一且非空的 id")

    metadata = _checkpoint_metadata(
        active_dataset_path,
        active_retrieval_mode,
        enable_judge,
        len(samples),
    )
    checkpoint_path, details = prepare_checkpoint(output_tag, metadata, resume)
    completed_ids = {str(detail["id"]) for detail in details}
    unknown_ids = completed_ids - set(sample_ids)
    if unknown_ids:
        raise ValueError(f"checkpoint 包含当前数据集不存在的样本：{sorted(unknown_ids)}")
    if details:
        print(f"从 checkpoint 恢复：已完成 {len(details)}/{len(samples)} 条样本。")

    agent = ReactAgent()
    completed_count = len(details)
    try:
        for sample in samples:
            sample_id = str(sample["id"])
            if sample_id in completed_ids:
                continue

            print(f"[{completed_count + 1}/{len(samples)}] 评测样本：{sample_id}")
            sample_type = sample.get("type", "general")
            retrieval_hit = ""
            retrieval_recall_at_k = ""
            retrieval_latency_ms = ""
            retrieval_error = ""
            retrieved_sources = ""
            judge_score = ""
            judge_passed = ""
            judge_reason = ""
            if is_retrieval_sample(sample):
                retrieval_hit, retrieval_recall_at_k, retrieval_latency_ms, retrieval_error, retrieved_sources = retrieve_for_eval(
                    sample["query"],
                    sample.get("retrieval_expected_keywords", []),
                    sample.get("gold_sources", []),
                )

            error_message = ""
            set_user_context(
                user_id=sample.get("user_id"),
                city=sample.get("user_city"),
            )
            try:
                result = agent.execute(
                    query=sample["query"],
                    history=sample.get("history", []),
                )
            except Exception as error:
                raise RuntimeError(
                    f"样本 {sample_id} 的 Agent 调用失败；此前完成的样本已保存，"
                    f"修复后使用 --resume 继续。原始错误：{str(error)[:200]}"
                ) from error
            finally:
                clear_user_context()

            answer = result["answer"]
            expected_keywords = sample.get("expected_keywords", [])
            expected_tools = sample.get("expected_tools", [])
            answer_correct = evaluate_answer(answer, expected_keywords)
            expected_tools_hit = evaluate_expected_tools(result["tool_calls"], expected_tools)
            if enable_judge and should_run_judge(sample):
                judge_score, judge_passed, judge_reason = judge_answer(sample, answer, retrieved_sources)

            detail = {
                "id": sample_id,
                "type": sample_type,
                "query": sample["query"],
                "has_expected_tools": int(bool(expected_tools)),
                "answer_correct": int(answer_correct),
                "expected_tools_hit": int(expected_tools_hit),
                "retrieval_hit": int(retrieval_hit) if retrieval_hit != "" else "",
                "retrieval_recall_at_k": retrieval_recall_at_k,
                "retrieval_latency_ms": retrieval_latency_ms,
                "latency_ms": result["latency_ms"],
                "tool_call_total": result["tool_call_total"],
                "tool_call_success": result["tool_call_success"],
                "tool_call_failed": result["tool_call_failed"],
                "tool_calls": "|".join(result["tool_calls"]),
                "retrieved_sources": retrieved_sources,
                "judge_score": judge_score,
                "judge_passed": judge_passed,
                "judge_reason": judge_reason,
                "answer_preview": answer[:120].replace("\n", " "),
                "error_message": error_message[:200],
                "retrieval_error": retrieval_error,
            }
            details.append(detail)
            append_checkpoint_detail(checkpoint_path, detail)
            completed_ids.add(sample_id)
            completed_count += 1
            print(f"[{completed_count}/{len(samples)}] 已保存 checkpoint：{sample_id}")
    except KeyboardInterrupt:
        print(
            f"\n评测已中断，已完成 {completed_count}/{len(samples)} 条。"
            f"使用 --resume --output-tag {output_tag} 可从 checkpoint 继续。"
        )
        raise

    total_samples = len(details)
    answer_correct_count = sum(item["answer_correct"] for item in details)
    total_tool_calls = sum(item["tool_call_total"] for item in details)
    total_tool_success = sum(item["tool_call_success"] for item in details)
    total_latency = sum(item["latency_ms"] for item in details)
    tool_expected_samples = [item for item in details if item["has_expected_tools"] == 1]
    retrieval_samples = [item for item in details if item["retrieval_hit"] != ""]
    multi_turn_samples = [item for item in details if item["type"] == "multi_turn"]

    retrieval_total = len(retrieval_samples)
    retrieval_hit_count = sum(int(item["retrieval_hit"]) for item in retrieval_samples)
    retrieval_recall_values = [
        float(item["retrieval_recall_at_k"])
        for item in retrieval_samples
        if item["retrieval_recall_at_k"] != ""
    ]
    retrieval_latency_values = [
        float(item["retrieval_latency_ms"])
        for item in retrieval_samples
        if item["retrieval_latency_ms"] != ""
    ]
    judge_samples = [item for item in details if item["judge_score"] != ""]
    judge_scores = [int(item["judge_score"]) for item in judge_samples]
    judge_pass_count = sum(int(item["judge_passed"]) for item in judge_samples if item["judge_passed"] != "")
    multi_turn_correct = sum(item["answer_correct"] for item in multi_turn_samples)
    tool_accuracy_count = sum(item["expected_tools_hit"] for item in tool_expected_samples)

    answer_accuracy = round((answer_correct_count / total_samples) * 100, 2) if total_samples else 0.0
    tool_success_rate = round((total_tool_success / total_tool_calls) * 100, 2) if total_tool_calls else 100.0
    tool_call_accuracy = round((tool_accuracy_count / len(tool_expected_samples)) * 100, 2) if tool_expected_samples else 0.0
    top_k_hit_rate = round((retrieval_hit_count / retrieval_total) * 100, 2) if retrieval_total else 0.0
    avg_recall_at_k = (
        round(sum(retrieval_recall_values) / len(retrieval_recall_values), 2)
        if retrieval_recall_values else 0.0
    )
    avg_retrieval_latency_ms = (
        round(sum(retrieval_latency_values) / len(retrieval_latency_values), 2)
        if retrieval_latency_values else 0.0
    )
    judge_pass_rate = (
        round((judge_pass_count / len(judge_samples)) * 100, 2)
        if judge_samples else 0.0
    )
    avg_judge_score = (
        round(sum(judge_scores) / len(judge_scores), 2)
        if judge_scores else 0.0
    )
    multi_turn_accuracy = (
        round((multi_turn_correct / len(multi_turn_samples)) * 100, 2)
        if multi_turn_samples else 0.0
    )
    avg_latency_ms = round(total_latency / total_samples, 2) if total_samples else 0.0

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(active_dataset_path),
        "evaluation_scope": "full",
        "retrieval_mode": active_retrieval_mode,
        "retrieval_k": rag.vector_store.top_k,
        "total_samples": total_samples,
        "model_unavailable": False,
        "metrics": {
            "answer_accuracy": answer_accuracy,
            "tool_success_rate": tool_success_rate,
            "tool_call_accuracy": tool_call_accuracy,
            "top_k_hit_rate": top_k_hit_rate,
            "avg_recall_at_k": avg_recall_at_k,
            "avg_retrieval_latency_ms": avg_retrieval_latency_ms,
            "judge_pass_rate": judge_pass_rate,
            "avg_judge_score": avg_judge_score,
            "multi_turn_accuracy": multi_turn_accuracy,
            "avg_latency_ms": avg_latency_ms,
        },
        "breakdown": {
            "answer_correct_count": answer_correct_count,
            "total_tool_calls": total_tool_calls,
            "total_tool_success": total_tool_success,
            "tool_expected_samples": len(tool_expected_samples),
            "retrieval_samples": retrieval_total,
            "retrieval_hit_count": retrieval_hit_count,
            "judge_samples": len(judge_samples),
            "judge_pass_count": judge_pass_count,
            "multi_turn_samples": len(multi_turn_samples),
            "multi_turn_correct_count": multi_turn_correct,
        },
        "detail_path": str(detail_path),
        "checkpoint_path": str(checkpoint_path),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "id", "type", "query", "has_expected_tools", "answer_correct", "expected_tools_hit", "retrieval_hit",
        "retrieval_recall_at_k", "retrieval_latency_ms", "latency_ms", "tool_call_total", "tool_call_success",
        "tool_call_failed", "tool_calls", "retrieved_sources", "judge_score", "judge_passed", "judge_reason",
        "answer_preview", "error_message", "retrieval_error",
    ]
    with open(detail_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)

    print("Evaluation completed.")
    print(f"Total samples: {total_samples}")
    print(f"Retrieval mode: {active_retrieval_mode} (k={rag.vector_store.top_k})")
    print(f"Answer accuracy: {answer_accuracy}%")
    print(f"Tool success rate: {tool_success_rate}%")
    print(f"Tool call accuracy: {tool_call_accuracy}%")
    print(f"Top-K hit rate: {top_k_hit_rate}%")
    print(f"Average Recall@K: {avg_recall_at_k}%")
    print(f"Average retrieval latency: {avg_retrieval_latency_ms} ms")
    print(f"Judge pass rate: {judge_pass_rate}%")
    print(f"Average judge score: {avg_judge_score}")
    print(f"Multi-turn accuracy: {multi_turn_accuracy}%")
    print(f"Average latency: {avg_latency_ms} ms")
    print(f"Report: {report_path}")
    print(f"Details: {detail_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Agent and RAG offline evaluation.")
    parser.add_argument("--retrieval-mode", choices=["vector", "hybrid"], default=None)
    parser.add_argument("--output-tag", default="latest", help="Output prefix, such as baseline or hybrid.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="JSONL dataset path. Defaults to evaluation/datasets/qa_samples.jsonl.",
    )
    parser.add_argument("--retrieval-only", action="store_true", help="Only calculate retrieval metrics without calling Agent models.")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-Judge to reduce model calls.")
    parser.add_argument("--resume", action="store_true", help="Resume a full evaluation from the checkpoint for the same output tag.")
    args = parser.parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve() if args.dataset else None
    if args.retrieval_only:
        run_retrieval_evaluation(
            retrieval_mode=args.retrieval_mode,
            output_tag=args.output_tag,
            dataset_path=dataset_path,
        )
    else:
        run_evaluation(
            retrieval_mode=args.retrieval_mode,
            output_tag=args.output_tag,
            enable_judge=not args.skip_judge,
            dataset_path=dataset_path,
            resume=args.resume,
        )
