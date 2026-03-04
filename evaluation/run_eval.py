import csv
import json
from datetime import datetime
from pathlib import Path

from agent.react_agent import ReactAgent
from agent.tools.agent_tools import set_user_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "qa_samples.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "output"
REPORT_PATH = OUTPUT_DIR / "latest_report.json"
DETAIL_PATH = OUTPUT_DIR / "latest_details.csv"


def generate_default_samples(total_samples: int = 100) -> list[dict]:
    cities = ["深圳", "杭州", "合肥", "上海", "北京"]
    user_ids = ["1001", "1002", "1003", "1004", "1005"]
    months = [
        "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
        "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
    ]
    knowledge_topics = [
        "扫地机器人吸力变弱怎么处理",
        "扫地机器人为什么会漏扫",
        "拖布多久清洗一次合适",
        "扫地机器人边刷多久更换",
        "地毯场景下要注意什么",
    ]

    samples = []
    for idx in range(total_samples):
        category = idx % 3
        if category == 0:
            query = f"{knowledge_topics[idx % len(knowledge_topics)]}？"
            samples.append(
                {
                    "id": f"knowledge_{idx + 1:03d}",
                    "query": query,
                    "expected_keywords": ["扫地机器人"],
                    "expected_tools": ["rag_summarize"],
                }
            )
            continue

        if category == 1:
            city = cities[idx % len(cities)]
            query = f"{city}今天的天气适合扫地机器人拖地吗？"
            samples.append(
                {
                    "id": f"weather_{idx + 1:03d}",
                    "query": query,
                    "user_city": city,
                    "expected_keywords": [city],
                    "expected_tools": ["get_weather"],
                }
            )
            continue

        user_id = user_ids[idx % len(user_ids)]
        month = months[idx % len(months)]
        query = f"请帮我生成{month}的扫地机器人使用报告。"
        samples.append(
            {
                "id": f"report_{idx + 1:03d}",
                "query": query,
                "user_id": user_id,
                "expected_keywords": ["报告"],
                "expected_tools": ["fill_context_for_report", "fetch_external_data"],
            }
        )

    return samples


def save_samples(samples: list[dict], dataset_path: Path):
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_samples(dataset_path: Path) -> list[dict]:
    if not dataset_path.exists():
        samples = generate_default_samples(100)
        save_samples(samples, dataset_path)

    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_answer(answer: str, expected_keywords: list[str]) -> bool:
    if not expected_keywords:
        return bool(answer.strip())
    return all(keyword in answer for keyword in expected_keywords)


def evaluate_expected_tools(called_tools: list[str], expected_tools: list[str]) -> bool:
    if not expected_tools:
        return True
    return all(tool_name in called_tools for tool_name in expected_tools)


def run_evaluation():
    samples = load_samples(DATASET_PATH)
    agent = ReactAgent()
    details = []
    model_unavailable = False
    model_error_hint = ""

    for sample in samples:
        set_user_context(
            user_id=sample.get("user_id"),
            city=sample.get("user_city"),
        )
        error_message = ""
        if model_unavailable:
            result = {
                "answer": "",
                "latency_ms": 0.0,
                "tool_call_total": 0,
                "tool_call_success": 0,
                "tool_call_failed": 0,
                "tool_calls": [],
                "tool_call_failed_names": [],
            }
            error_message = model_error_hint
        else:
            try:
                result = agent.execute(
                    query=sample["query"],
                    history=sample.get("history", []),
                )
            except Exception as e:
                model_unavailable = True
                model_error_hint = str(e)[:200]
                result = {
                    "answer": "",
                    "latency_ms": 0.0,
                    "tool_call_total": 0,
                    "tool_call_success": 0,
                    "tool_call_failed": 0,
                    "tool_calls": [],
                    "tool_call_failed_names": [],
                }
                error_message = model_error_hint

        answer = result["answer"]
        expected_keywords = sample.get("expected_keywords", [])
        expected_tools = sample.get("expected_tools", [])
        answer_correct = evaluate_answer(answer, expected_keywords)
        expected_tools_hit = evaluate_expected_tools(result["tool_calls"], expected_tools)

        details.append(
            {
                "id": sample.get("id", ""),
                "query": sample["query"],
                "answer_correct": int(answer_correct),
                "expected_tools_hit": int(expected_tools_hit),
                "latency_ms": result["latency_ms"],
                "tool_call_total": result["tool_call_total"],
                "tool_call_success": result["tool_call_success"],
                "tool_call_failed": result["tool_call_failed"],
                "tool_calls": "|".join(result["tool_calls"]),
                "answer_preview": answer[:120].replace("\n", " "),
                "error_message": error_message[:200],
            }
        )

    total_samples = len(details)
    answer_correct_count = sum(item["answer_correct"] for item in details)
    total_tool_calls = sum(item["tool_call_total"] for item in details)
    total_tool_success = sum(item["tool_call_success"] for item in details)
    total_latency = sum(item["latency_ms"] for item in details)

    answer_accuracy = round((answer_correct_count / total_samples) * 100, 2) if total_samples else 0.0
    tool_success_rate = round((total_tool_success / total_tool_calls) * 100, 2) if total_tool_calls else 100.0
    avg_latency_ms = round(total_latency / total_samples, 2) if total_samples else 0.0

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(DATASET_PATH),
        "total_samples": total_samples,
        "model_unavailable": model_unavailable,
        "metrics": {
            "answer_accuracy": answer_accuracy,
            "tool_success_rate": tool_success_rate,
            "avg_latency_ms": avg_latency_ms,
        },
        "breakdown": {
            "answer_correct_count": answer_correct_count,
            "total_tool_calls": total_tool_calls,
            "total_tool_success": total_tool_success,
        },
        "detail_path": str(DETAIL_PATH),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "id", "query", "answer_correct", "expected_tools_hit", "latency_ms",
        "tool_call_total", "tool_call_success", "tool_call_failed", "tool_calls", "answer_preview", "error_message",
    ]
    with open(DETAIL_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)

    print("Evaluation completed.")
    print(f"Total samples: {total_samples}")
    print(f"Answer accuracy: {answer_accuracy}%")
    print(f"Tool success rate: {tool_success_rate}%")
    print(f"Average latency: {avg_latency_ms} ms")
    print(f"Report: {REPORT_PATH}")
    print(f"Details: {DETAIL_PATH}")


if __name__ == "__main__":
    run_evaluation()
