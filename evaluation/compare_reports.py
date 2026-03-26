import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_latency_p95(detail_path: str | None) -> float | None:
    if not detail_path:
        return None
    detail_file = Path(detail_path)
    if not detail_file.exists():
        return None

    latencies = []
    with open(detail_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row.get("latency_ms")
            if value is None or value == "":
                continue
            try:
                latencies.append(float(value))
            except ValueError:
                continue

    if not latencies:
        return None

    latencies.sort()
    idx = int(0.95 * (len(latencies) - 1))
    return round(latencies[idx], 2)


def safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def calc_metric_change(baseline: float, optimized: float) -> dict:
    abs_change = round(optimized - baseline, 2)
    rel_change = None
    if baseline != 0:
        rel_change = round((optimized - baseline) / baseline * 100, 2)
    elif optimized == 0:
        rel_change = 0.0

    return {
        "baseline": baseline,
        "optimized": optimized,
        "absolute_change": abs_change,
        "relative_change_pct": rel_change,
    }


def calc_latency_change(baseline: float, optimized: float) -> dict:
    delta_ms = round(optimized - baseline, 2)
    reduction_ms = round(baseline - optimized, 2)

    reduction_pct = None
    if baseline != 0:
        reduction_pct = round((baseline - optimized) / baseline * 100, 2)
    elif optimized == 0:
        reduction_pct = 0.0

    return {
        "baseline": baseline,
        "optimized": optimized,
        "delta_ms": delta_ms,
        "reduction_ms": reduction_ms,
        "reduction_pct": reduction_pct,
    }


def format_change_line(name: str, change: dict, unit: str = "%", higher_is_better: bool = True) -> str:
    b = change["baseline"]
    o = change["optimized"]
    abs_change = change["absolute_change"]
    rel_change = change["relative_change_pct"]

    direction = "提升" if abs_change >= 0 else "下降"
    if not higher_is_better:
        direction = "下降" if abs_change <= 0 else "上升"

    if rel_change is None:
        rel_text = "相对变化：N/A（基线为0）"
    else:
        rel_text = f"相对变化：{rel_change}%"

    return (
        f"- {name}: {b}{unit} -> {o}{unit}，"
        f"{direction}{abs(abs_change)} 个百分点，{rel_text}"
    )


def format_latency_line(change: dict, label: str = "平均响应时延") -> str:
    b = change["baseline"]
    o = change["optimized"]
    reduction_ms = change["reduction_ms"]
    reduction_pct = change["reduction_pct"]

    if reduction_pct is None:
        return f"- {label}: {b} ms -> {o} ms，降幅：N/A（基线为0）"
    return f"- {label}: {b} ms -> {o} ms，降低 {reduction_ms} ms（{reduction_pct}%）"


def build_markdown(result: dict) -> str:
    lines = []
    lines.append("## 评测对比结果")
    lines.append("")
    lines.append(f"- 基线报告: `{result['baseline_report']}`")
    lines.append(f"- 优化后报告: `{result['optimized_report']}`")
    lines.append(f"- 样本量: baseline={result['baseline_total_samples']} / optimized={result['optimized_total_samples']}")
    lines.append("")

    if result["warnings"]:
        lines.append("## 注意事项")
        for item in result["warnings"]:
            lines.append(f"- {item}")
        lines.append("")

    metrics = result["metrics"]
    lines.append("## 核心指标")
    lines.append(format_change_line("回答正确率", metrics["answer_accuracy"], unit="%", higher_is_better=True))
    lines.append(format_change_line("工具调用成功率", metrics["tool_success_rate"], unit="%", higher_is_better=True))
    lines.append(format_latency_line(metrics["avg_latency_ms"], label="平均响应时延"))
    if metrics.get("p95_latency_ms") is not None:
        lines.append(format_latency_line(metrics["p95_latency_ms"], label="P95响应时延"))
    lines.append("")

    lines.append("## 简历可用表述")
    aa = metrics["answer_accuracy"]
    ts = metrics["tool_success_rate"]
    al = metrics["avg_latency_ms"]
    lines.append(
        "- 在统一评测集上，回答正确率由 "
        f"{aa['baseline']}% 提升至 {aa['optimized']}%（+{aa['absolute_change']} 个百分点）；"
        f"工具调用成功率由 {ts['baseline']}% 提升至 {ts['optimized']}%（+{ts['absolute_change']} 个百分点）；"
        f"平均响应时延由 {al['baseline']}ms 降至 {al['optimized']}ms（{al['reduction_pct']}%）。"
    )
    return "\n".join(lines)


def compare_reports(baseline_path: Path, optimized_path: Path) -> dict:
    baseline = load_json(baseline_path)
    optimized = load_json(optimized_path)

    baseline_metrics = baseline.get("metrics", {})
    optimized_metrics = optimized.get("metrics", {})

    answer_accuracy = calc_metric_change(
        safe_float(baseline_metrics.get("answer_accuracy", 0.0)),
        safe_float(optimized_metrics.get("answer_accuracy", 0.0)),
    )
    tool_success_rate = calc_metric_change(
        safe_float(baseline_metrics.get("tool_success_rate", 0.0)),
        safe_float(optimized_metrics.get("tool_success_rate", 0.0)),
    )
    avg_latency_ms = calc_latency_change(
        safe_float(baseline_metrics.get("avg_latency_ms", 0.0)),
        safe_float(optimized_metrics.get("avg_latency_ms", 0.0)),
    )

    baseline_p95 = read_latency_p95(baseline.get("detail_path"))
    optimized_p95 = read_latency_p95(optimized.get("detail_path"))
    p95_latency_ms = None
    if baseline_p95 is not None and optimized_p95 is not None:
        p95_latency_ms = calc_latency_change(baseline_p95, optimized_p95)

    warnings = []
    if baseline.get("model_unavailable"):
        warnings.append("基线报告标记为 model_unavailable=true，基线数据可能不可靠。")
    if optimized.get("model_unavailable"):
        warnings.append("优化后报告标记为 model_unavailable=true，优化数据可能不可靠。")
    if baseline.get("total_samples") != optimized.get("total_samples"):
        warnings.append("两份报告样本数不一致，建议使用相同评测集重跑。")
    if baseline.get("dataset_path") != optimized.get("dataset_path"):
        warnings.append("两份报告的数据集路径不同，建议使用同一份数据集。")

    result = {
        "baseline_report": str(baseline_path),
        "optimized_report": str(optimized_path),
        "baseline_total_samples": baseline.get("total_samples"),
        "optimized_total_samples": optimized.get("total_samples"),
        "metrics": {
            "answer_accuracy": answer_accuracy,
            "tool_success_rate": tool_success_rate,
            "avg_latency_ms": avg_latency_ms,
            "p95_latency_ms": p95_latency_ms,
        },
        "warnings": warnings,
    }
    return result


def print_summary(result: dict):
    print("=== 评测对比结果 ===")
    print(f"baseline: {result['baseline_report']}")
    print(f"optimized: {result['optimized_report']}")
    print(
        "samples: "
        f"{result['baseline_total_samples']} -> {result['optimized_total_samples']}"
    )
    print("")
    if result["warnings"]:
        print("注意事项:")
        for item in result["warnings"]:
            print(f"- {item}")
        print("")

    metrics = result["metrics"]
    print(format_change_line("回答正确率", metrics["answer_accuracy"], unit="%", higher_is_better=True))
    print(format_change_line("工具调用成功率", metrics["tool_success_rate"], unit="%", higher_is_better=True))
    print(format_latency_line(metrics["avg_latency_ms"], label="平均响应时延"))
    if metrics.get("p95_latency_ms") is not None:
        print(format_latency_line(metrics["p95_latency_ms"], label="P95响应时延"))


def main():
    parser = argparse.ArgumentParser(description="Compare two evaluation reports and calculate performance deltas.")
    parser.add_argument("--baseline", required=True, help="Path to baseline report json")
    parser.add_argument("--optimized", required=True, help="Path to optimized report json")
    parser.add_argument("--out-json", default="", help="Optional output path for comparison json")
    parser.add_argument("--out-md", default="", help="Optional output path for markdown summary")
    args = parser.parse_args()

    baseline_path = Path(args.baseline).resolve()
    optimized_path = Path(args.optimized).resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline report not found: {baseline_path}")
    if not optimized_path.exists():
        raise FileNotFoundError(f"Optimized report not found: {optimized_path}")

    result = compare_reports(baseline_path, optimized_path)
    print_summary(result)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved comparison json: {out_json}")

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        md = build_markdown(result)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"Saved markdown summary: {out_md}")


if __name__ == "__main__":
    main()
