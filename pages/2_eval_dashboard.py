import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "evaluation" / "output" / "latest_report.json"
DETAIL_PATH = PROJECT_ROOT / "evaluation" / "output" / "latest_details.csv"


st.title("自动评测指标看板")

if not REPORT_PATH.exists():
    st.info("还没有评测结果，请先运行：python -m evaluation.run_eval")
    st.stop()

with open(REPORT_PATH, "r", encoding="utf-8") as f:
    report = json.load(f)

if report.get("model_unavailable"):
    st.warning("评测过程中模型不可用，结果中可能包含失败样本。请检查模型网络和 API Key 后重跑。")

metrics = report.get("metrics", {})
col1, col2, col3 = st.columns(3)
col1.metric("回答正确率", f"{metrics.get('answer_accuracy', 0)}%")
col2.metric("工具调用成功率", f"{metrics.get('tool_success_rate', 0)}%")
col3.metric("平均响应时延", f"{metrics.get('avg_latency_ms', 0)} ms")

st.caption(f"样本数：{report.get('total_samples', 0)} | 生成时间：{report.get('generated_at', '-')}")

if DETAIL_PATH.exists():
    df = pd.read_csv(DETAIL_PATH)

    st.subheader("工具调用分布")
    tool_series = df["tool_calls"].fillna("")
    tool_counter = {}
    for item in tool_series:
        for tool_name in str(item).split("|"):
            name = tool_name.strip()
            if not name:
                continue
            tool_counter[name] = tool_counter.get(name, 0) + 1
    if tool_counter:
        tool_df = pd.DataFrame(
            [{"tool": name, "count": count} for name, count in tool_counter.items()]
        ).sort_values(by="count", ascending=False)
        st.bar_chart(tool_df.set_index("tool"))
    else:
        st.write("暂无工具调用数据。")

    st.subheader("评测明细")
    st.dataframe(df, use_container_width=True)
else:
    st.warning("评测汇总已生成，但未找到明细文件。")
