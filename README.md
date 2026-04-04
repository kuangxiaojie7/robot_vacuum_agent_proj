# 面向扫地机器人的agetic rag 智能客服平台

## 项目简介

本项目是一个面向扫地机器人场景的 `Agentic RAG` 智能客服系统，基于 `LangChain + LangGraph + Chroma + Streamlit + FastAPI` 实现，支持多轮对话、知识库检索、工具调用、报告生成和自动化评测。

项目目标是将传统问答型客服升级为可检索、可调用工具、可生成结构化结果的智能体系统，覆盖产品咨询、实时天气查询、用户信息获取和使用报告生成等典型场景。

## 核心功能

- `多轮对话`：支持历史消息传递与上下文连续追问。
- `RAG 检索增强`：对扫地机器人领域文档进行切分、向量化与本地检索。
- `工具调用`：支持天气查询、用户城市获取、用户 ID 获取、外部记录查询等能力。
- `报告生成`：结合用户记录与知识库内容输出结构化使用报告。
- `流式交互`：支持 Streamlit 页面流式输出。
- `服务化接口`：提供 FastAPI 同步与流式接口，便于前后端联调和第三方调用。
- `自动评测`：内置 100 条样本和评测脚本，用于量化系统效果。

## 技术栈

- `LangChain`
- `LangGraph`
- `Chroma`
- `Streamlit`
- `FastAPI`
- `DashScope / ChatTongyi`

## 项目结构

```text
robot_vacuum_agent_proj/
├── app.py                    # Streamlit 前端入口
├── api/main.py               # FastAPI 服务入口
├── agent/                    # Agent 核心逻辑与工具
├── rag/                      # RAG 检索与向量库
├── evaluation/               # 自动评测脚本与数据集
├── config/                   # 模型、向量库、Prompt、Agent 配置
├── prompts/                  # Prompt 模板
├── data/                     # 知识库文档与外部记录
└── README.md
```

## 环境要求

- 可用的 `DASHSCOPE_API_KEY`
- 可用的高德 `Web 服务 API Key`

## 安装与配置

安装依赖：

```bash
uv sync
```


配置百炼 API Key：

```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
```

配置高德 Key，编辑 `config/agent.yml`：

```yaml
external_data_path: data/external/records.csv
gaodekey: 你的高德key
gaode_base_url: https://restapi.amap.com
gaode_timeout: 5
public_ip_sources:
  - https://ipv4.icanhazip.com
public_ip_timeout: 3
```

配置模型，编辑 `config/rag.yml`：

```yaml
chat_model_name: qwen3.5-flash
embedding_model_name: text-embedding-v4
```

## 运行方式

启动 Streamlit 页面：

```bash
streamlit run app.py
```

启动 FastAPI 服务：

```bash
uvicorn api.main:app --reload
```

接口文档地址：

```text
http://127.0.0.1:8000/docs
```

## 接口说明

- `GET /health`：服务健康检查
- `POST /chat`：同步返回完整回答与工具统计
- `POST /chat/stream`：流式返回回答内容

## 自动评测

评测数据位于：

`evaluation/datasets/qa_samples.jsonl`

运行评测：

```bash
python evaluation/run_eval.py
```

评测输出：

- `evaluation/output/latest_report.json`
- `evaluation/output/latest_details.csv`

支持统计的指标包括：

- `回答正确率`
- `工具调用成功率`
- `工具调用准确率`
- `端到端平均响应时延`

## 项目亮点

- 基于 `LangChain + LangGraph` 搭建 ReAct 风格智能体，支持复杂问答场景下的动态工具调用。
- 实现 `RAG` 检索链路，对 `PDF/TXT/CSV` 文档进行向量化存储与检索增强。
- 接入 `高德 API` 提供天气与定位能力，增强系统的实时信息处理能力。
- 提供 `FastAPI` 接口和 `Streamlit` 页面，兼顾本地演示与服务化调用。
- 构建 `100` 条自动化评测样本，用于量化评估 Agent 的回答效果与工具调用表现。
