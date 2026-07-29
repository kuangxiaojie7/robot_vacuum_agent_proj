# 面向扫地机器人的 Agentic RAG 智能客服平台

## 项目简介

本项目是一个面向扫地机器人场景的 `Agentic RAG` 智能客服系统，基于 `LangChain + LangGraph + Chroma + Streamlit + FastAPI` 实现，支持多轮对话、知识库检索、工具调用、报告生成和自动化评测。

项目目标是将传统问答型客服升级为可检索、可调用工具、可生成结构化结果的智能体系统，覆盖产品咨询、实时天气查询、用户信息获取和使用报告生成等典型场景。

## 核心功能

- `多轮对话`：支持历史消息传递与上下文连续追问。
- `RAG 检索增强`：对扫地机器人领域文档进行切分、向量化检索，并支持 BM25 + RRF 混合检索，返回来源文件与证据片段。
- `工具调用`：支持高德天气查询、用户城市上下文、用户 ID 与外部使用记录查询等能力。
- `报告生成`：结合用户记录与知识库内容输出结构化使用报告。
- `流式交互`：支持 Streamlit 页面流式输出。
- `服务化接口`：提供 FastAPI 同步与流式接口，便于前后端联调和第三方调用。
- `自动评测`：提供 150 条端到端样本、100 条检索开发集和 500 条证据级检索测试集，用于量化系统效果。

## 技术栈

- `LangChain`
- `LangGraph`
- `Chroma`
- `Streamlit`
- `FastAPI`
- `百炼 OpenAI-compatible Chat Completions / Embeddings`

## 项目结构

```text
robot_vacuum_agent_proj/
├── app.py                    # Streamlit 前端入口
├── api/main.py               # FastAPI 服务入口
├── agent/                    # Agent 核心逻辑与工具
├── rag/                      # RAG 检索与向量库
├── evaluation/               # 自动评测脚本与数据集
├── tests/                    # SQLite、API、工具与检索基础测试
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


复制环境变量模板并填入 API Key：

```bash
cp .env.example .env
```

`.env` 内容示例：

```dotenv
DASHSCOPE_API_KEY=your_dashscope_api_key
GAODE_API_KEY=your_gaode_web_service_api_key
```

`.env` 已被 Git 忽略；也可以通过系统环境变量注入这两个 Key，系统环境变量优先级更高。

配置模型，编辑 `config/rag.yml`：

```yaml
chat_model_name: qwen3.7-plus
judge_model_name: qwen3.7-max
embedding_model_name: text-embedding-v4
```

### 准备本地数据

项目提交了脱敏的示例知识库和用户记录，便于直接构建 RAG。运行时生成的 `data/*.db`、`chroma_db/` 与评测输出不会提交到 Git；接入真实业务数据时，请替换为脱敏数据，且不要提交真实用户信息。

```text
data/
├── *.txt / *.pdf / *.csv       # 用于构建 RAG 知识库的领域文档
└── external/records.csv        # 用户使用记录，供报告工具查询
```

`records.csv` 需要包含 `用户ID`、`时间`、`特征`、`清洁效率`、`耗材`、`对比` 等列。请使用脱敏或示例数据，不要提交真实用户信息。

## 运行方式

首次运行时，先构建 Chroma 向量库：

```bash
./.venv/bin/python -m rag.build_knowledge_base
```

检查知识库是否可用：

```bash
./.venv/bin/python -m rag.build_knowledge_base --check
```

更新 `data/` 下的知识库文件，或需要清空旧向量时，使用强制重建：

```bash
./.venv/bin/python -m rag.build_knowledge_base --rebuild
```

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

同步接口示例：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "合肥今天适合让机器人拖地吗？",
    "conversation_id": null,
    "user_id": "1001",
    "city": "合肥"
  }'
```

首次请求将返回服务端生成的 `conversation_id`；后续请求携带该值即可从 SQLite 读取该会话历史。当前 `user_id` 与 `city` 是业务上下文参数，项目尚未实现登录鉴权或会话归属校验。

## 自动评测

评测数据分为开发集和测试集：

- `evaluation/datasets/qa_samples.jsonl`：150 条端到端测试样本，包含 24 条上下文依赖的多轮样本和 78 条来源标注 RAG 样本。
- `evaluation/datasets/retrieval_dev.jsonl`：100 条独立检索开发样本，仅用于调节混合检索参数或验证候选方案。
- `evaluation/datasets/retrieval_test.jsonl`：500 条独立检索测试样本，覆盖产品 FAQ、扫拖 FAQ、故障排查、选购建议和技术 FAQ；使用场景化问法并保留 `source_question` 供审计，全部标注 `gold_sources`，仅用于最终检索指标。

检索开发/测试集由本地知识库的原始 FAQ、故障现象和选购主题改写为用户场景问法；原始题目会保存在 `source_question` 字段中，可使用下面命令重新构建：

```bash
./.venv/bin/python -m evaluation.build_retrieval_benchmark
```

运行评测：

```bash
./.venv/bin/python evaluation/run_eval.py
```

评测输出：

- `evaluation/output/latest_report.json`
- `evaluation/output/latest_details.csv`

支持统计的指标包括：

- `回答正确率`
- `工具调用成功率`
- `工具调用准确率`
- `Top-k Hit Rate`
- `Recall@k`
- `多轮对话正确率`
- `检索平均时延`
- `端到端平均响应时延`

调节混合检索参数时，先在开发集上比较不同方案：

```bash
./.venv/bin/python evaluation/run_eval.py --retrieval-only --dataset evaluation/datasets/retrieval_dev.jsonl --retrieval-mode vector --output-tag dev_vector
./.venv/bin/python evaluation/run_eval.py --retrieval-only --dataset evaluation/datasets/retrieval_dev.jsonl --retrieval-mode hybrid --output-tag dev_hybrid
./.venv/bin/python evaluation/compare_reports.py \
  --baseline evaluation/output/dev_vector_report.json \
  --optimized evaluation/output/dev_hybrid_report.json
```

确定方案后，在 500 条检索测试集上验证最终收益：

```bash
./.venv/bin/python evaluation/run_eval.py --retrieval-only --dataset evaluation/datasets/retrieval_test.jsonl --retrieval-mode vector --output-tag test_vector
./.venv/bin/python evaluation/run_eval.py --retrieval-only --dataset evaluation/datasets/retrieval_test.jsonl --retrieval-mode hybrid --output-tag test_hybrid
./.venv/bin/python evaluation/compare_reports.py \
  --baseline evaluation/output/test_vector_report.json \
  --optimized evaluation/output/test_hybrid_report.json
```

### 已验证的检索结果

在 500 条独立检索测试集上，BM25 + RRF 混合检索相较纯向量检索取得以下结果：

- `Top-K Hit Rate`：`97.2% -> 99.4%`（提升 `2.2` 个百分点）
- `Recall@3`：`71.08% -> 73.77%`（提升 `2.69` 个百分点）
- 平均检索时延：`199.26 ms -> 201.80 ms`（增加 `2.54 ms`）

因此项目默认使用 `hybrid` 检索模式；该模式结合语义召回与关键词匹配，并用 RRF 进行无量纲排名融合。

### 严格证据级 V2 评测

`retrieval_v2_dev.jsonl`（100 条）和 `retrieval_v2_test.jsonl`（500 条）是独立保存的严格评测版本，不修改上述来源文件级 V1 数据。每个样本标注 `gold_evidence`，要求返回的分片同时命中正确来源文件和对应知识段落，而不只是命中同一文件。

V2 额外统计：

- `Evidence Hit@1`：首个检索分片是否为标准证据。
- `Evidence Hit@3`：前三个检索分片是否包含标准证据。
- `Evidence MRR@3`：标准证据首次出现的平均倒数排名。

当前每条 V2 样本只标注一个最小标准证据段，因此 `Evidence Recall@3` 与 `Evidence Hit@3` 数值相同，不作为独立的核心结论；后续若扩展为多证据问题，可直接复用该字段计算真正的 Recall@3。

V2 数据在运行评测前已经冻结；它用于更严格地观察排序质量，不应用于反向修改检索策略或旧测试集。

```bash
./.venv/bin/python -m evaluation.build_retrieval_benchmark_v2

./.venv/bin/python evaluation/run_eval.py --retrieval-only --dataset evaluation/datasets/retrieval_v2_dev.jsonl --retrieval-mode vector --output-tag v2_dev_vector
./.venv/bin/python evaluation/run_eval.py --retrieval-only --dataset evaluation/datasets/retrieval_v2_dev.jsonl --retrieval-mode hybrid --output-tag v2_dev_hybrid
./.venv/bin/python evaluation/compare_reports.py \
  --baseline evaluation/output/v2_dev_vector_report.json \
  --optimized evaluation/output/v2_dev_hybrid_report.json

./.venv/bin/python evaluation/run_eval.py --retrieval-only --dataset evaluation/datasets/retrieval_v2_test.jsonl --retrieval-mode vector --output-tag v2_test_vector
./.venv/bin/python evaluation/run_eval.py --retrieval-only --dataset evaluation/datasets/retrieval_v2_test.jsonl --retrieval-mode hybrid --output-tag v2_test_hybrid
./.venv/bin/python evaluation/compare_reports.py \
  --baseline evaluation/output/v2_test_vector_report.json \
  --optimized evaluation/output/v2_test_hybrid_report.json
```

### V2 最终测试结果

在冻结的 500 条证据级场景化测试集上，BM25 + RRF 混合检索相较纯向量检索的结果如下：

- `Evidence Hit@1`：`43.2% -> 65.2%`（提升 `22.0` 个百分点）
- `Evidence Hit@3`：`64.6% -> 85.6%`（提升 `21.0` 个百分点）
- `Evidence MRR@3`：`0.5270 -> 0.7463`（提升 `0.2193`）
- 平均检索时延：`194.04 ms -> 208.72 ms`（增加 `14.68 ms`）

结果表明，混合检索能够明显提升正确证据的召回能力和排序位置，额外检索开销约为 `15 ms`。由于当前 V2 每条样本仅标注一个标准证据，`Evidence Recall@3` 与 `Evidence Hit@3` 数值一致，未作为独立结论展示。

最后使用端到端测试集统计 Agent、工具和多轮指标：

```bash
./.venv/bin/python evaluation/run_eval.py --retrieval-mode hybrid --output-tag final_hybrid --skip-judge
```

端到端评测会逐样本写入 checkpoint。若因网络、额度或手动 `Ctrl+C` 中断，修复问题后使用相同的 `output-tag` 继续，不会重复调用已经完成的样本：

```bash
./.venv/bin/python evaluation/run_eval.py \
  --retrieval-mode hybrid \
  --output-tag final_hybrid \
  --skip-judge \
  --resume
```

checkpoint 会保存为 `evaluation/output/final_hybrid_checkpoint.jsonl`；恢复时必须保持数据集、检索模式和是否启用 Judge 的参数一致。

## 自动化测试

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖会话持久化、天气降级、混合检索排序、评测指标和断点续跑等确定性逻辑；真实模型与外部 API 的网络行为需要在集成环境中单独验证。

## 项目亮点

- 基于 `LangChain + LangGraph` 搭建 ReAct 风格智能体，支持复杂问答场景下的动态工具调用。
- 实现 `RAG` 检索链路，对 `PDF/TXT/CSV` 文档进行向量化存储与检索增强。
- 接入 `高德 API` 提供实时天气查询，并支持手动城市上下文与服务端 IP 定位兜底。
- 提供 `FastAPI` 接口和 `Streamlit` 页面，兼顾本地演示与服务化调用。
- 构建 `500` 条证据级检索测试集和 `150` 条端到端评测集，量化评估检索、工具调用与多轮对话表现。
