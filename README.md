小曹老师全新录制《AI大模型RAG与Agent智能体开发项目实战课程》已经上线B站。

本仓库为`Agent项目`配套代码。

![](https://image-set.oss-cn-zhangjiakou.aliyuncs.com/img-out/20260122233149482.png)
https://www.bilibili.com/video/BV1yjz5BLEoY

![](https://image-set.oss-cn-zhangjiakou.aliyuncs.com/img-out/20260122233045951.png)
小曹老师个人B站主页：https://space.bilibili.com/1032221418
大家多多关注哦。

## 本地运行

### Streamlit 客服页面

```bash
streamlit run app.py
```

### FastAPI 接口服务

```bash
uvicorn api.main:app --reload
```

接口示例：

- `GET /health`
- `POST /chat`
- `POST /chat/stream`

### 自动评测与指标看板

先运行评测（默认自动生成 100 条评测样本）：

```bash
python -m evaluation.run_eval
```

然后启动 Streamlit，在页面中打开 `自动评测指标看板`：

```bash
streamlit run app.py
```
