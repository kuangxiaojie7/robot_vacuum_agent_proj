from typing import Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.react_agent import ReactAgent
from agent.tools.agent_tools import set_user_context


app = FastAPI(
    title="zhisaotong Agent API",
    description="RAG + Agent chat service",
    version="0.1.0",
)
agent = ReactAgent()


class ChatMessage(BaseModel):
    # literal类型表示只能取指定的值，不能取其他值
    # 例如，role 只能取 user、assistant、system、tool 四个值
    role: Literal["user", "assistant", "system", "tool"]
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User input query")
    history: list[ChatMessage] = Field(default_factory=list)
    # default_factory指定一个函数来生成默认值，如list, dict, set, tuple 等
    user_id: str | None = Field(default=None)
    city: str | None = Field(default=None)


class ChatResponse(BaseModel):
    answer: str
    latency_ms: float
    tool_call_total: int
    tool_call_success: int
    tool_call_failed: int
    tool_calls: list[str]
    tool_call_failed_names: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "service": "zhisaotong-agent-api"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    set_user_context(user_id=request.user_id, city=request.city)
    result = agent.execute(
        query=request.query,
        history=[m.model_dump() for m in request.history],
        # model_dump() 方法将模型实例转换为字典，用于传递给模型
    )
    return ChatResponse(**result)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    set_user_context(user_id=request.user_id, city=request.city)

    def stream_generator():
        for chunk in agent.execute_stream(
            query=request.query,
            history=[m.model_dump() for m in request.history],
        ):
            yield chunk

    return StreamingResponse(stream_generator(), media_type="text/plain; charset=utf-8")
