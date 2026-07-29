from typing import Literal
# 从typing模块导入Literal类型，用于限制变量只能取指定的值

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.react_agent import ReactAgent
from agent.tools.agent_tools import clear_user_context, set_user_context
from utils.sqlite_store import sqlite_store


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
    conversation_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    city: str | None = Field(default=None)


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[str] = Field(default_factory=list)
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
    conversation_id = sqlite_store.ensure_conversation(request.conversation_id)
    stored_history = sqlite_store.list_messages(conversation_id)
    request_history = [m.model_dump() for m in request.history]
    history = stored_history or request_history

    if request_history and not stored_history:
        sqlite_store.seed_messages(conversation_id, request_history)

    set_user_context(user_id=request.user_id, city=request.city)
    try:
        result = agent.execute(
            query=request.query,
            history=history,
            # model_dump() 方法将模型实例转换为字典，用于传递给模型
        )
    finally:
        clear_user_context()

    sqlite_store.append_message(conversation_id, "user", request.query)
    sqlite_store.append_message(conversation_id, "assistant", result["answer"])
    return ChatResponse(conversation_id=conversation_id, **result)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    conversation_id = sqlite_store.ensure_conversation(request.conversation_id)
    stored_history = sqlite_store.list_messages(conversation_id)
    request_history = [m.model_dump() for m in request.history]
    history = stored_history or request_history

    if request_history and not stored_history:
        sqlite_store.seed_messages(conversation_id, request_history)

    def stream_generator():
        latest_chunk = ""
        set_user_context(user_id=request.user_id, city=request.city)
        try:
            for chunk in agent.execute_stream(
                query=request.query,
                history=history,
            ):
                latest_chunk = chunk
                yield chunk
        finally:
            clear_user_context()
            sqlite_store.append_message(conversation_id, "user", request.query)
            if latest_chunk.strip():
                sqlite_store.append_message(conversation_id, "assistant", latest_chunk.strip())

    return StreamingResponse(
        stream_generator(),
        media_type="text/plain; charset=utf-8",
        headers={"X-Conversation-Id": conversation_id},
    )
