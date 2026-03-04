from typing import Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.react_agent import ReactAgent
from agent.tools.agent_tools import set_user_context


app = FastAPI(
    title="Heima Agent API",
    description="RAG + Agent chat service",
    version="0.1.0",
)
agent = ReactAgent()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User input query")
    history: list[ChatMessage] = Field(default_factory=list)
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
    return {"status": "ok", "service": "heima-agent-api"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    set_user_context(user_id=request.user_id, city=request.city)
    result = agent.execute(
        query=request.query,
        history=[m.model_dump() for m in request.history],
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
