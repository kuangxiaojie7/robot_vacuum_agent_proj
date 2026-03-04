from langchain.agents import create_agent
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id, get_current_month,
                                     fetch_external_data, fill_context_for_report)
from model.factory import chat_model
from utils.prompt_loader import load_system_prompt
import time


class ReactAgent(object):
    def __init__(self, max_history_messages: int = 20):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompt(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id, get_current_month,
                   fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )
        self.max_history_messages = max_history_messages

    @staticmethod
    def _build_runtime_context(context=None):
        runtime_context = {
            "report": False,
            "tool_call_total": 0,
            "tool_call_success": 0,
            "tool_call_failed": 0,
            "tool_calls": [],
            "tool_call_failed_names": [],
        }
        if context:
            runtime_context.update(context)
        return runtime_context

    @staticmethod
    def _message_content_to_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                else:
                    parts.append(str(part))
            return "".join(parts)
        return str(content)

    def _normalize_history(self, history):
        if not history:
            return []

        normalized = []
        for msg in history:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if not role or content is None:
                continue
            if role not in {"user", "assistant", "system", "tool"}:
                continue
            normalized.append({"role": role, "content": str(content)})

        if self.max_history_messages and len(normalized) > self.max_history_messages:
            return normalized[-self.max_history_messages:]
        return normalized

    def _build_input_messages(self, query, history=None):
        messages = self._normalize_history(history)
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != query:
            messages.append({"role": "user", "content": query})
        return {"messages": messages}

    def execute(self, query, history=None, context=None):
        input_dict = self._build_input_messages(query, history)
        runtime_context = self._build_runtime_context(context)
        start = time.perf_counter()
        result = self.agent.invoke(input_dict, context=runtime_context)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        answer = ""
        messages = result.get("messages", [])
        if messages:
            answer = self._message_content_to_text(messages[-1].content).strip()

        return {
            "answer": answer,
            "latency_ms": latency_ms,
            "tool_call_total": int(runtime_context.get("tool_call_total", 0)),
            "tool_call_success": int(runtime_context.get("tool_call_success", 0)),
            "tool_call_failed": int(runtime_context.get("tool_call_failed", 0)),
            "tool_calls": list(runtime_context.get("tool_calls", [])),
            "tool_call_failed_names": list(runtime_context.get("tool_call_failed_names", [])),
        }

    def execute_stream(self, query, history=None, context=None):
        input_dict = self._build_input_messages(query, history)
        runtime_context = self._build_runtime_context(context)

        for chunk in self.agent.stream(input_dict, stream_mode="values", context=runtime_context):
            latest_message = chunk["messages"][-1]  # 有历史记录所以取最后一条
            if latest_message.content:
                yield self._message_content_to_text(latest_message.content).strip() + "\n"

if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("扫地机器人在我所在地区的气温下如何保养"):
        print(chunk, end="", flush=True)
