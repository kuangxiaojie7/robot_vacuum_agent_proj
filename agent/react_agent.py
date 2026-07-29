from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
import time
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from rag.rag_service import RagSummarizeService


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    #将消息内容转换为文本，用于后续处理
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

    @staticmethod
    def _normalize_history(history):
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
        return normalized

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

    def _build_input_messages(self, query: str, history=None):
        messages = self._normalize_history(history)
        # 首次对话，新查询，连续相同查询
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != query:
            messages.append({"role": "user", "content": query})
        return {"messages": messages}

    @classmethod
    def _extract_rag_sources(cls, messages) -> list[str]:
        sources = []
        for message in messages:
            if not isinstance(message, ToolMessage) or getattr(message, "name", "") != "rag_summarize":
                continue
            content = cls._message_content_to_text(message.content)
            sources.extend(RagSummarizeService.extract_source_references(content))
        return list(dict.fromkeys(sources))

    @staticmethod
    def _append_sources(answer: str, sources: list[str]) -> str:
        if not sources or "【检索来源】" in answer or "参考来源：" in answer:
            return answer
        return f"{answer}\n\n参考来源：\n" + "\n".join(sources)

    def execute(self, query: str, history=None, context=None):
        input_dict = self._build_input_messages(query, history)
        runtime_context = self._build_runtime_context(context)
        start = time.perf_counter()
        result = self.agent.invoke(input_dict, context=runtime_context)
        # perf_counter()记录当前时间，单位是秒，2位小数，转换为毫秒
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        answer = ""
        messages = result.get("messages", [])
        if messages:
            answer = self._message_content_to_text(messages[-1].content).strip()
        sources = self._extract_rag_sources(messages)
        answer = self._append_sources(answer, sources)

        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": latency_ms,
            "tool_call_total": int(runtime_context.get("tool_call_total", 0)),
            "tool_call_success": int(runtime_context.get("tool_call_success", 0)),
            "tool_call_failed": int(runtime_context.get("tool_call_failed", 0)),
            "tool_calls": list(runtime_context.get("tool_calls", [])),
            "tool_call_failed_names": list(runtime_context.get("tool_call_failed_names", [])),
        }

    def execute_stream(self, query: str, history=None, context=None):
        input_dict = self._build_input_messages(query, history)
        runtime_context = self._build_runtime_context(context)
        sources = []

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context=runtime_context):
            latest_message = chunk["messages"][-1]
            if isinstance(latest_message, ToolMessage) and getattr(latest_message, "name", "") == "rag_summarize":
                content = self._message_content_to_text(latest_message.content)
                sources = list(dict.fromkeys(sources + RagSummarizeService.extract_source_references(content)))
                continue
            # 这里chunk是一个字典，每次更新messages，然后返回给streamlit显示
            if latest_message.content:
                content = self._message_content_to_text(latest_message.content).strip()
                if isinstance(latest_message, AIMessage) and not latest_message.tool_calls:
                    content = self._append_sources(content, sources)
                yield content + "\n"
                
        '''
        stream_mode 参数控制流输出的格式：
        "values"：只输出最终的结果值，不包含中间状态
        "updates"：输出所有的中间状态更新
        "messages"：只输出新生成的消息
        
        chunk = {
            "messages": [
                # 所有历史消息 + 新生成的消息
                {"role": "user", "content": "给我生成我的使用报告"},
                {"role": "assistant", "content": "好的，我将为您生成使用报告。"},
                # 可能还有工具调用和工具响应消息
            ]
        }
        '''


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
