from langchain.agents import create_agent
import time
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

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

    def execute(self, query: str, history=None, context=None):
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

    def execute_stream(self, query: str, history=None, context=None):
        input_dict = self._build_input_messages(query, history)
        runtime_context = self._build_runtime_context(context)

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context=runtime_context):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield self._message_content_to_text(latest_message.content).strip() + "\n"
                
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
