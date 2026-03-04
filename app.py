import os
import random
import time

import streamlit as st
from agent.react_agent import ReactAgent
from agent.tools.agent_tools import set_user_context, USER_ID_POOL

st.title("智扫通机器人智能客服")
st.divider()

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，我是智扫通机器人智能客服，请问有什么可以帮助你？"}]

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "user_id" not in st.session_state:
    st.session_state["user_id"] = os.getenv("AGENT_USER_ID") or random.choice(USER_ID_POOL)

if "user_city" not in st.session_state:
    st.session_state["user_city"] = os.getenv("AGENT_USER_CITY") or ""

with st.sidebar:
    st.subheader("用户信息")
    st.text_input("用户ID", key="user_id")
    st.text_input("城市", key="user_city")

set_user_context(
    user_id=st.session_state.get("user_id") or None,
    city=st.session_state.get("user_city") or None,
)

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 在页面最下方提供用户输入栏
prompt = st.chat_input()

if prompt:
    # 在页面输出用户的提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(
            prompt,
            history=st.session_state["message"],
        )

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        st.rerun()
