import time

import streamlit as st
from agent.react_agent import ReactAgent
from agent.tools.agent_tools import set_user_context

# 标题
st.title("智扫通机器人智能客服")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

with st.sidebar:
    st.subheader("用户上下文")
    user_id = st.text_input("用户ID", value=st.session_state.get("user_id", ""))
    user_city = st.text_input("城市", value=st.session_state.get("user_city", ""))
    st.session_state["user_id"] = user_id
    st.session_state["user_city"] = user_city

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})
    set_user_context(
        user_id=st.session_state.get("user_id", "").strip(),
        city=st.session_state.get("user_city", "").strip(),
    )

    response_messages = []
    with st.spinner("智能客服思考中..."):
        history = st.session_state["message"][:-1]
        res_stream = st.session_state["agent"].execute_stream(prompt, history=history)

        def capture(generator, cache_list):

            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        st.rerun()

