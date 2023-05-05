import os
import sys
import logging

import pandas as pd
import streamlit as st

from streamlit_chat import message
from langchain.chat_models import ChatOpenAI

sys.path.append("../../")

from chat2plot import Chat2Plot

logger = logging.getLogger('root')
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Chat2Plot Demo", page_icon=":robot:")
st.header("Chat2Plot Demo")
st.subheader("Settings")

api_key = st.text_input("Step1: Input your OpenAI API-KEY", value="")
csv_file = st.file_uploader("Step2: Upload csv file", type={"csv"})

if api_key and csv_file:
    os.environ["OPENAI_API_KEY"] = api_key

    df = pd.read_csv(csv_file)

    st.write(df.head())
    model_name = st.selectbox("Step3: Choose model type", ("gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314"), index=0)

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    if "chat" not in st.session_state:
        st.session_state["chat"] = Chat2Plot(df, verbose=True)

    c2p = st.session_state["chat"]

    c2p.set_chatmodel(ChatOpenAI(temperature=0, model_name=model_name))

    st.subheader("Chat")

    def get_text():
        input_text = st.text_input("You: ", key="input")
        return input_text

    user_input = get_text()

    if user_input:
        with st.spinner(text="Wait for LLM response..."):
            res = c2p(user_input)
        plot = res.plot
        response_type = res.response

        st.session_state.past.append(user_input)
        if plot is not None:
            st.session_state.generated.append(plot)
        else:
            st.session_state.generated.append(str(response_type.value))

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            if isinstance(st.session_state["generated"][i], str):
                message(st.session_state["generated"][i], key=str(i))
            else:
                st.plotly_chart(st.session_state["generated"][i])

            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
