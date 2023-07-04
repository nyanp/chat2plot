import json
import logging
import os
import subprocess
import sys
import time
import traceback

import pandas as pd
import streamlit as st
from plotly.graph_objs import Figure
from pydantic import BaseModel
from streamlit_chat import message

sys.path.append("../../")

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Chat2Plot Demo", page_icon=":robot:", layout="wide")
st.header("Chat2Plot Demo")


def dynamic_install(module):
    sleep_time = 30
    dependency_warning = st.warning(
        f"Installing dependencies, this takes {sleep_time} seconds."
    )
    subprocess.Popen([f"{sys.executable} -m pip install {module}"], shell=True)
    # wait for subprocess to install package before running your actual code below
    time.sleep(sleep_time)
    # remove the installing dependency warning
    dependency_warning.empty()


# https://python.plainenglish.io/how-to-install-your-own-private-github-package-on-streamlit-cloud-eb3aaed9b179
try:
    from chat2plot import ResponseType, chat2plot
    from chat2plot.chat2plot import Chat2Vega
except ModuleNotFoundError:
    github_token = st.secrets["github_token"]
    dynamic_install(f"git+https://{github_token}@github.com/nyanp/chat2plot.git")


def initialize_logger():
    logger = logging.getLogger("root")
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    logger.handlers = [handler]
    return True


if "logger" not in st.session_state:
    st.session_state["logger"] = initialize_logger()

api_key = st.text_input("Step1: Input your OpenAI API-KEY", value="", type="password")
csv_file = st.file_uploader("Step2: Upload csv file", type={"csv"})

if api_key and csv_file:
    os.environ["OPENAI_API_KEY"] = api_key

    df = pd.read_csv(csv_file)

    st.write(df.head())

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    st.subheader("Chat")

    def initialize_c2p():
        st.session_state["chat"] = chat2plot(
            df,
            st.session_state["chart_format"],
            verbose=True,
            description_strategy="head",
        )

    def reset_history():
        initialize_c2p()
        st.session_state["generated"] = []
        st.session_state["past"] = []

    with st.sidebar:
        chart_format = st.selectbox(
            "Chart format",
            ("simple", "vega"),
            key="chart_format",
            index=0,
            on_change=initialize_c2p,
        )

        st.button("Reset conversation history", on_click=reset_history)

    if "chat" not in st.session_state:
        initialize_c2p()

    c2p = st.session_state["chat"]

    chat_container = st.container()
    input_container = st.container()

    def submit():
        submit_text = st.session_state["input"]
        st.session_state["input"] = ""
        with st.spinner(text="Wait for LLM response..."):
            try:
                if isinstance(c2p, Chat2Vega):
                    res = c2p(submit_text, config_only=True)
                else:
                    res = c2p(submit_text, config_only=False, show_plot=False)
            except Exception:
                res = traceback.format_exc()
        st.session_state.past.append(submit_text)
        st.session_state.generated.append(res)

    def get_text():
        input_text = st.text_input("You: ", key="input", on_change=submit)
        return input_text

    with input_container:
        user_input = get_text()

    if st.session_state["generated"]:
        with chat_container:
            for i in range(
                len(st.session_state["generated"])
            ):  # range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

                res = st.session_state["generated"][i]

                if isinstance(res, str):
                    # something went wrong
                    st.error(res.replace("\n", "\n\n"))
                elif res.response_type == ResponseType.SUCCESS:
                    message(res.explanation, key=str(i))

                    col1, col2 = st.columns([2, 1])

                    with col2:
                        config = res.config
                        if isinstance(config, BaseModel):
                            st.code(
                                config.json(indent=2, exclude_none=True),
                                language="json",
                            )
                        else:
                            st.code(json.dumps(config, indent=2), language="json")
                    with col1:
                        if isinstance(res.figure, Figure):
                            st.plotly_chart(res.figure, use_container_width=True)
                        else:
                            st.vega_lite_chart(df, res.config, use_container_width=True)
                else:
                    st.warning(
                        f"Failed to render chart. last message: {res.conversation_history[-1].content}",
                        icon="⚠️",
                    )
                    # message(res.conversation_history[-1].content, key=str(i))
