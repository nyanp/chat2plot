import json
import re
import traceback
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import altair as alt
import jsonschema
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from plotly.graph_objs import Figure

from chat2plot.dataset_description import description
from chat2plot.render import draw_altair, draw_plotly
from chat2plot.schema import PlotConfig, ResponseType, get_schema_of_chart_config
from chat2plot.dictionary_helper import delete_null_field

_logger = getLogger(__name__)


def _build_prompt() -> str:
    schema_json = json.dumps(get_schema_of_chart_config(inlining_refs=False, remove_title=True), indent=2)
    return (
        """
Your task is to generate chart configuration for the given dataset and user question delimited by <>.

Responses should be in JSON format compliant to the following JSON Schema.


"""
        + schema_json.replace("{", "{{").replace("}", "}}")
        + """

The user's question may be an instruction to fine-tune the previous chart, or it may be an instruction to create a new chart based on a completely new context. In the latter case, be careful not to use the context used for the previous chart.

This is the result of `print(df.head())`:

{dataset}

You should do the following step by step:
1. Explain whether filters should be applied to the data, which chart_type and columns should be used, and what transformations are necessary to fulfill the user's request.
2. Generate schema-compliant JSON that represents 1.

Make sure to prefix the requested json string with triple backticks exactly and suffix the json with triple backticks exactly.
"""
    )

# If the user's question does not fall under any of above keys and is not a request about the appearance of the chart, simply reply "not related".


def _build_error_correcting_prompt(base_prompt: str) -> str:
    return (
        base_prompt
        + """
The user asked the following question:
{question}

You generated this json:
```
{generated_json}
```

It fails with the following error:
{error_message}

Correct the json and return a new json (do not import anything) that fixes the above mentioned error.
Do not generate the same json again.
    """
    )


_PROMPT = _build_prompt()


_PROMPT_VEGA = """
Your task is to generate chart configuration for the given dataset and user question delimited by <>.

Responses should be in JSON format compliant with the vega-lite specification, but `data` field must be excluded.

If the user's question does not fall under any of above keys and is not a request about the appearance of the chart, simply reply "not related".

This is the result of `print(df.head())`:

{dataset}

Make sure to prefix the requested json string with triple backticks exactly and suffix the json with triple backticks exactly.
"""


@dataclass(frozen=True)
class Plot:
    figure: alt.Chart | Figure | None
    config: PlotConfig | dict[str, Any] | None
    response_type: ResponseType
    raw_response: str


class ChatSession:
    """chat with conversasion history"""

    def __init__(
        self,
        df: pd.DataFrame,
        system_prompt_template: str,
        user_prompt_template: str,
        chat: BaseChatModel | None = None,
    ):
        self._system_prompt_template = system_prompt_template
        self._user_prompt_template = user_prompt_template
        self._chat = chat or ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")  # type: ignore
        self._conversation_history: list[BaseMessage] = [
            SystemMessage(
                content=system_prompt_template.format(dataset=description(df))
            )
        ]

    @property
    def history(self) -> list[BaseMessage]:
        return list(self._conversation_history)

    def set_chatmodel(self, chat: BaseChatModel) -> None:
        self._chat = chat

    def query_without_history(self, q: str) -> str:
        response = self._chat([HumanMessage(content=q)])
        return response.content

    def query(self, q: str) -> str:
        prompt = self._user_prompt_template.format(text=q)
        response = self._query(prompt)
        return response.content

    def _query(self, prompt: str) -> BaseMessage:
        self._conversation_history.append(HumanMessage(content=prompt))
        response = self._chat(self._conversation_history)
        self._conversation_history.append(response)
        return response

    def last_response(self) -> str:
        return self._conversation_history[-1].content


class Chat2PlotBase:
    @property
    def session(self) -> ChatSession:
        raise NotImplementedError()

    def query(self, q: str, config_only: bool = False, show_plot: bool = False) -> Plot:
        raise NotImplementedError()

    def __call__(
        self, q: str, config_only: bool = False, show_plot: bool = False
    ) -> Plot:
        return self.query(q, config_only, show_plot)


class Chat2Plot(Chat2PlotBase):
    def __init__(
        self, df: pd.DataFrame, chat: BaseChatModel | None = None, verbose: bool = False
    ):
        self._session = ChatSession(df, _PROMPT, "<{text}>", chat)
        self._df = df
        self._verbose = verbose

    @property
    def session(self) -> ChatSession:
        return self._session

    def query(self, q: str, config_only: bool = False, show_plot: bool = False) -> Plot:
        raw_response = self._session.query(q)

        try:
            return self._parse_response(raw_response, config_only, show_plot)
        except Exception as e:
            if self._verbose:
                _logger.info(f"first response: {raw_response}")
                _logger.warning(traceback.format_exc())

            msg = e.message if isinstance(e, jsonschema.ValidationError) else str(e)
            error_correction = _build_error_correcting_prompt(_PROMPT).format(
                dataset=description(self._df),
                question=q,
                generated_json=self._session.last_response(),
                error_message=msg,
            )
            corrected_response = self._session.query_without_history(error_correction)
            if self._verbose:
                _logger.info(f"retry response: {corrected_response}")

            try:
                return self._parse_response(corrected_response, config_only, show_plot)
            except Exception as e:
                if self._verbose:
                    _logger.warning(e)
                    _logger.warning(traceback.format_exc())
                return Plot(
                    None, None, ResponseType.FAILED_TO_RENDER, corrected_response
                )

    def __call__(
        self, q: str, config_only: bool = False, show_plot: bool = False
    ) -> Plot:
        return self.query(q, config_only, show_plot)

    def render(
        self, df: pd.DataFrame, config: PlotConfig, show_plot: bool = True
    ) -> Any:
        return draw_plotly(df, config, show_plot)

    def _parse_response(self, content: str, config_only: bool, show_plot: bool) -> Plot:
        if content == "not related":
            return Plot(None, None, ResponseType.NOT_RELATED, content)

        json_data = delete_null_field(parse_json(content))
        jsonschema.validate(json_data, PlotConfig.schema())
        config = PlotConfig.from_json(json_data)
        if self._verbose:
            _logger.info(config)

        if config_only:
            return Plot(None, config, ResponseType.SUCCESS, content)

        figure = self.render(self._df, config, show_plot)
        return Plot(figure, config, ResponseType.SUCCESS, content)


class Chat2Vega(Chat2PlotBase):
    def __init__(
        self, df: pd.DataFrame, chat: BaseChatModel | None = None, verbose: bool = False
    ):
        self._session = ChatSession(df, _PROMPT_VEGA, "<{text}>", chat)
        self._df = df
        self._verbose = verbose

    @property
    def session(self) -> ChatSession:
        return self._session

    def query(self, q: str, config_only: bool = False, show_plot: bool = False) -> Plot:
        res = self._session.query(q)
        if res == "not related":
            return Plot(None, None, ResponseType.NOT_RELATED, res)

        try:
            config = parse_json(res)
            if "data" in config:
                del config["data"]
            if self._verbose:
                _logger.info(config)
        except Exception:
            _logger.warning(f"failed to parse LLM response: {res}")
            _logger.warning(traceback.format_exc())
            return Plot(None, None, ResponseType.UNKNOWN, res)

        if config_only:
            return Plot(None, config, ResponseType.SUCCESS, res)

        try:
            plot = draw_altair(self._df, config, show_plot)
            return Plot(plot, config, ResponseType.SUCCESS, res)
        except Exception:
            _logger.warning(traceback.format_exc())
            return Plot(None, config, ResponseType.FAILED_TO_RENDER, res)

    def __call__(
        self, q: str, config_only: bool = False, show_plot: bool = False
    ) -> Plot:
        return self.query(q, config_only, show_plot)


def chat2plot(
    df: pd.DataFrame,
    model_type: str = "default",
    chat: BaseChatModel | None = None,
    verbose: bool = False,
) -> Chat2PlotBase:
    if model_type == "default":
        return Chat2Plot(df, chat, verbose)
    elif model_type == "vega":
        return Chat2Vega(df, chat, verbose)
    else:
        raise ValueError(
            f"model_type should be one of [default, vega] (given: {model_type})"
        )


def parse_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)  # type: ignore
    except ValueError:
        ptn = r"```json(.*)```" if "```json" in content else r"```(.*)```"
        s = re.search(ptn, content, re.MULTILINE | re.DOTALL)
        if s:
            return json.loads(s.group(1))  # type: ignore

        # sometimes LLM forgets start marker
        ptn = r"(.*)```"
        s = re.search(ptn, content, re.MULTILINE | re.DOTALL)
        if s:
            return json.loads(s.group(1))  # type: ignore

        raise
