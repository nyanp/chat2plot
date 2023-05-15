import copy
import re
import traceback
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import altair as alt
import commentjson
import jsonschema
import pandas as pd
import pydantic
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from plotly.graph_objs import Figure

from chat2plot.dataset_description import description
from chat2plot.dictionary_helper import delete_null_field
from chat2plot.prompt import JSON_TAG, error_correction_prompt, system_prompt
from chat2plot.render import draw_altair, draw_plotly
from chat2plot.schema import PlotConfig, ResponseType

_logger = getLogger(__name__)

# These errors are caught within the application.
# Other errors (e.g. openai.error.RateLimitError) are propagated to user code.
_APPLICATION_ERRORS = (
    pydantic.ValidationError,
    jsonschema.ValidationError,
    ValueError,
    KeyError,
    AssertionError,
)


@dataclass(frozen=True)
class Plot:
    figure: alt.Chart | Figure | None
    config: PlotConfig | dict[str, Any] | None
    response_type: ResponseType
    explanation: str
    conversation_history: list[BaseMessage] | None


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
        return copy.deepcopy(self._conversation_history)

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
        self._session = ChatSession(df, system_prompt("simple"), "<{text}>", chat)
        self._df = df
        self._verbose = verbose

    @property
    def session(self) -> ChatSession:
        return self._session

    def query(self, q: str, config_only: bool = False, show_plot: bool = False) -> Plot:
        raw_response = self._session.query(q)

        try:
            if self._verbose:
                _logger.info(f"request: {q}")
                _logger.info(f"first response: {raw_response}")
            return self._parse_response(raw_response, config_only, show_plot)
        except _APPLICATION_ERRORS as e:
            if self._verbose:
                _logger.warning(traceback.format_exc())
            msg = e.message if isinstance(e, jsonschema.ValidationError) else str(e)
            error_correction = error_correction_prompt().format(
                error_message=msg,
            )
            corrected_response = self._session.query(error_correction)
            if self._verbose:
                _logger.info(f"retry response: {corrected_response}")

            try:
                return self._parse_response(corrected_response, config_only, show_plot)
            except _APPLICATION_ERRORS as e:
                if self._verbose:
                    _logger.warning(e)
                    _logger.warning(traceback.format_exc())
                return Plot(
                    None,
                    None,
                    ResponseType.FAILED_TO_RENDER,
                    "",
                    self._session.history,
                )

    def __call__(
        self, q: str, config_only: bool = False, show_plot: bool = False
    ) -> Plot:
        return self.query(q, config_only, show_plot)

    def _parse_response(self, content: str, config_only: bool, show_plot: bool) -> Plot:
        explanation, json_data = parse_json(content)

        try:
            config = PlotConfig.from_json(json_data)
        except _APPLICATION_ERRORS:
            _logger.warning(traceback.format_exc())
            # To reduce the number of failure cases as much as possible,
            # only check against the json schema when instantiation fails.
            jsonschema.validate(json_data, PlotConfig.schema())
            raise

        if self._verbose:
            _logger.info(config)

        if config_only:
            return Plot(
                None, config, ResponseType.SUCCESS, explanation, self._session.history
            )

        figure = draw_plotly(self._df, config, show_plot)
        return Plot(
            figure, config, ResponseType.SUCCESS, explanation, self._session.history
        )


class Chat2Vega(Chat2PlotBase):
    def __init__(
        self, df: pd.DataFrame, chat: BaseChatModel | None = None, verbose: bool = False
    ):
        self._session = ChatSession(df, system_prompt("vega"), "<{text}>", chat)
        self._df = df
        self._verbose = verbose

    @property
    def session(self) -> ChatSession:
        return self._session

    def query(self, q: str, config_only: bool = False, show_plot: bool = False) -> Plot:
        res = self._session.query(q)

        try:
            explanation, config = parse_json(res)
            if "data" in config:
                del config["data"]
            if self._verbose:
                _logger.info(config)
        except _APPLICATION_ERRORS:
            _logger.warning(f"failed to parse LLM response: {res}")
            _logger.warning(traceback.format_exc())
            return Plot(None, None, ResponseType.UNKNOWN, res, self._session.history)

        if config_only:
            return Plot(
                None, config, ResponseType.SUCCESS, explanation, self._session.history
            )

        try:
            plot = draw_altair(self._df, config, show_plot)
            return Plot(
                plot, config, ResponseType.SUCCESS, explanation, self._session.history
            )
        except _APPLICATION_ERRORS:
            _logger.warning(traceback.format_exc())
            return Plot(
                None,
                config,
                ResponseType.FAILED_TO_RENDER,
                explanation,
                self._session.history,
            )

    def __call__(
        self, q: str, config_only: bool = False, show_plot: bool = False
    ) -> Plot:
        return self.query(q, config_only, show_plot)


def chat2plot(
    df: pd.DataFrame,
    model_type: str = "simple",
    chat: BaseChatModel | None = None,
    verbose: bool = False,
) -> Chat2PlotBase:
    """Create Chat2Plot instance.

    Args:
        df: Data source for visualization.
        model_type: Type of json format. "vega" for a vega-lite compliant format, or "simple" or a simpler format.
        chat: The chat instance for interaction with LLMs.
              If omitted, `ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")` will be used.
        verbose: If `True`, chat2plot will output logs.

    Returns:
        Chat instance.
    """

    if model_type == "simple":
        return Chat2Plot(df, chat, verbose)
    elif model_type == "vega":
        return Chat2Vega(df, chat, verbose)
    else:
        raise ValueError(
            f"model_type should be one of [default, vega] (given: {model_type})"
        )


def _extract_tag_content(s: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*)</{tag}>", s, re.MULTILINE | re.DOTALL)
    if m:
        return m.group(1)
    else:
        m = re.search(rf"<{tag}>(.*)<{tag}>", s, re.MULTILINE | re.DOTALL)
        if m:
            return m.group(1)
    return ""


def parse_json(content: str) -> tuple[str, dict[str, Any]]:
    """parse json and split contents by pre-defined tags"""
    json_part = _extract_tag_content(content, "json")  # type: ignore
    if not json_part:
        raise ValueError(f"failed to find {JSON_TAG[0]} and {JSON_TAG[1]} tags")

    explanation_part = _extract_tag_content(content, "explain")
    if not explanation_part:
        explanation_part = _extract_tag_content(content, "explanation")

    # LLM rarely generates JSON with comments, so use the commentjson package instead of json
    return explanation_part.strip(), delete_null_field(commentjson.loads(json_part))
