import json
import re
import traceback
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import altair as alt
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from plotly.graph_objs import Figure

from chat2plot.dataset_description import description
from chat2plot.render import draw_altair, draw_plotly
from chat2plot.schema import LLMResponse, PlotConfig, ResponseType

_logger = getLogger(__name__)

_PROMPT = """
Your task is to generate chart configuration for the given dataset and user question delimited by <>.

Responses should be in JSON format compliant to the following JSON Schema.

{{
  "title": "Chart config",
  "description": "A configuration of chart",
  "type": "object",
  "properties": {{
    "chart_type": {{
      "description": "the type of the chart",
      "type": "string",
      "enum": [
        "line",
        "scatter",
        "bar",
        "pie",
        "horizontal-bar",
        "area"
      ]
    }},
    "x": {{
      "description": "X-axis column for the chart, or label column for pie chart.",
      "type": "object",
      "properties": {{
        "column": {{
          "description": "Column name of the dataset used as x-axis",
          "type": "string"
        }}
      }},
      "required": ["column"]
    }},
    "y": {{
      "description": "Y-axis column for the chart, or value column for pie chart.",
      "type": "object",
      "properties": {{
        "column": {{
          "description": "Column name of the dataset used as y-axis",
          "type": "string"
        }},
        "aggregation": {{
          "description": "Type of aggregation. will be ignored when it is scatter plot.
          "type": "string",
          "enum": [
            "SUM",
            "AVG",
            "COUNT",
            "MAX",
            "MIN",
            "DISTINCT_COUNT"
          ]
        }}
      }},
      "required": ["column"]
    }},
    "filters": {{
      "description": "List of filter conditions, where each filter must be in a valid format as an argument to the pandas df.query() method.",
      "type": "array",
      "items": {{
        "type": "string"
      }}
    }},
    "hue": {{
      "description": "Dimension used as grouping variables that will produce different colors.",
      "type": "object",
      "properties": {{
        "column": {{
          "type": "string"
        }}
      }},
      "required": ["column"]
    }},
    "xmin": {{
      "description": "Minimum value of x-axis",
      "type": "number"
    }},
    "xmax": {{
      "description": "Maximum value of x-axis",
      "type": "number"
    }},
    "ymin": {{
      "description": "Minimum value of y-axis",
      "type": "number"
    }},
    "ymax": {{
      "description": "Maximum value of y-axis",
      "type": "number"
    }},
    "xlabel": {{
      "description": "Label of x-axis",
      "type": "string"
    }},
    "ylabel": {{
      "description": "Label of y-axis",
      "type": "string"
    }},
    "sort_criteria": {{
      "description": "The sorting criteria for x-axis",
      "type": "string",
      "enum": [
        "name",
        "value"
      ]
    }},
    "sort_order": {{
      "description": "Sorting order for x-axis",
      "type": "string",
      "enum": [
        "asc",
        "desc"
      ]
    }},
  }},
  "required": [
    "chart_type",
    "y"
  ]
}}

Instead of specifying column names in the dataset directly for `x.column` and `y.column`, you can use one of the following conversion functions if necessary:

BINNING(column, interval): binning a numerical column to the specified interval. interval should be integer literal. example: BINNING(x, 10)
ROUND_DATETIME(column, period): binning a date/datetime column to the specified period. period should be one of [day, week, month, year]. example: ROUND_DATETIME(x, year)

The user's question may be an instruction to fine-tune the previous chart, or it may be an instruction to create a new chart based on a completely new context. In the latter case, be careful not to use the context used for the previous chart.

If the user's question does not fall under any of above keys and is not a request about the appearance of the chart, simply reply "not related".

This is the result of `print(df.head())`:

{dataset}

Make sure to prefix the requested json string with triple backticks exactly and suffix the json with triple backticks exactly.
"""


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

    def query(self, q: str) -> str:
        prompt = self._user_prompt_template.format(text=q)
        response = self._query(prompt)
        return response.content

    def _query(self, prompt: str) -> BaseMessage:
        self._conversation_history.append(HumanMessage(content=prompt))
        response = self._chat(self._conversation_history)
        self._conversation_history.append(response)
        return response


class Chat2PlotBase:
    @property
    def session(self) -> ChatSession:
        raise NotImplementedError()

    def query(self, q: str, show_plot: bool = True) -> Plot:
        raise NotImplementedError()

    def __call__(self, q: str, show_plot: bool = True) -> Plot:
        return self.query(q, show_plot)


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

    def query(self, q: str, show_plot: bool = True) -> Plot:
        raw_response = self._session.query(q)
        res = self._parse_response(raw_response)
        if res.response_type == ResponseType.SUCCESS:
            assert res.config is not None
            try:
                return Plot(
                    self.render(self._df, res.config, show_plot),
                    res.config,
                    res.response_type,
                    raw_response,
                )
            except Exception:
                _logger.warning(traceback.format_exc())
                return Plot(
                    None, res.config, ResponseType.FAILED_TO_RENDER, raw_response
                )
        return Plot(None, None, res.response_type, raw_response)

    def __call__(self, q: str, show_plot: bool = True) -> Plot:
        return self.query(q, show_plot)

    def render(
        self, df: pd.DataFrame, config: PlotConfig, show_plot: bool = True
    ) -> Any:
        return draw_plotly(df, config, show_plot)

    def _parse_response(self, content: str) -> LLMResponse:
        if content == "not related":
            return LLMResponse(ResponseType.NOT_RELATED)

        try:
            config = PlotConfig.from_json(parse_json(content))
            if self._verbose:
                _logger.info(config)
            return LLMResponse(ResponseType.SUCCESS, config)
        except Exception:
            _logger.warning(f"failed to parse LLM response: {content}")
            _logger.warning(traceback.format_exc())
            return LLMResponse(ResponseType.UNKNOWN)


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

    def query(self, q: str, show_plot: bool = True) -> Plot:
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

        try:
            plot = draw_altair(self._df, config, show_plot)
            return Plot(plot, config, ResponseType.SUCCESS, res)
        except Exception:
            _logger.warning(traceback.format_exc())
            return Plot(None, config, ResponseType.FAILED_TO_RENDER, res)

    def __call__(self, q: str, show_plot: bool = True) -> Plot:
        return self.query(q, show_plot)


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
        raise
