import json

import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, HumanMessage

from chat2plot.dataset_description import description
from chat2plot.render import draw_plotly
from chat2plot.schema import LLMResponse, PlotConfig, ResponseType

_PROMPT = """
Your task is to generate chart configuration for the given dataset and user question delimited by <>.

Responses should be in JSON format including the following keys:

chart_type: the type of chart, should be one of [line, scatter, bar, pie, horizontal-bar, area]
measures: list of measure, which each measure shoule be expressed as the combination of aggregations (should be one of [SUM, AVG, COUNT, MAX, MIN, DISTINCT_COUNT]) and column (should be numeric), like "SUM(price)". If the chart is a scatter plot, aggregation should be omitted and simply answer the column name. The length of the list is 2 only for scatter plot and 1 otherwise.
dimension: group-by column, which should be categorical/datetime variables used as axis.
filters: list of filter conditions, where each filter must be in a valid format as an argument to the pandas df.query() method.
hue: (optional) dimension used as grouping variables that will produce different colors.
xmin: (optional) minimum value of x-axis.
xmax: (optional) maximum value of x-axis.
ymin: (optional) minimum value of y-axis.
ymax: (optional) maximum value of y-axis.
xlabel: (optional) label of x-axis.
ylabel: (optional) label of y-axis.

The user's question may be an instruction to fine-tune the previous chart, or it may be an instruction to create a new chart based on a completely new context. In the latter case, be careful not to use the context used for the previous chart.

If the user's question is about the appearance of the chart and does not fall under any of above keys, simply reply "change style" instead of JSON.
If the user's question does not fall under any of above keys and is not a request about the appearance of the chart, simply reply "not related".

Dataset contains the following contents:

{dataset}

User Question: <{text}>
"""


class Chat2PlotConfig:
    def __init__(
        self,
        df: pd.DataFrame,
        chat: BaseChatModel | None = None,
        prompt: str | None = None,
    ):
        self._df = df
        self._conversation_history: list[BaseMessage] = []
        self._chat = chat or ChatOpenAI(temperature=0, model_name="gpt-4")  # type: ignore
        self._first_prompt = prompt or _PROMPT
        self._second_prompt = "User Question: <{text}>"

    def query(self, q: str) -> LLMResponse:
        if not self._conversation_history:
            # first question
            prompt = self._first_prompt.format(text=q, dataset=description(self._df))
        else:
            prompt = self._second_prompt.format(text=q)

        res = self._query(prompt)

        return self._parse_response(res.content)

    def _parse_response(self, content: str) -> LLMResponse:
        if content == "not related":
            return LLMResponse(ResponseType.NOT_RELATED)

        try:
            config = PlotConfig.from_json(json.loads(content))
            return LLMResponse(ResponseType.CHART, config)
        except Exception:
            raise  # TODO

    def _query(self, prompt: str) -> BaseMessage:
        self._conversation_history.append(HumanMessage(content=prompt))
        response = self._chat(self._conversation_history)
        self._conversation_history.append(response)
        return response


class Chat2Plot:
    def __init__(self, df: pd.DataFrame, chat: BaseChatModel | None = None):
        self._config_generator = Chat2PlotConfig(df, chat)
        self.df = df
        self._config_history: list[PlotConfig] = []

    def query(self, q: str) -> ResponseType:
        res = self._config_generator.query(q)
        if res.response_type == ResponseType.CHART:
            assert res.config is not None
            self._config_history.append(res.config)
            self.render(self.df, res.config)
        return res.response_type

    def __call__(self, q: str) -> ResponseType:
        return self.query(q)

    def render(self, df: pd.DataFrame, config: PlotConfig) -> None:
        draw_plotly(df, config)
