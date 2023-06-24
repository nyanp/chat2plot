import pandas as pd
import pydantic
import pytest
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema import FunctionMessage

from chat2plot import PlotConfig, chat2plot, schema


@pytest.mark.parametrize(
    "prompt",
    [
        "Average price per category",
        "カテゴリごとの平均価格",
        "avg price for each category",
        "Show me average price per category in bar chart.",
    ],
)
def test_plot_bar(prompt: str):
    df = pd.DataFrame(
        {
            "category": ["A", "B", "C", "A", "B"],
            "price": [100, 200, 100, 150, 250],
            "x": [1, 2, 3, 4, 5],
        }
    )

    plot = chat2plot(df)
    ret = plot.query(prompt, config_only=True)
    config = ret.config
    assert isinstance(config, PlotConfig)
    assert config.chart_type == schema.ChartType.BAR
    assert config.x.column == "category"
    assert config.y.column == "price"
    assert config.y.aggregation == schema.AggregationType.AVG


def test_vega_json():
    df = pd.DataFrame(
        {
            "date": [
                "2021-01-01",
                "2021-02-02",
                "2021-02-03",
                "2021-02-04",
                "2021-02-05",
            ],
            "price": [100, 200, 300, 400, 500],
            "x": [1, 2, 3, 4, 5],
        }
    )
    plot = chat2plot(df, schema_definition="vega")
    ret = plot.query("Daily total sales in line chart", config_only=True)

    assert isinstance(ret.config, dict)

    # https://vega.github.io/vega-lite/docs/line.html#line-chart
    expected = {
        "mark": "line",
        "encoding": {
            "x": {"field": "date", "type": "temporal"},
            "y": {"field": "price", "aggregate": "sum", "type": "quantitative"},
        },
    }
    assert ret.config["mark"] == expected["mark"]
    assert ret.config["encoding"]["x"] == expected["encoding"]["x"]
    assert ret.config["encoding"]["y"] == expected["encoding"]["y"]


class CustomChartConfig(pydantic.BaseModel):
    chart_type: str
    x_axis_name: str
    y_axis_name: str
    y_axis_aggregate: str


def test_custom_schema():
    df = pd.DataFrame(
        {
            "date": [
                "2021-01-01",
                "2021-02-02",
                "2021-02-03",
                "2021-02-04",
                "2021-02-05",
            ],
            "price": [100, 200, 300, 400, 500],
            "x": [1, 2, 3, 4, 5],
        }
    )
    plot = chat2plot(df, schema_definition=CustomChartConfig)
    ret = plot.query("Daily total sales in line chart", config_only=True)

    assert isinstance(ret.config, CustomChartConfig)
    assert ret.config.chart_type == "line"
    assert ret.config.x_axis_name == "date"
    assert ret.config.y_axis_name == "price"
    assert ret.config.y_axis_aggregate.lower() == "sum"


def test_function_call():
    df = pd.DataFrame(
        {
            "date": [
                "2021-01-01",
                "2021-02-02",
                "2021-02-03",
                "2021-02-04",
                "2021-02-05",
            ],
            "price": [100, 200, 300, 400, 500],
            "x": [1, 2, 3, 4, 5],
        }
    )

    for function_call in [False, True, "auto"]:
        plot = chat2plot(df, function_call=function_call)
        if function_call == "auto":
            assert plot.function_call
        else:
            assert plot.function_call == function_call
        ret = plot.query("Daily total sales in line chart", config_only=True)
        assert ret.config.chart_type == schema.ChartType.LINE
        assert ret.config.x.column == "date"
        assert ret.config.y.column == "price"
        assert ret.config.y.aggregation == schema.AggregationType.SUM

        if plot.function_call:
            assert any(
                isinstance(msg, FunctionMessage) for msg in ret.conversation_history
            )
        else:
            assert not any(
                isinstance(msg, FunctionMessage) for msg in ret.conversation_history
            )


def test_function_call_auto():
    chat = ChatOpenAI(model_name="gpt-3.5-turbo")
    plot = chat2plot(pd.DataFrame(), chat=chat)
    assert not plot.function_call

    chat = ChatOpenAI(model_name="gpt-4")
    plot = chat2plot(pd.DataFrame(), chat=chat)
    assert not plot.function_call

    chat = ChatOpenAI(model_name="gpt-3.5-turbo-0613")
    plot = chat2plot(pd.DataFrame(), chat=chat)
    assert plot.function_call

    chat = AzureChatOpenAI(openai_api_base="azure", openai_api_version="dummy")
    plot = chat2plot(pd.DataFrame(), chat=chat)
    assert not plot.function_call
