import pandas as pd

from chat2plot import schema, chat2plot, PlotConfig


def test_filter():
    f = schema.Filter.parse_from_llm("a == 3")
    assert f.lhs == "a"
    assert f.op == "=="
    assert f.rhs == "3"
    assert f.escaped() == "`a` == 3"

    f = schema.Filter.parse_from_llm("a b c < 4")
    assert f.lhs == "a b c"
    assert f.op == "<"
    assert f.rhs == "4"
    assert f.escaped() == "`a b c` < 4"


def test_plot_bar():
    df = pd.DataFrame({
        "category": ["A", "B", "C", "A", "B"],
        "price": [100, 200, 100, 150, 250],
        "x": [1, 2, 3, 4, 5]
    })

    plot = chat2plot(df)
    ret = plot.query("Average price per category", config_only=True)
    config = ret.config
    assert isinstance(config, PlotConfig)
    assert config.chart_type == schema.ChartType.BAR
    assert config.x.column == "category"
    assert config.y.column == "price"
    assert config.y.aggregation == schema.AggregationType.AVG
