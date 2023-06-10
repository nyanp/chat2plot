import pandas as pd
import pytest

from chat2plot import schema, chat2plot, PlotConfig


@pytest.mark.parametrize(
    "prompt",
    [
        "Average price per category",
        "カテゴリごとの平均価格",
        "avg price for each category",
        "Show me average price per category in bar chart."
    ]
)
def test_plot_bar(prompt):
    df = pd.DataFrame({
        "category": ["A", "B", "C", "A", "B"],
        "price": [100, 200, 100, 150, 250],
        "x": [1, 2, 3, 4, 5]
    })

    plot = chat2plot(df)
    ret = plot.query(prompt, config_only=True)
    config = ret.config
    assert isinstance(config, PlotConfig)
    assert config.chart_type == schema.ChartType.BAR
    assert config.x.column == "category"
    assert config.y.column == "price"
    assert config.y.aggregation == schema.AggregationType.AVG

