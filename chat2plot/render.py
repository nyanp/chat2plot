import copy
from typing import Any

import altair as alt
import pandas as pd
import plotly.express as px
import vegafusion as vf
from altair.utils.data import to_values
from plotly.graph_objs import Figure

from chat2plot.schema import (
    AggregationType,
    ChartType,
    Filter,
    PlotConfig,
    SortingCriteria,
    SortOrder,
)
from chat2plot.transform import transform


def _ax_config(config: PlotConfig, x: str, y: str) -> dict[str, str | dict[str, str]]:
    ax: dict[str, str | dict[str, str]] = {"x": x, "y": y}
    labels: dict[str, str] = {}

    if config.xlabel:
        labels[x] = config.xlabel
    if config.ylabel:
        labels[y] = config.ylabel

    if labels:
        ax["labels"] = labels

    return ax


def draw_plotly(df: pd.DataFrame, config: PlotConfig, show: bool = True) -> Figure:
    df_filtered = filter_data(df, config.filters).copy()
    df_filtered = transform(df_filtered, config)

    chart_type = config.chart_type

    if chart_type in [ChartType.BAR, ChartType.HORIZONTAL_BAR]:
        agg = groupby_agg(df_filtered, config)
        x = agg.columns[0]
        y = agg.columns[-1]
        orientation = "v"

        if chart_type == ChartType.HORIZONTAL_BAR:
            x, y = y, x
            orientation = "h"

        fig = px.bar(
            agg,
            color=config.hue.column if config.hue else None,
            orientation=orientation,
            **_ax_config(config, x, y),
        )
    elif chart_type == ChartType.SCATTER:
        assert config.x is not None
        fig = px.scatter(
            df_filtered,
            color=config.hue.column if config.hue else None,
            **_ax_config(config, config.x.column, config.y.column),
        )
    elif chart_type == ChartType.PIE:
        agg = groupby_agg(df_filtered, config)
        fig = px.pie(agg, names=agg.columns[0], values=agg.columns[-1])
    elif chart_type in [ChartType.LINE, ChartType.AREA]:
        func_table = {ChartType.LINE: px.line, ChartType.AREA: px.area}

        if is_aggregation(config):
            agg = groupby_agg(df_filtered, config)
            fig = func_table[chart_type](
                agg, **_ax_config(config, agg.columns[0], y=agg.columns[-1])
            )
        else:
            assert config.x is not None
            fig = func_table[chart_type](
                df_filtered,
                color=config.hue.column if config.hue else None,
                **_ax_config(config, config.x.column, config.y.column),
            )
    else:
        raise ValueError(f"Unknown chart_type: {chart_type}")

    if show:
        fig.show()

    return fig


def draw_altair(
    df: pd.DataFrame,
    config: dict[str, Any],
    show: bool = True,
    use_vega_fusion: bool = True,
) -> alt.Chart:
    if use_vega_fusion:
        vf.enable()
    spec = copy.deepcopy(config)
    spec["data"] = to_values(df)
    chart = alt.Chart.from_dict(spec)
    if show:
        chart.show()

    return chart


def groupby_agg(df: pd.DataFrame, config: PlotConfig) -> pd.DataFrame:
    group_by = [config.x.column] if config.x is not None else []

    if config.hue and config.hue != config.x:
        group_by.append(config.hue.column)

    agg_method = {
        AggregationType.AVG: "mean",
        AggregationType.SUM: "sum",
        AggregationType.COUNT: "count",
        AggregationType.DISTINCT_COUNT: "nunique",
        AggregationType.MIN: "min",
        AggregationType.MAX: "max",
    }

    m = config.y
    assert m.aggregation is not None

    if not group_by:
        return pd.DataFrame({str(m): [df[m.column].agg(agg_method[m.aggregation])]})
    else:
        agg = (
            df.groupby(group_by, dropna=False)[m.column]
            .agg(agg_method[m.aggregation])
            .rename(str(m))
        )
        ascending = config.sort_order == SortOrder.ASC

        if config.sort_criteria == SortingCriteria.NAME:
            agg = agg.sort_index(ascending=ascending)
        elif config.sort_criteria == SortingCriteria.VALUE:
            agg = agg.sort_values(ascending=ascending)

        return agg.reset_index()


def is_aggregation(config: PlotConfig) -> bool:
    return config.y.aggregation is not None


def filter_data(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    if not filters:
        return df
    return df.query(" and ".join([f.query for f in filters]))
