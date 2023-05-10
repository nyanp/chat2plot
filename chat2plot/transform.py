import copy
import re


import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from chat2plot.schema import PlotConfig, Axis, TimeUnit


def transform(df: pd.DataFrame, config: PlotConfig) -> tuple[pd.DataFrame, PlotConfig]:
    config = copy.deepcopy(config)

    for col in config.required_columns:
        if col not in df:
            df[col] = transform_one(df, col, config)

    if config.x and config.x.transform:
        x_trans = _transform(df, config.x)
        df[x_trans.name] = x_trans
        config.x.column = x_trans.name

    if config.y.transform:
        y_trans = _transform(df, config.y)
        df[y_trans.name] = y_trans
        config.y.column = y_trans.name

    return df, config


def _transform(df: pd.DataFrame, ax: Axis) -> pd.Series:
    if not ax.transform:
        return df[ax.column]

    dst = df[ax.column].copy()

    if ax.transform.bin_size:
        dst = binning(dst, ax.transform.bin_size)

    if ax.transform.time_unit:
        dst = round_datetime(dst, ax.transform.time_unit)

    return pd.Series(dst.values, name=ax.transformed_name())


def transform_one(df: pd.DataFrame, col: str, config: PlotConfig) -> pd.Series:
    m = re.match(r"BINNING\((.*),(.*)\)", col)
    if m:
        col = m.group(1).strip()
        interval = int(m.group(2).strip())
        return binning(df[col], interval)

    m = re.match(r"ROUND_DATETIME\((.*),(.*)\)", col)
    if m:
        col = m.group(1).strip()
        period = m.group(2).strip()
        return round_datetime(df[col], period)

    raise ValueError(f"Unknown column transform: {col}")


def binning(series: pd.Series, interval: int) -> pd.Series:
    start_point = np.floor(series.min() / interval) * interval
    end_point = np.ceil((series.max() + 1) / interval) * interval
    bins = pd.interval_range(
        start=start_point,
        end=end_point,
        freq=interval,
        closed="both" if is_integer_dtype(series) else "left",
    )
    binned_series = pd.cut(series, bins=bins)
    return binned_series.astype(str)


def round_datetime(series: pd.Series, period: TimeUnit) -> pd.Series:
    series = pd.to_datetime(series)

    period_map = {TimeUnit.DAY: "D", TimeUnit.WEEK: "W", TimeUnit.MONTH: "M", TimeUnit.QUARTER: "Q", TimeUnit.YEAR: "Y"}
    return series.dt.to_period(period_map.get(period, period)).dt.to_timestamp()
