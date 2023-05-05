import re

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from chat2plot.schema import PlotConfig


def transform(df: pd.DataFrame, config: PlotConfig) -> pd.DataFrame:
    for col in config.required_columns:
        if col not in df:
            df[col] = transform_one(df, col, config)

    return df


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


def round_datetime(series: pd.Series, period: str) -> pd.Series:
    series = pd.to_datetime(series)

    period_map = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}
    return series.dt.to_period(period_map.get(period, period)).dt.to_timestamp()
