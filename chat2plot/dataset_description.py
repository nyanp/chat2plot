import pandas as pd


def description(df: pd.DataFrame) -> str:
    return str(df.head().to_markdown())
