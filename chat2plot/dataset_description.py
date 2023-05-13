import pandas as pd


def description(df: pd.DataFrame) -> str:
    return str(df.sample(5, random_state=0).to_markdown())
