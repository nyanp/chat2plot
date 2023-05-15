import pandas as pd


def description(df: pd.DataFrame, num_rows: int = 5) -> str:
    """Returns a description of the given data for LLM"""
    return str(df.sample(num_rows, random_state=0).to_markdown())
