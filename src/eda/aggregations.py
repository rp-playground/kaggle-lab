import pandas as pd


def grouped_median(df: pd.DataFrame, target: str, by: list[str]) -> pd.Series:
    return df.groupby(by)[target].median()
