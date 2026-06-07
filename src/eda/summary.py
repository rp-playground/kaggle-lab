"""Wrapper around df.info() / df.describe() with selectable options."""
from __future__ import annotations

from typing import Literal

import pandas as pd

try:
    from IPython.display import display, Markdown
    _IN_NOTEBOOK = True
except ImportError:  # pragma: no cover
    _IN_NOTEBOOK = False


def _show(obj, header: str | None = None) -> None:
    if header:
        if _IN_NOTEBOOK:
            display(Markdown(f"**{header}**"))
        else:
            print(f"\n=== {header} ===")
    if _IN_NOTEBOOK:
        display(obj)
    else:
        print(obj)


def describe_df(
    df: pd.DataFrame,
    info: bool = True,
    numeric: bool = True,
    categorical: bool = True,
    title: str | None = None,
    percentiles: list[float] | None = None,
    categorical_include: Literal["object", "str", "all_non_numeric"] = "object",
    return_results: bool = False,
) -> dict[str, pd.DataFrame | None] | None:
    """Display and optionally return selected info/describe outputs.

    Parameters
    ----------
    df : DataFrame to analyze.
    info : if True, call df.info().
    numeric : if True, compute df.describe() on numerical columns.
    categorical : if True, compute df.describe() on non-numerical columns.
    title : optional header label.
    percentiles : percentiles forwarded to describe (e.g. [.05, .5, .95]).
    categorical_include : how to select categorical columns
        ("object", "str" or "all_non_numeric").
    return_results : if True return a dict with the computed DataFrames;
        if False (default) return None to avoid auto-display of the dict
        in Jupyter.

    Returns
    -------
    dict with keys "numeric" and "categorical" (DataFrame or None) when
    ``return_results=True``, otherwise None.
    """
    if title:
        if _IN_NOTEBOOK:
            display(Markdown(f"## {title}"))
        else:
            print(f"===== {title} =====")

    if info:
        if _IN_NOTEBOOK:
            display(Markdown("**Info**"))
        else:
            print("\n=== Info ===")
        df.info()

    out: dict[str, pd.DataFrame | None] = {"numeric": None, "categorical": None}

    if numeric:
        num = df.describe(percentiles=percentiles)
        out["numeric"] = num
        _show(num, header="Numerical summary")

    if categorical:
        if categorical_include == "all_non_numeric":
            cat_df = df.select_dtypes(exclude="number")
        else:
            cat_df = df.select_dtypes(include=categorical_include)
        if cat_df.shape[1] > 0:
            cat = cat_df.describe()
            out["categorical"] = cat
            _show(cat, header="Categorical summary")

    return out if return_results else None


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    num_rows = df.shape[0]
    missing = df.isnull().sum()
    out_df = pd.DataFrame(index=df.columns)
    out_df["Missing Count"] = missing
    out_df["Missing %"] = (100.0 * missing / num_rows).round(2)
    return out_df


def count_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> int:
    return int(df.duplicated(subset=subset).sum())
