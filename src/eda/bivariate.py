"""Bivariate analysis tools: target vs categorical / numerical features."""
from __future__ import annotations

import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .distributions import numeric_columns


def _grid_axes(n: int, ncols: int, figsize_per: tuple[float, float]):
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        squeeze=False,
    )
    flat = axes.flatten()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def categorical_columns(
    df: pd.DataFrame,
    exclude: Sequence[str] | None = None,
    max_unique: int = 20,
) -> list[str]:
    exclude = set(exclude or [])
    out: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            out.append(c)
        elif pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) <= max_unique:
            out.append(c)
    return out


def target_rate_table(
    df: pd.DataFrame,
    col: str,
    target: str,
    dropna: bool = False,
) -> pd.DataFrame:
    """Frequency and mean target rate per category (works for binary target)."""
    g = df.groupby(col, dropna=dropna)[target]
    out = pd.DataFrame({
        "count": g.size(),
        "target_sum": g.sum(),
        "target_rate": g.mean(),
    })
    out["pct"] = out["count"] / out["count"].sum()
    return out.sort_values("target_rate", ascending=False)


def plot_categorical_vs_target(
    df: pd.DataFrame,
    cols: Sequence[str] | None = None,
    target: str = "Survived",
    ncols: int = 2,
    figsize_per: tuple[float, float] = (5.5, 3.5),
    kind: str = "both",  # "count", "rate", "both"
    order_by: str = "auto",  # "auto", "value", "count", "rate"
):
    """For each categorical feature plot counts stacked by target and/or target mean rate.

    ``order_by`` controls the x-axis ordering of categories:
      - "auto": numeric-dtype columns are sorted by value (ascending),
        everything else by descending count.
      - "value": sort ascending by the category value itself.
      - "count": descending frequency.
      - "rate": descending mean target rate.
    """
    if cols is None:
        cols = categorical_columns(df, exclude=[target])
    cols = [c for c in cols if c != target]

    plots_per_feat = 2 if kind == "both" else 1
    total = len(cols) * plots_per_feat
    fig, axes = _grid_axes(total, ncols, figsize_per)

    NA_LABEL = "<NA>"
    i = 0
    for c in cols:
        is_numeric = pd.api.types.is_numeric_dtype(df[c])
        plot_df = df[[c, target]].copy()
        # sort by the original (pre-stringified) value when needed
        value_sorted = sorted(
            plot_df[c].dropna().unique().tolist()
        )
        plot_df[c] = plot_df[c].astype(object).where(plot_df[c].notna(), NA_LABEL).astype(str)

        strategy = order_by
        if strategy == "auto":
            strategy = "value" if is_numeric else "count"

        if strategy == "value":
            order = [str(v) for v in value_sorted]
            if (plot_df[c] == NA_LABEL).any():
                order.append(NA_LABEL)
        elif strategy == "rate":
            order = (
                plot_df.groupby(c)[target].mean().sort_values(ascending=False).index.tolist()
            )
        else:  # "count"
            order = plot_df[c].value_counts().index.tolist()
        if kind in ("count", "both"):
            sns.countplot(
                data=plot_df, x=c, hue=target, order=order, ax=axes[i],
            )
            axes[i].set_title(f"Count — {c} by {target}")
            axes[i].tick_params(axis="x", rotation=30)
            i += 1
        if kind in ("rate", "both"):
            rates = plot_df.groupby(c)[target].mean().reindex(order)
            sns.barplot(x=list(rates.index), y=rates.values, ax=axes[i], color="steelblue")
            axes[i].set_title(f"{target} rate — {c}")
            axes[i].set_ylabel(f"mean({target})")
            axes[i].set_xlabel(c)
            axes[i].tick_params(axis="x", rotation=30)
            i += 1

    fig.tight_layout()


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    figsize: tuple[float, float] = (6.5, 4.5),
    alpha: float = 0.7,
    s: float = 30,
    ax: plt.Axes | None = None,
    **kwargs,
):
    """Scatter plot of ``x`` vs ``y``, optionally coloured by ``hue``.

    Example: ``plot_scatter(df, "Age", "Fare", hue="Survived")``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=df, x=x, y=y, hue=hue, alpha=alpha, s=s, ax=ax, **kwargs,
    )
    title = f"{y} vs {x}" + (f" by {hue}" if hue else "")
    ax.set_title(title)
    return ax


def plot_numerical_vs_target(
    df: pd.DataFrame,
    cols: Sequence[str] | None = None,
    target: str = "Survived",
    bins: int = 30,
    ncols: int = 2,
    figsize_per: tuple[float, float] = (5.5, 3.5),
    kind: str = "both",  # "kde", "box", "both"
):
    """For each numerical feature plot KDE and/or boxplot split by target classes."""
    if cols is None:
        cols = numeric_columns(df, exclude=[target])
    cols = [c for c in cols if c != target]

    plots_per_feat = 2 if kind == "both" else 1
    total = len(cols) * plots_per_feat
    fig, axes = _grid_axes(total, ncols, figsize_per)

    i = 0
    for c in cols:
        if kind in ("kde", "both"):
            sns.kdeplot(
                data=df, x=c, hue=target, common_norm=False, fill=True, ax=axes[i],
            )
            axes[i].set_title(f"KDE — {c} by {target}")
            i += 1
        if kind in ("box", "both"):
            sns.boxplot(data=df, x=target, y=c, ax=axes[i])
            axes[i].set_title(f"Boxplot — {c} by {target}")
            i += 1

    fig.tight_layout()
