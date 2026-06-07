"""Distribution analysis tools for numerical / continuous features."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def numeric_columns(
    df: pd.DataFrame,
    exclude: Iterable[str] | None = None,
    min_unique: int = 1,
) -> list[str]:
    exclude = set(exclude or [])
    cols = df.select_dtypes(include=np.number).columns
    return [c for c in cols if c not in exclude and df[c].nunique(dropna=True) >= min_unique]


def _resolve_cols(df: pd.DataFrame, cols: Sequence[str] | None) -> list[str]:
    if cols is None:
        return numeric_columns(df)
    return list(cols)


def _grid(n: int, ncols: int) -> tuple[int, int]:
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def _make_axes(n: int, ncols: int, figsize_per: tuple[float, float]):
    nrows, ncols = _grid(n, ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        squeeze=False,
    )
    flat = axes.flatten()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def plot_histograms(
    df: pd.DataFrame,
    cols: Sequence[str] | None = None,
    bins: int = 30,
    ncols: int = 3,
    kde: bool = False,
    figsize_per: tuple[float, float] = (4.5, 3.2),
):
    cols = _resolve_cols(df, cols)
    fig, axes = _make_axes(len(cols), ncols, figsize_per)
    for ax, c in zip(axes, cols):
        sns.histplot(df[c].dropna(), bins=bins, kde=kde, ax=ax, color="steelblue")
        ax.set_title(f"Histogram — {c}")
        ax.set_xlabel(c)
    fig.tight_layout()


def plot_kde(
    df: pd.DataFrame,
    cols: Sequence[str] | None = None,
    ncols: int = 3,
    fill: bool = True,
    figsize_per: tuple[float, float] = (4.5, 3.2),
):
    cols = _resolve_cols(df, cols)
    fig, axes = _make_axes(len(cols), ncols, figsize_per)
    for ax, c in zip(axes, cols):
        sns.kdeplot(df[c].dropna(), ax=ax, fill=fill, color="darkorange")
        ax.set_title(f"KDE — {c}")
        ax.set_xlabel(c)
    fig.tight_layout()


def plot_boxplots(
    df: pd.DataFrame,
    cols: Sequence[str] | None = None,
    ncols: int = 3,
    by: str | None = None,
    figsize_per: tuple[float, float] = (4.5, 3.2),
):
    cols = _resolve_cols(df, cols)
    if by is not None:
        cols = [c for c in cols if c != by]
    fig, axes = _make_axes(len(cols), ncols, figsize_per)
    for ax, c in zip(axes, cols):
        if by is None:
            sns.boxplot(x=df[c].dropna(), ax=ax, color="seagreen")
            ax.set_title(f"Boxplot — {c}")
        else:
            sns.boxplot(data=df, x=by, y=c, ax=ax)
            ax.set_title(f"Boxplot — {c} by {by}")
    fig.tight_layout()


def plot_distributions(
    df: pd.DataFrame,
    cols: Sequence[str] | None = None,
    bins: int = 30,
    by: str | None = None,
    figsize_per: tuple[float, float] = (4.2, 3.0),
):
    """Panel with histogram+KDE, pure KDE and boxplot for each feature (one row per feature)."""
    cols = _resolve_cols(df, cols)
    if by is not None:
        cols = [c for c in cols if c != by]
    n = len(cols)
    fig, axes = plt.subplots(
        n, 3,
        figsize=(figsize_per[0] * 3, figsize_per[1] * n),
        squeeze=False,
    )
    for i, c in enumerate(cols):
        s = df[c].dropna()
        sns.histplot(s, bins=bins, kde=True, ax=axes[i, 0], color="steelblue")
        axes[i, 0].set_title(f"Histogram+KDE — {c}")
        sns.kdeplot(s, ax=axes[i, 1], fill=True, color="darkorange")
        axes[i, 1].set_title(f"KDE — {c}")
        if by is None:
            sns.boxplot(x=s, ax=axes[i, 2], color="seagreen")
            axes[i, 2].set_title(f"Boxplot — {c}")
        else:
            sns.boxplot(data=df, x=by, y=c, ax=axes[i, 2])
            axes[i, 2].set_title(f"Boxplot — {c} by {by}")
    fig.tight_layout()
