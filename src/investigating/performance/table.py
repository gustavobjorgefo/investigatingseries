"""
The metrics table of the performance report.

Two steps, kept apart on purpose:

- :func:`performance_metrics` computes: raw floats, one row per metric, one
  column per series (the factsheet orientation). Use it for inline numbers in
  the text, e.g. ``f"{metrics.loc['sharpe', 'strategy']:.2f}"``.
- :func:`format_metrics` presents: readable labels and units (``15.34%``,
  ``0.82``, ``412``), ready to render as a table in a Quarto post.

The same split exists in ibexQuant (``analytics.summary`` computes,
presentation formats), so this module can move there later.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final, Literal

import pandas as pd

from investigating.performance._analytics import (
    TRADING_DAYS,
    annualized_volatility,
    cagr,
    calmar_ratio,
    hit_rate,
    kurtosis,
    longest_drawdown_days,
    max_drawdown,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    time_in_market,
    total_return,
)

Unit = Literal["percent", "ratio", "days"]

# Metric key -> (display label, unit). The order here is the order of the table.
METRICS: Final[dict[str, tuple[str, Unit]]] = {
    "total_return": ("Cumulative return", "percent"),
    "cagr": ("CAGR", "percent"),
    "volatility": ("Annualized volatility", "percent"),
    "sharpe": ("Sharpe ratio", "ratio"),
    "sortino": ("Sortino ratio", "ratio"),
    "max_drawdown": ("Max drawdown", "percent"),
    "longest_drawdown_days": ("Longest drawdown (days)", "days"),
    "calmar": ("Calmar ratio", "ratio"),
    "time_in_market": ("Time in market", "percent"),
    "hit_rate": ("Hit rate", "percent"),
    "skew": ("Skew", "ratio"),
    "kurtosis": ("Excess kurtosis", "ratio"),
}

MISSING: Final[str] = "–"


def performance_metrics(
    returns: pd.Series | pd.DataFrame,
    *,
    risk_free: float | pd.Series = 0.0,
    periods_per_year: int = TRADING_DAYS,
    common_period: bool = True,
) -> pd.DataFrame:
    """
    Compute every metric of the report, as raw floats.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily simple returns; one column per series (strategy, benchmark, ...).
        A Series becomes a single column named after it (or ``"strategy"``).
    risk_free : float or pd.Series, default 0.0
        Annual rate (float) or periodic returns (Series), for Sharpe and Sortino.
    periods_per_year : int, default 252
        Annualization factor.
    common_period : bool, default True
        Measure every column over the dates they all share, so a strategy and
        a benchmark are compared over the same window. With False, each column
        uses its own history.

    Returns
    -------
    pd.DataFrame
        Rows are the keys of :data:`METRICS`; columns are the series. The
        period used is stored in ``.attrs["start"]`` and ``.attrs["end"]``.
    """
    frame = returns.to_frame(returns.name or "strategy") if isinstance(returns, pd.Series) else returns
    if common_period:
        frame = frame.dropna(how="any")

    yearly = {"periods_per_year": periods_per_year}
    compute: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "total_return": total_return,
        "cagr": lambda r: cagr(r, **yearly),
        "volatility": lambda r: annualized_volatility(r, **yearly),
        "sharpe": lambda r: sharpe_ratio(r, risk_free, **yearly),
        "sortino": lambda r: sortino_ratio(r, risk_free, **yearly),
        "max_drawdown": max_drawdown,
        "longest_drawdown_days": longest_drawdown_days,
        "calmar": lambda r: calmar_ratio(r, **yearly),
        "time_in_market": time_in_market,
        "hit_rate": hit_rate,
        "skew": skewness,
        "kurtosis": kurtosis,
    }

    table = pd.DataFrame({key: compute[key](frame) for key in METRICS}).T
    table.columns.name = None
    table.attrs["start"] = frame.dropna(how="all").index[0]
    table.attrs["end"] = frame.dropna(how="all").index[-1]
    return table


def format_metrics(metrics: pd.DataFrame, *, decimals: int = 2) -> pd.DataFrame:
    """
    Turn :func:`performance_metrics` into a readable table of strings.

    Percentages get a % sign, ratios a fixed number of decimals, days a
    thousands separator, and missing values a dash. The index becomes the
    display labels of :data:`METRICS`.
    """

    def fmt(value: float, unit: Unit) -> str:
        if value is None or math.isnan(value):
            return MISSING
        if unit == "percent":
            return f"{value:.{decimals}%}"
        if unit == "days":
            return f"{value:,.0f}"
        return f"{value:.{decimals}f}"

    rows = {
        label: [fmt(value, unit) for value in metrics.loc[key]]
        for key, (label, unit) in METRICS.items()
        if key in metrics.index
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=metrics.columns)


def metrics_table(
    returns: pd.Series | pd.DataFrame,
    *,
    risk_free: float | pd.Series = 0.0,
    periods_per_year: int = TRADING_DAYS,
    common_period: bool = True,
    decimals: int = 2,
) -> pd.DataFrame:
    """Shortcut: :func:`performance_metrics` followed by :func:`format_metrics`, for a post's table chunk."""
    metrics = performance_metrics(
        returns,
        risk_free=risk_free,
        periods_per_year=periods_per_year,
        common_period=common_period,
    )
    return format_metrics(metrics, decimals=decimals)