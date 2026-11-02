"""
Performance analytics: the numbers behind the performance report.

Adapted from ibexQuant (github.com/gustavobjorgefo/ibexQuant, commit 6fb7b4c,
``src/ibexQuant/analytics/``: ``returns.py``, ``metrics.py``,
``distribution.py`` and ``rolling.py``). The formulas are the same; the strict
input validation of ibexQuant was trimmed to keep the code short and easy to
read in a blog repository.

Metric definitions follow QuantStats (github.com/ranaroussi/quantstats), so the
tables in the older posts, generated with QuantStats, stay comparable. The few
places where the two libraries differ are noted in the docstrings.

Conventions
-----------
- Input: periodic **simple** returns, as a ``pd.Series`` (one strategy) or a
  ``pd.DataFrame`` (one column per series, e.g. strategy and benchmark), with
  a ``DatetimeIndex``. Use :func:`to_simple_returns` to convert log returns.
- Output keeps the shape: a Series gives a float (or a Series for time series
  functions); a DataFrame gives one value (or one column) per column.
- Each column is measured over its own observed period: leading and trailing
  ``NaN`` are dropped, so a benchmark that starts later does not shorten the
  strategy.
- Annualization uses ``periods_per_year`` (252 for daily B3 or US sessions).
- Standard deviations use ``ddof=1`` (sample estimate).
- A metric that is undefined for the input (a ratio over zero risk, say) is
  ``NaN``, never ``inf``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final, Literal

import numpy as np
import pandas as pd

TRADING_DAYS: Final[int] = 252
ROLLING_6M: Final[int] = 126  # about six months of daily sessions

Period = Literal["month", "quarter", "year"]
PERIOD_RULES: Final[dict[str, str]] = {"month": "ME", "quarter": "QE", "year": "YE"}

Returns = pd.Series | pd.DataFrame


# =============================================================================
# Helpers
# =============================================================================


def _check(returns: Returns) -> Returns:
    """Light input check: pandas object with a sorted DatetimeIndex."""
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError(f"returns must be a pd.Series or pd.DataFrame, got {type(returns)!r}.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex.")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns index must be sorted in increasing order.")
    return returns


def _per_series(returns: Returns, statistic: Callable[[pd.Series], float]) -> float | pd.Series:
    """Apply *statistic* to each series (without its edge NaN): float for a Series, Series for a DataFrame."""
    returns = _check(returns)
    if isinstance(returns, pd.Series):
        return statistic(returns.dropna())
    return pd.Series(
        {column: statistic(values.dropna()) for column, values in returns.items()},
        dtype="float64",
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``; NaN when the denominator is zero or NaN."""
    if math.isnan(denominator) or denominator == 0.0:
        return math.nan
    return numerator / denominator


def _std(values: pd.Series) -> float:
    """Sample standard deviation (ddof=1); exactly 0 for constant values, NaN below two values."""
    if len(values) < 2:
        return math.nan
    # The mean of a repeated value is not always that value in floating point,
    # which would leave ~1e-17 of spurious dispersion; compare exactly instead.
    if values.max() == values.min():
        return 0.0
    return float(values.std(ddof=1))


# =============================================================================
# Returns: conversion, compounding and drawdowns
# =============================================================================


def to_simple_returns(log_returns: Returns) -> Returns:
    """
    Convert log returns into simple returns, ``exp(r) - 1``.

    Every function in this module expects simple returns, which compound by
    multiplication, ``∏(1 + r)``. Log returns add up instead, so feeding them
    as if they were simple overstates gains and understates losses.
    """
    return np.expm1(_check(log_returns))


def equity_curve(returns: Returns, *, initial_value: float = 1.0) -> Returns:
    """Compound returns into an equity curve, ``V_0 · ∏(1 + r)``."""
    return initial_value * (1.0 + _check(returns)).cumprod()


def cumulative_returns(returns: Returns) -> Returns:
    """Compound returns into the cumulative return, ``∏(1 + r) - 1``."""
    return (1.0 + _check(returns)).cumprod() - 1.0


def drawdown(returns: Returns) -> Returns:
    """
    Drawdown at each period, ``V_t / max(V_0, ..., V_t) - 1``.

    The running peak starts at the initial capital (1.0), not at the first
    equity value, so a loss on the first day is already a drawdown. Zero at a
    new peak, negative below it.
    """
    equity = (1.0 + _check(returns)).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    return equity / peak - 1.0


def drawdown_periods(returns: pd.Series) -> pd.DataFrame:
    """
    List every drawdown episode of a single series, deepest first.

    An episode starts on the first period below the running peak and ends on
    the last period before the equity closes back at or above it.

    Returns
    -------
    pd.DataFrame
        One row per episode, with columns:

        - ``start`` — first period under water.
        - ``trough`` — period of the deepest point.
        - ``end`` — last period under water (the last date of the series if
          the episode is still open).
        - ``recovered`` — whether the equity got back to the peak.
        - ``depth`` — drawdown at the trough (negative).
        - ``days`` — calendar days from ``start`` to ``end``, both included.
          This is the QuantStats convention (ibexQuant counts sessions from
          the peak to the recovery instead).
        - ``periods`` — sessions under water.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("drawdown_periods takes a single series (pd.Series).")
    depth = drawdown(returns.dropna())
    underwater = depth < 0.0

    # Consecutive underwater periods share an id; each new episode increments it.
    episode = (underwater != underwater.shift(fill_value=False)).cumsum()

    rows = []
    for _, segment in depth[underwater].groupby(episode[underwater]):
        start, end = segment.index[0], segment.index[-1]
        rows.append(
            {
                "start": start,
                "trough": segment.idxmin(),
                "end": end,
                "recovered": end != depth.index[-1],
                "depth": float(segment.min()),
                "days": (end - start).days + 1,
                "periods": len(segment),
            }
        )

    columns = ["start", "trough", "end", "recovered", "depth", "days", "periods"]
    table = pd.DataFrame(rows, columns=columns)
    return table.sort_values("depth", ignore_index=True)


def period_returns(returns: Returns, period: Period) -> Returns:
    """
    Compound returns within each calendar month, quarter or year.

    Rows are labeled by the period's calendar end (2024-01-31 for January). A
    period with no observation is NaN, never 0%. The first and last periods
    may be partial.
    """
    if period not in PERIOD_RULES:
        raise ValueError(f"period must be one of {tuple(PERIOD_RULES)}, got {period!r}.")
    # min_count=1: a period with no observation is NaN, not an empty product of 1.0.
    return (1.0 + _check(returns)).resample(PERIOD_RULES[period]).prod(min_count=1) - 1.0


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """
    Monthly returns of a single series as a year x month table.

    Rows are years; columns are ``1`` to ``12`` (the month's compounded return)
    and ``"total"`` (the year's compounded return). Missing months are NaN.
    This is the input of the monthly returns heatmap.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("monthly_returns_table takes a single series (pd.Series).")
    monthly = period_returns(returns, "month")
    yearly = period_returns(returns, "year")

    table = (
        pd.DataFrame(
            {"year": monthly.index.year, "month": monthly.index.month, "value": monthly.to_numpy()}
        )
        .pivot(index="year", columns="month", values="value")
        .reindex(index=yearly.index.year, columns=range(1, 13))
    )
    table["total"] = yearly.to_numpy()
    return table.rename_axis(index="year", columns=None)


def excess_returns(
    returns: Returns, risk_free: float | pd.Series = 0.0, *, periods_per_year: int = TRADING_DAYS
) -> Returns:
    """
    Subtract the risk-free return from each period, ``r_t - rf_t``.

    - ``float`` *risk_free* is an **annual** rate, converted to the period
      geometrically: ``(1 + rf)^(1 / periods_per_year) - 1``.
    - ``pd.Series`` *risk_free* holds **periodic** returns already (e.g. the
      daily CDI), aligned on the dates of *returns*.
    """
    returns = _check(returns)
    if isinstance(risk_free, pd.Series):
        return returns.sub(risk_free.reindex(returns.index), axis=0)
    periodic_rate = (1.0 + risk_free) ** (1.0 / periods_per_year) - 1.0
    return returns - periodic_rate


# =============================================================================
# Metrics: one number per series
# =============================================================================


def total_return(returns: Returns) -> float | pd.Series:
    """Total compounded return, ``∏(1 + r) - 1``."""
    return _per_series(returns, lambda s: float((1.0 + s).prod() - 1.0))


def cagr(returns: Returns, *, periods_per_year: int = TRADING_DAYS) -> float | pd.Series:
    """
    Compound annual growth rate, ``(1 + total)^(N / n) - 1``.

    ``N`` is *periods_per_year* and ``n`` the number of observed periods, so
    time is measured in sessions. Over less than a year it extrapolates: a few
    strong sessions give a very large CAGR.
    """

    def statistic(s: pd.Series) -> float:
        growth = float((1.0 + s).prod())
        return growth ** (periods_per_year / len(s)) - 1.0

    return _per_series(returns, statistic)


def annualized_volatility(
    returns: Returns, *, periods_per_year: int = TRADING_DAYS
) -> float | pd.Series:
    """Annualized volatility, ``std(r) · √N``."""
    scale = math.sqrt(periods_per_year)
    return _per_series(returns, lambda s: _std(s) * scale)


def sharpe_ratio(
    returns: Returns, risk_free: float | pd.Series = 0.0, *, periods_per_year: int = TRADING_DAYS
) -> float | pd.Series:
    """
    Annualized Sharpe ratio, ``mean(r - rf) / std(r - rf) · √N``.

    Mean and standard deviation are both taken over excess returns (Lo, 2002).
    """
    excess = excess_returns(returns, risk_free, periods_per_year=periods_per_year)
    scale = math.sqrt(periods_per_year)
    return _per_series(excess, lambda s: _safe_ratio(float(s.mean()), _std(s)) * scale)


def sortino_ratio(
    returns: Returns, risk_free: float | pd.Series = 0.0, *, periods_per_year: int = TRADING_DAYS
) -> float | pd.Series:
    """
    Annualized Sortino ratio, ``mean(r - rf) / DD · √N``.

    ``DD`` is the downside deviation, ``√mean(min(r - rf, 0)²)``, averaged over
    **all** periods: periods above the target contribute zero (Sortino and
    Price, 1994). Unlike Sharpe, gains are not penalized as risk.
    """
    excess = excess_returns(returns, risk_free, periods_per_year=periods_per_year)
    scale = math.sqrt(periods_per_year)

    def statistic(s: pd.Series) -> float:
        downside = math.sqrt(float((s.clip(upper=0.0) ** 2).mean()))
        return _safe_ratio(float(s.mean()), downside) * scale

    return _per_series(excess, statistic)


def max_drawdown(returns: Returns) -> float | pd.Series:
    """Deepest fall from a running peak, as a non-positive number (-0.25 is a 25% drawdown)."""
    return _per_series(returns, lambda s: float(drawdown(s).min()))


def calmar_ratio(returns: Returns, *, periods_per_year: int = TRADING_DAYS) -> float | pd.Series:
    """Calmar ratio, ``CAGR / |max drawdown|``; NaN if the series never fell."""

    def statistic(s: pd.Series) -> float:
        return _safe_ratio(cagr(s, periods_per_year=periods_per_year), abs(max_drawdown(s)))

    return _per_series(returns, statistic)


def skewness(returns: Returns) -> float | pd.Series:
    """Sample skewness (bias-adjusted, as pandas and QuantStats compute it)."""
    return _per_series(returns, lambda s: float(s.skew()) if len(s) >= 3 else math.nan)


def kurtosis(returns: Returns) -> float | pd.Series:
    """Sample **excess** kurtosis (0 for a normal distribution), bias-adjusted."""
    return _per_series(returns, lambda s: float(s.kurt()) if len(s) >= 4 else math.nan)


# --- Not in ibexQuant (added for the report) ---------------------------------


def longest_drawdown_days(returns: Returns) -> float | pd.Series:
    """
    Longest drawdown episode, in calendar days (QuantStats convention).

    Counts an episode still open at the end of the series; 0 if the series
    never fell. See :func:`drawdown_periods`.
    """

    def statistic(s: pd.Series) -> float:
        days = drawdown_periods(s)["days"]
        return float(days.max()) if len(days) else 0.0

    return _per_series(returns, statistic)


def time_in_market(returns: Returns) -> float | pd.Series:
    """
    Fraction of periods with a non-zero return, a proxy for being invested.

    Computed from returns alone, as QuantStats ``exposure`` does (QuantStats
    also rounds it up to a whole percent; we do not). A day in the market with
    an exactly zero return counts as out, which is rare with daily prices.
    """
    return _per_series(returns, lambda s: float((s != 0.0).mean()))


def hit_rate(returns: Returns) -> float | pd.Series:
    """
    Fraction of positive periods among the periods with a non-zero return.

    QuantStats convention (``win_rate``): days out of the market are ignored,
    so a strategy that is flat most of the time is judged only on the days it
    traded. ibexQuant's ``positive_ratio`` counts every period instead.
    """

    def statistic(s: pd.Series) -> float:
        active = s[s != 0.0]
        return float((active > 0.0).mean()) if len(active) else math.nan

    return _per_series(returns, statistic)


# =============================================================================
# Rolling metrics: one value per period, over a trailing window
# =============================================================================


def rolling_volatility(
    returns: Returns, window: int = ROLLING_6M, *, periods_per_year: int = TRADING_DAYS
) -> Returns:
    """Annualized volatility over a trailing window of *window* periods (NaN during warm-up)."""
    deviation = _check(returns).rolling(window, min_periods=window).std(ddof=1)
    return deviation * math.sqrt(periods_per_year)


def rolling_sharpe(
    returns: Returns,
    window: int = ROLLING_6M,
    risk_free: float | pd.Series = 0.0,
    *,
    periods_per_year: int = TRADING_DAYS,
) -> Returns:
    """Annualized Sharpe ratio over a trailing window of *window* periods (NaN during warm-up)."""
    excess = excess_returns(returns, risk_free, periods_per_year=periods_per_year)
    rolling = excess.rolling(window, min_periods=window)
    deviation = rolling.std(ddof=1)
    # A window with zero dispersion has no defined Sharpe: NaN instead of inf.
    return rolling.mean() / deviation.where(deviation != 0.0) * math.sqrt(periods_per_year)