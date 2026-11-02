"""
Performance report of the Investigating Series: metrics, figures and export.

Usage in a post::

    from investigating.performance import to_simple_returns, sharpe_ratio, max_drawdown

    returns = to_simple_returns(log_returns)      # every function expects simple returns
    sharpe_ratio(returns)                          # Series in -> float
    sharpe_ratio(df[["strategy", "benchmark"]])    # DataFrame in -> one value per column

Modules:

- ``_analytics`` — the numbers (adapted from ibexQuant, QuantStats conventions).
- ``table``, ``plots``, ``export`` — the metrics table, the figures and the
  images for social media (coming in the next steps).
"""

from investigating.performance._analytics import (
    ROLLING_6M,
    TRADING_DAYS,
    annualized_volatility,
    cagr,
    calmar_ratio,
    cumulative_returns,
    drawdown,
    drawdown_periods,
    equity_curve,
    excess_returns,
    hit_rate,
    kurtosis,
    longest_drawdown_days,
    max_drawdown,
    monthly_returns_table,
    period_returns,
    rolling_sharpe,
    rolling_volatility,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    time_in_market,
    to_simple_returns,
    total_return,
)

__all__ = [
    "ROLLING_6M",
    "TRADING_DAYS",
    "annualized_volatility",
    "cagr",
    "calmar_ratio",
    "cumulative_returns",
    "drawdown",
    "drawdown_periods",
    "equity_curve",
    "excess_returns",
    "hit_rate",
    "kurtosis",
    "longest_drawdown_days",
    "max_drawdown",
    "monthly_returns_table",
    "period_returns",
    "rolling_sharpe",
    "rolling_volatility",
    "sharpe_ratio",
    "skewness",
    "sortino_ratio",
    "time_in_market",
    "to_simple_returns",
    "total_return",
]