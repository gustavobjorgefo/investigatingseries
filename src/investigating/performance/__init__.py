"""
Performance report of the Investigating Series: metrics, figures and export.

Usage in a post::

    from investigating.performance import to_simple_returns, sharpe_ratio, max_drawdown

    returns = to_simple_returns(log_returns)      # every function expects simple returns
    sharpe_ratio(returns)                          # Series in -> float
    sharpe_ratio(df[["strategy", "benchmark"]])    # DataFrame in -> one value per column

Modules:

- ``_analytics`` — the numbers (adapted from ibexQuant, QuantStats conventions).
- ``table`` — the metrics table (raw floats, then formatted strings).
- ``plots`` — the figures (one function per figure, each accepting ``ax=``).
- ``export`` — the images for social media (next step).
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
from investigating.performance.table import (
    METRICS,
    format_metrics,
    metrics_table,
    performance_metrics,
)
from investigating.performance.plots import (
    plot_cumulative,
    plot_drawdown,
    plot_monthly_heatmap,
    plot_rolling_sharpe,
    plot_rolling_volatility,
)

__all__ = [
    "METRICS",
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
    "format_metrics",
    "hit_rate",
    "kurtosis",
    "longest_drawdown_days",
    "max_drawdown",
    "metrics_table",
    "monthly_returns_table",
    "period_returns",
    "performance_metrics",
    "plot_cumulative",
    "plot_drawdown",
    "plot_monthly_heatmap",
    "plot_rolling_sharpe",
    "plot_rolling_volatility",
    "rolling_sharpe",
    "rolling_volatility",
    "sharpe_ratio",
    "skewness",
    "sortino_ratio",
    "time_in_market",
    "to_simple_returns",
    "total_return",
]