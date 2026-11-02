"""
The figures of the performance report, in the house style.

Every function follows the same contract, so it can be reused as it is in
ibexQuant's future ``plotting`` module (only the style import changes):

- input: daily **simple** returns, a ``pd.Series`` (one strategy) or a
  ``pd.DataFrame`` (one column per series);
- ``ax=None`` draws a new figure; passing an ``ax`` draws into it, so the same
  function fills a panel of the tearsheet (``export.py``);
- returns the ``Axes``, so the post can still adjust it (``ax.set_title(...)``);
- ``legend=False`` hides the legend (in a grid, one legend is enough);
- computes with ``_analytics`` and only draws: no number is calculated here.

The benchmark, when there is one, is named by ``benchmark="column"``: it is
drawn in a recessive gray and dashed. Every other column takes the categorical
palette in order (teal, rust, purple, mustard) with its own line style, so no
series depends on color alone.

Call ``investigating.style.apply_style()`` once at the top of the post first.
Colors are read from the active theme when a figure is drawn, so the same
function draws a light figure for the post or a dark one for a social card
(inside ``style_context("dark")``).
"""

from __future__ import annotations

import calendar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator, PercentFormatter

from investigating.performance._analytics import (
    ROLLING_6M,
    drawdown,
    equity_curve,
    monthly_returns_table,
    rolling_sharpe,
    rolling_volatility,
)

# The only coupling to the blog's identity: swap this import to reuse the module elsewhere.
from investigating.style import BENCHMARK_LINE_STYLE, LINE_STYLES, theme

DRAWDOWN_FILL_ALPHA = 0.18
HEATMAP_TEXT_THRESHOLD = 0.6  # |value| / max |value| above which cell text switches color


# =============================================================================
# Helpers
# =============================================================================


def _as_frame(returns: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """One column per series; a Series keeps its name (or becomes "Strategy")."""
    if isinstance(returns, pd.Series):
        return returns.to_frame(returns.name or "Strategy")
    return returns


def _axes(ax: Axes | None) -> Axes:
    """The given axes, or the axes of a new figure in the house size."""
    if ax is None:
        _, ax = plt.subplots()
    return ax


def _line_styles(columns: pd.Index, benchmark: str | None) -> dict[str, dict[str, object]]:
    """Color and line style of each column: the benchmark gray and dashed, the others in order."""
    palette = theme().categorical
    styles: dict[str, dict[str, object]] = {}
    others = [column for column in columns if column != benchmark]
    for i, column in enumerate(others):
        styles[column] = {"color": palette[i % len(palette)], "linestyle": LINE_STYLES[i % len(LINE_STYLES)]}
    if benchmark is not None:
        if benchmark not in columns:
            raise KeyError(f"benchmark {benchmark!r} is not a column of returns.")
        styles[benchmark] = {"color": theme().benchmark, "linestyle": BENCHMARK_LINE_STYLE}
    return styles


def _draw_order(columns: pd.Index, benchmark: str | None) -> list[str]:
    """Benchmark first, so the strategies are drawn on top of it."""
    return ([benchmark] if benchmark in columns else []) + [c for c in columns if c != benchmark]


def _date_axis(ax: Axes) -> None:
    """Compact date labels (2019, Jul, 2020, ...) without overlaps."""
    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    ax.margins(x=0)


def _legend(ax: Axes, columns: pd.Index, benchmark: str | None, *, loc: str = "upper left") -> None:
    """
    A legend only when there is more than one series (one series is named by the title).

    Entries follow the column order with the benchmark last, whatever the
    drawing order (the benchmark is drawn first, underneath).
    """
    if len(columns) < 2:
        return
    handles = dict(zip(*reversed(ax.get_legend_handles_labels()), strict=True))
    order = [c for c in columns if c != benchmark] + ([benchmark] if benchmark in columns else [])
    # A box in the background color, without border: lines passing behind the
    # legend are hidden instead of crossing the text.
    ax.legend(
        [handles[c] for c in order], order, loc=loc,
        frameon=True, facecolor=theme().background, edgecolor="none", framealpha=0.85,
    )


def _full_period(ax: Axes, frame: pd.DataFrame) -> None:
    """Show the whole period even when the first values are NaN (rolling warm-up), to align with other figures."""
    ax.set_xlim(frame.index[0], frame.index[-1])


# =============================================================================
# Figures
# =============================================================================


def plot_cumulative(
    returns: pd.Series | pd.DataFrame,
    *,
    benchmark: str | None = None,
    log_scale: bool = False,
    legend: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """
    Cumulative return of each series over time.

    The curve drawn is the growth of 1 (``∏(1 + r)``) and the axis is labeled as
    cumulative return (1.5 reads +50%). With ``log_scale=True`` equal vertical
    distances mean equal percentage moves, which keeps long histories readable.
    """
    frame = _as_frame(returns)
    ax = _axes(ax)
    styles = _line_styles(frame.columns, benchmark)

    for column in _draw_order(frame.columns, benchmark):
        equity = equity_curve(frame[column].dropna())
        ax.plot(equity.index, equity.to_numpy(), label=column, **styles[column])

    ax.axhline(1.0, color=theme().reference_line, linewidth=0.8, zorder=0)
    if log_scale:
        ax.set_yscale("log")
        # Default log ticks sit only at powers of 10 (1, 10, 100), too sparse for
        # most backtests; 1-2-3-5 steps read as round cumulative returns.
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0)))
        ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y - 1:+.0%}"))
    ax.set_title("Cumulative return" + (" (log scale)" if log_scale else ""))
    _date_axis(ax)
    if legend:
        _legend(ax, frame.columns, benchmark)
    return ax


def plot_drawdown(
    returns: pd.Series | pd.DataFrame,
    *,
    benchmark: str | None = None,
    legend: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """
    Underwater chart: how far each series is below its previous peak.

    The first strategy is filled, to show the depth and length of its
    drawdowns at a glance; the other series are lines only.
    """
    frame = _as_frame(returns)
    ax = _axes(ax)
    styles = _line_styles(frame.columns, benchmark)
    filled = next(column for column in frame.columns if column != benchmark)

    for column in _draw_order(frame.columns, benchmark):
        depth = drawdown(frame[column].dropna())
        ax.plot(depth.index, depth.to_numpy(), label=column, linewidth=1.2, **styles[column])
        if column == filled:
            ax.fill_between(
                depth.index, depth.to_numpy(), 0.0,
                color=styles[column]["color"], alpha=DRAWDOWN_FILL_ALPHA, linewidth=0,
            )

    ax.axhline(0.0, color=theme().reference_line, linewidth=0.8, zorder=0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Drawdown")
    _date_axis(ax)
    if legend:
        _legend(ax, frame.columns, benchmark, loc="lower left")
    return ax


def plot_monthly_heatmap(
    returns: pd.Series,
    *,
    decimals: int = 1,
    show_year: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """
    Monthly returns of one series as a year x month heatmap, plus the year's return.

    Diverging colors centered on zero: rust for losses, teal for gains, a
    neutral for months near 0% (light or dark, following the theme). The scale
    is symmetric, so equal gains and losses get equally strong colors. Each cell
    shows its value.

    With ``show_year=True`` a last column, set apart by a gap, shows the
    compounded return of each year. It has **its own color scale**: a year
    moves much more than a month, and on a shared scale every month would fade.
    """
    if isinstance(returns, pd.DataFrame):
        raise TypeError("plot_monthly_heatmap takes a single series; pass returns['column'].")
    table = monthly_returns_table(returns.dropna())
    months = table.drop(columns="total").to_numpy(dtype=float)
    n_years = len(table)

    if ax is None:
        # Height grows with the number of years, so cells stay legible.
        _, ax = plt.subplots(figsize=(9.0, 0.9 + 0.32 * n_years))

    # Columns as (x position, values): the 12 months at 0..11, the year after a gap.
    columns = [(float(m), months[:, m]) for m in range(12)]
    gap = 0.35
    if show_year:
        columns.append((12.0 + gap, table["total"].to_numpy(dtype=float)))

    month_limit = float(np.nanmax(np.abs(months))) or 1.0
    year_limit = float(np.nanmax(np.abs(table["total"]))) or 1.0

    for x, values in columns:
        limit = year_limit if x > 11.5 else month_limit
        ax.imshow(
            values[:, None], cmap=theme().diverging,
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
            extent=(x - 0.5, x + 0.5, n_years - 0.5, -0.5), aspect="auto",
        )
        for row, value in enumerate(values):
            if np.isnan(value):
                continue
            strong = abs(value) / limit > HEATMAP_TEXT_THRESHOLD
            ax.text(
                x, row, _percent(value, decimals),
                ha="center", va="center", fontsize=7.5,
                color=theme().text_on_strong if strong else theme().text,
            )

    # Background-colored gaps between cells instead of a grid.
    edges = [x + side for x, _ in columns for side in (-0.5, 0.5)]
    ax.set_xticks(sorted(set(edges)), minor=True)
    ax.set_yticks(np.arange(-0.5, n_years, 1), minor=True)
    ax.grid(which="minor", color=theme().background, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    labels = [calendar.month_abbr[m] for m in range(1, 13)] + (["Year"] if show_year else [])
    ax.set_xticks([x for x, _ in columns], labels)
    ax.set_xlim(-0.5, columns[-1][0] + 0.5)
    ax.set_ylim(n_years - 0.5, -0.5)
    ax.set_yticks(range(n_years), table.index.astype(str))
    ax.xaxis.tick_top()
    ax.set_title("Monthly returns", pad=18)
    return ax


def plot_rolling_sharpe(
    returns: pd.Series | pd.DataFrame,
    *,
    window: int = ROLLING_6M,
    benchmark: str | None = None,
    legend: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Sharpe ratio over a trailing window (default 126 sessions, about six months)."""
    frame = _as_frame(returns)
    ax = _axes(ax)
    styles = _line_styles(frame.columns, benchmark)
    rolling = rolling_sharpe(frame, window)

    for column in _draw_order(frame.columns, benchmark):
        ax.plot(rolling.index, rolling[column].to_numpy(), label=column, linewidth=1.2, **styles[column])

    ax.axhline(0.0, color=theme().reference_line, linewidth=0.8, zorder=0)
    ax.set_title(f"Rolling Sharpe ratio ({_window_label(window)})")
    _date_axis(ax)
    _full_period(ax, frame)
    if legend:
        _legend(ax, frame.columns, benchmark)
    return ax


def plot_rolling_volatility(
    returns: pd.Series | pd.DataFrame,
    *,
    window: int = ROLLING_6M,
    benchmark: str | None = None,
    legend: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Annualized volatility over a trailing window (default 126 sessions, about six months)."""
    frame = _as_frame(returns)
    ax = _axes(ax)
    styles = _line_styles(frame.columns, benchmark)
    rolling = rolling_volatility(frame, window)

    for column in _draw_order(frame.columns, benchmark):
        ax.plot(rolling.index, rolling[column].to_numpy(), label=column, linewidth=1.2, **styles[column])

    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title(f"Rolling volatility, annualized ({_window_label(window)})")
    _date_axis(ax)
    _full_period(ax, frame)
    if legend:
        _legend(ax, frame.columns, benchmark)
    return ax


def _window_label(window: int) -> str:
    """126 -> "6M", 252 -> "12M", 63 -> "3M"; other sizes in sessions."""
    months = window / 21
    return f"{months:.0f}M" if months.is_integer() else f"{window} sessions"


def _percent(value: float, decimals: int) -> str:
    """Format as a percentage, without the "-0%" that rounding leaves for small losses."""
    text = f"{value:.{decimals}%}"
    return text[1:] if text.startswith("-") and float(text[1:-1]) == 0 else text
