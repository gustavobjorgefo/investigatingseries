"""
Images of the performance report for outside the blog: Substack, LinkedIn.

Inside a post, Quarto already renders and embeds every figure; nothing here is
needed for that. This module produces image files to post elsewhere:

- :func:`save_figure` — any figure as a PNG at twice the screen resolution,
  e.g. a light figure for the body of a Substack post.
- :func:`tearsheet` — the four main figures in one 2 x 2 figure.
- :func:`card` — a social media card in the style of the #5 cover: kicker,
  title, subtitle, the chart(s), and a footer with source and signature. Dark
  by default.
- :func:`tearsheet_card` — the tearsheet inside a card, ready to post.

Every chart inside a card is drawn by the same functions as in the post
(``plots.py``); the card only adds the frame and switches the theme.

Typical use, at the end of a post (with ``#| include: false``)::

    from investigating.performance import card, tearsheet_card, plot_cumulative

    tearsheet_card(returns, "figures/tearsheet.png", benchmark="VALE3",
                   kicker="Investigating Series #1", title="VALE3 volatility strategy",
                   subtitle="Daily, 2019–2025 · after costs", source="Yahoo Finance")

    with card("figures/cover.png", kicker="Investigating Series #1",
              title="VALE3 volatility strategy") as (fig, ax):
        plot_cumulative(returns, benchmark="VALE3", ax=ax)
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from investigating.performance.plots import (
    plot_cumulative,
    plot_drawdown,
    plot_monthly_heatmap,
    plot_rolling_sharpe,
)
from investigating.style import Mode, style_context, theme

EXPORT_DPI: Final[int] = 220  # twice the screen dpi of the house style: sharp on phones and retina screens

# Card sizes in inches; at EXPORT_DPI the 16:9 card is about 2200 x 1240 px.
CARD_WIDE: Final[tuple[float, float]] = (10.0, 5.625)   # 16:9 — a single chart, Substack cover
CARD_SQUARE: Final[tuple[float, float]] = (10.0, 10.0)  # 1:1 — the 2 x 2 tearsheet, LinkedIn feed

# Card layout, in inches, so it stays the same on any card size.
MARGIN: Final[float] = 0.8
HEADER: Final[float] = 1.5    # kicker + title + subtitle
FOOTER: Final[float] = 0.55   # source + signature
SIGNATURE: Final[str] = "Investigating Series · gustavojorgefilho.substack.com"


# =============================================================================
# Plain export
# =============================================================================


def save_figure(fig: Figure, path: str | Path, *, dpi: int = EXPORT_DPI) -> Path:
    """
    Save *fig* as an image (PNG unless the extension says otherwise), creating folders as needed.

    The background follows the figure, so a dark figure stays dark. Returns the path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.25, facecolor=fig.get_facecolor())
    return path


# =============================================================================
# Tearsheet
# =============================================================================


def _strategy_column(frame: pd.DataFrame, benchmark: str | None) -> str:
    """The first column that is not the benchmark: the series of the heatmap."""
    return next(column for column in frame.columns if column != benchmark)


def _fill_tearsheet(
    returns: pd.Series | pd.DataFrame,
    axes: Sequence[Axes],
    *,
    benchmark: str | None,
    log_scale: bool,
) -> None:
    """Draw the four panels: cumulative, drawdown, monthly heatmap (strategy only), rolling Sharpe."""
    frame = returns.to_frame(returns.name or "Strategy") if isinstance(returns, pd.Series) else returns
    # One legend (first panel) is enough: colors and line styles are the same in all four.
    plot_cumulative(frame, benchmark=benchmark, log_scale=log_scale, ax=axes[0])
    plot_drawdown(frame, benchmark=benchmark, ax=axes[1], legend=False)
    # Whole percents: the panel is half as wide as a post figure, "-12.4%" would not fit.
    plot_monthly_heatmap(frame[_strategy_column(frame, benchmark)], decimals=0, ax=axes[2])
    plot_rolling_sharpe(frame, benchmark=benchmark, ax=axes[3], legend=False)


def tearsheet(
    returns: pd.Series | pd.DataFrame,
    *,
    benchmark: str | None = None,
    log_scale: bool = False,
    figsize: tuple[float, float] = (11.0, 7.5),
) -> Figure:
    """
    The four main figures of the report in one 2 x 2 figure, in the active theme.

    Reading order: cumulative return | drawdown, then monthly returns | rolling
    Sharpe. Good for a quick look in a notebook, or with :func:`save_figure`.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(hspace=0.08)  # room between the date labels above and the titles below
    _fill_tearsheet(returns, axes.ravel(), benchmark=benchmark, log_scale=log_scale)
    return fig


# =============================================================================
# Social media cards
# =============================================================================


def _from_top(inches: float, height: float) -> float:
    """A vertical position given in inches from the top, as a figure fraction."""
    return 1.0 - inches / height


def _frame_card(
    fig: Figure,
    *,
    kicker: str,
    title: str,
    subtitle: str,
    source: str,
    signature: str,
) -> None:
    """Write the header and the footer of a card (texts in the active theme)."""
    t = theme()
    width, height = fig.get_size_inches()
    left, right = MARGIN / width, 1.0 - MARGIN / width

    if kicker:
        fig.text(left, _from_top(MARGIN, height), kicker.upper(),
                 color=t.accent, fontsize=12, va="top")
    fig.text(left, _from_top(MARGIN + 0.32, height), title,
             color=t.text, fontsize=26, va="top")
    if subtitle:
        fig.text(left, _from_top(MARGIN + 0.85, height), subtitle,
                 color=t.text_muted, fontsize=11, va="top")

    footer_y = (FOOTER * 0.55) / height
    if source:
        fig.text(left, footer_y, f"Source: {source}", color=t.text_muted, fontsize=8.5, va="center")
    if signature:
        fig.text(right, footer_y, signature, color=t.text_muted, fontsize=8.5, va="center", ha="right")


def _chart_area(fig: Figure) -> tuple[float, float, float, float]:
    """The rectangle left for the chart(s) between header and footer: (left, bottom, width, height)."""
    width, height = fig.get_size_inches()
    left = MARGIN / width
    bottom = (FOOTER + 0.45) / height
    top = _from_top(MARGIN + HEADER, height)
    return left, bottom, 1.0 - 2 * left, top - bottom


@contextmanager
def card(
    path: str | Path | None = None,
    *,
    kicker: str = "",
    title: str,
    subtitle: str = "",
    source: str = "",
    signature: str = SIGNATURE,
    grid: tuple[int, int] = (1, 1),
    size: tuple[float, float] = CARD_WIDE,
    mode: Mode = "dark",
    keep_titles: bool | None = None,
) -> Iterator[tuple[Figure, Axes | np.ndarray]]:
    """
    A social media card, used as a ``with`` block: draw inside it, it saves on exit.

    Inside the block the card's theme is active, so the plot functions draw in
    its colors. The axes come already placed below the header.

    Parameters
    ----------
    path : str or Path, optional
        Where to save the image when the block ends. ``None`` saves nothing
        (the figure is still returned, e.g. to show it in a notebook).
    kicker, title, subtitle : str
        Header texts. The kicker is shown in capitals, in the accent color.
    source, signature : str
        Footer texts, left and right.
    grid : (rows, cols), default (1, 1)
        Number of charts. One chart gives a single ``Axes``; more give an array.
    size : (width, height) in inches, default 16:9
        ``CARD_WIDE`` (16:9) or ``CARD_SQUARE`` (1:1), or any size.
    mode : "dark" or "light", default "dark"
    keep_titles : bool, optional
        Keep the chart titles ("Drawdown", ...). By default they are removed
        from a single chart (the card title says it) and kept in a grid.

    Yields
    ------
    (Figure, Axes or array of Axes)
    """
    with style_context(mode):
        fig = plt.figure(figsize=size)
        _frame_card(fig, kicker=kicker, title=title, subtitle=subtitle,
                    source=source, signature=signature)

        rows, cols = grid
        left, bottom, width, height = _chart_area(fig)
        spec = fig.add_gridspec(
            rows, cols, left=left, right=left + width, bottom=bottom, top=bottom + height,
            hspace=0.55 if rows > 1 else 0.0, wspace=0.18 if cols > 1 else 0.0,
        )
        axes = np.array([[fig.add_subplot(spec[r, c]) for c in range(cols)] for r in range(rows)])
        single = rows * cols == 1

        yield fig, (axes[0, 0] if single else axes)

        if not (keep_titles if keep_titles is not None else not single):
            for ax in axes.ravel():
                ax.set_title("")
        if path is not None:
            # No bbox_inches="tight": the card keeps its exact size and margins.
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=EXPORT_DPI, facecolor=fig.get_facecolor())


def tearsheet_card(
    returns: pd.Series | pd.DataFrame,
    path: str | Path | None = None,
    *,
    title: str,
    kicker: str = "",
    subtitle: str = "",
    source: str = "",
    benchmark: str | None = None,
    log_scale: bool = False,
    signature: str = SIGNATURE,
    size: tuple[float, float] = CARD_SQUARE,
    mode: Mode = "dark",
) -> Figure:
    """The 2 x 2 tearsheet inside a card (square and dark by default), saved to *path*."""
    with card(path, kicker=kicker, title=title, subtitle=subtitle, source=source,
              signature=signature, grid=(2, 2), size=size, mode=mode) as (fig, axes):
        _fill_tearsheet(returns, axes.ravel(), benchmark=benchmark, log_scale=log_scale)
    return fig
