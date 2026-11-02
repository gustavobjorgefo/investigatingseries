"""
The Investigating Series visual identity, for matplotlib.

Same structure as ibexQuant.style (tokens -> rcParams), with its own palette:
tones from the film *Joker* (2019) — the teal shirt, the rust suit, the
character's classic purple and the mustard vest.

Two themes share that identity:

- ``"light"`` (default) — white background, for the figures inside posts.
- ``"dark"`` — near-black background, for the social media cards (LinkedIn,
  Substack cover). Its colors are lighter steps of the same hues, validated
  against the dark background, not an automatic inversion of the light ones.

Usage, once at the top of a post or notebook::

    from investigating.style import apply_style, CATEGORICAL, LINE_STYLES, MARKERS
    apply_style()                     # light, for the rest of the session

For a single figure in the other theme, without changing the session::

    with style_context("dark"):
        ...                           # figures created here are dark

Code that must follow the active theme (the performance figures) reads its
colors from :func:`theme` when it draws, never from the module constants.

Fonts: IBM Plex Sans / Mono, read from ``assets/fonts/`` at the repo root
(copy them from ibexQuant). If the folder is missing, charts fall back to
matplotlib's default font with a warning instead of failing.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import matplotlib as mpl
from cycler import cycler
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

Mode = Literal["light", "dark"]

# --- Palette: Joker (2019) ---------------------------------------------------
# Validated as a categorical sequence (in this order) for colorblind separation,
# normal-vision separation and contrast on white. Keep the order.
JOKER_TEAL: Final[str] = "#00826B"
JOKER_RUST: Final[str] = "#C83E26"
JOKER_PURPLE: Final[str] = "#7D4B91"
JOKER_MUSTARD: Final[str] = "#B08400"

CATEGORICAL: Final[tuple[str, ...]] = (JOKER_TEAL, JOKER_RUST, JOKER_PURPLE, JOKER_MUSTARD)

# Secondary encoding, so series never rely on color alone (print, colorblind readers)
LINE_STYLES: Final[tuple[str, ...]] = ("-", "--", ":", "-.")
MARKERS: Final[tuple[str, ...]] = ("o", "s", "^", "D")

# --- Neutrals ----------------------------------------------------------------
INK: Final[str] = "#1F2933"
SLATE: Final[str] = "#4B5563"
STEEL: Final[str] = "#6B7280"
ASH: Final[str] = "#D1D5DB"
WHITE: Final[str] = "#FFFFFF"

TEXT: Final[str] = INK
AXIS: Final[str] = SLATE
GRID: Final[str] = ASH
REFERENCE_LINE: Final[str] = STEEL

# --- Semantic colors (performance report) ------------------------------------
# Charts say "strategy" and "benchmark", never "teal" and "gray", so a palette
# change is one edit here. The benchmark is a neutral, so the strategy stands out.
# Its gray is lighter than STEEL on purpose: STEEL has almost the same lightness
# as the teal and the two merge for colorblind readers (checked: CVD ΔE 2.8 vs
# 9.3 for this gray, contrast on white still >= 3:1). Always drawn dashed too.
STRATEGY: Final[str] = JOKER_TEAL
BENCHMARK: Final[str] = "#8A919C"
BENCHMARK_LINE_STYLE: Final[str] = "--"

# --- Diverging colormap (polarity: losses vs. gains) --------------------------
# Two poles from the palette and a light neutral midpoint (never a hue at zero):
# rust for negative, teal for positive. Used by the monthly returns heatmap and
# by any signed matrix (e.g. correlations in post #3).
DIVERGING_MIDPOINT: Final[str] = "#F1F2F4"
DIVERGING: Final[LinearSegmentedColormap] = LinearSegmentedColormap.from_list(
    "joker_diverging", [JOKER_RUST, DIVERGING_MIDPOINT, JOKER_TEAL]
)

# --- Typography & geometry ---------------------------------------------------
FONT_FAMILY: Final[str] = "IBM Plex Sans"
FONT_DIRECTORY: Final[Path] = Path(__file__).resolve().parents[2] / "assets" / "fonts"

LINE_WIDTH: Final[float] = 1.6
AXIS_LINE_WIDTH: Final[float] = 0.7
GRID_LINE_WIDTH: Final[float] = 0.6
FIGURE_SIZE: Final[tuple[float, float]] = (9.0, 4.0)
SCREEN_DPI: Final[int] = 110


# =============================================================================
# Themes
# =============================================================================


@dataclass(frozen=True)
class Theme:
    """Every color a figure needs, by role. One instance per background."""

    name: Mode
    background: str
    text: str            # titles, labels, values
    text_muted: str      # subtitles, sources, secondary labels
    axis: str            # axis lines and tick labels
    grid: str
    reference_line: str  # the 0% / break-even line
    accent: str          # the kicker of a social card ("INVESTIGATING SERIES #1")
    categorical: tuple[str, ...]
    benchmark: str
    diverging: LinearSegmentedColormap
    text_on_strong: str  # text over a strongly colored heatmap cell

    @property
    def strategy(self) -> str:
        """The first categorical color: the main series of a chart."""
        return self.categorical[0]


LIGHT: Final[Theme] = Theme(
    name="light",
    background=WHITE,
    text=TEXT,
    text_muted=STEEL,
    axis=AXIS,
    grid=GRID,
    reference_line=REFERENCE_LINE,
    accent=JOKER_TEAL,
    categorical=CATEGORICAL,
    benchmark=BENCHMARK,
    diverging=DIVERGING,
    text_on_strong=WHITE,
)

# Dark steps of the same four hues, validated as a categorical sequence against
# the dark background #0F1619 (OKLCH lightness 0.48–0.67, CVD ΔE >= 8 between
# neighbors, contrast >= 3:1). The benchmark gray is darker than the lines, so it
# recedes (CVD ΔE 18 against the teal).
DARK_TEAL: Final[str] = "#27A88F"
DARK_RUST: Final[str] = "#E5533B"
DARK_PURPLE: Final[str] = "#A57FC4"
DARK_MUSTARD: Final[str] = "#B88A18"
DARK_BACKGROUND: Final[str] = "#0F1619"

DARK: Final[Theme] = Theme(
    name="dark",
    background=DARK_BACKGROUND,
    text="#E6EAED",
    text_muted="#8D979E",
    axis="#8D979E",
    grid="#253036",
    reference_line="#5F6773",
    accent="#5DD3B6",
    categorical=(DARK_TEAL, DARK_RUST, DARK_PURPLE, DARK_MUSTARD),
    benchmark="#5F6773",
    # Midpoint is a dark neutral: on a dark card, "about zero" should blend into
    # the background, and the poles light up.
    diverging=LinearSegmentedColormap.from_list(
        "joker_diverging_dark", [DARK_RUST, "#1E272C", DARK_TEAL]
    ),
    text_on_strong="#FFFFFF",
)

THEMES: Final[dict[str, Theme]] = {"light": LIGHT, "dark": DARK}

_active: Theme = LIGHT


def theme() -> Theme:
    """The theme in use right now (light unless set by apply_style or style_context)."""
    return _active


# =============================================================================
# matplotlib
# =============================================================================


def register_fonts() -> bool:
    """Register the IBM Plex files found in assets/fonts (and its subfolders). Returns False if none were found."""
    files = sorted(FONT_DIRECTORY.rglob("*.ttf"))   # includes subfolders (ibm_plex_sans/, ibm_plex_mono/)
    for path in files:
        font_manager.fontManager.addfont(str(path))
    return bool(files)


def rc_params(mode: Mode = "light") -> dict[str, object]:
    """matplotlib settings of the house style, in the given theme."""
    t = THEMES[mode]
    return {
        "font.size": 9.0,
        "text.color": t.text,
        "figure.figsize": FIGURE_SIZE,
        "figure.dpi": SCREEN_DPI,
        "figure.facecolor": t.background,
        "axes.facecolor": t.background,
        "axes.edgecolor": t.axis,
        "axes.linewidth": AXIS_LINE_WIDTH,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.labelcolor": t.text,
        "axes.titlesize": 10.0,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.titlecolor": t.text,
        "axes.prop_cycle": cycler(color=list(t.categorical)),
        "grid.color": t.grid,
        "grid.linewidth": GRID_LINE_WIDTH,
        "xtick.color": t.axis,
        "ytick.color": t.axis,
        "xtick.labelcolor": t.axis,
        "ytick.labelcolor": t.axis,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "ytick.major.size": 0.0,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": t.text,
        "lines.linewidth": LINE_WIDTH,
        "savefig.dpi": 300,
        "savefig.facecolor": t.background,
        "pdf.fonttype": 42,
    }


def _settings(mode: Mode) -> dict[str, object]:
    """rc_params plus the font family, if the font files are found."""
    # Silence matplotlib's per-weight font fallback messages (they would print into rendered posts)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    settings = rc_params(mode)
    if register_fonts():
        settings["font.family"] = [FONT_FAMILY]
    else:
        warnings.warn(
            f"No fonts found in {FONT_DIRECTORY}; using matplotlib's default font. "
            "Copy the IBM Plex .ttf files there (see assets/fonts/README.md).",
            stacklevel=3,
        )
    return settings


def apply_style(mode: Mode = "light") -> None:
    """Apply the house style for the rest of the session (notebooks, Quarto posts)."""
    global _active
    mpl.rcParams.update(_settings(mode))
    _active = THEMES[mode]


@contextmanager
def style_context(mode: Mode) -> Iterator[Theme]:
    """
    Use a theme only inside a ``with`` block; the previous style comes back after it.

    Example: a dark social card in the middle of a light post::

        with style_context("dark"):
            fig = ...
    """
    global _active
    previous = _active
    with mpl.rc_context(_settings(mode)):
        _active = THEMES[mode]
        try:
            yield _active
        finally:
            _active = previous