"""
The Investigating Series visual identity, for matplotlib.

Same structure as ibexQuant.style (tokens -> rcParams), with its own palette:
tones from the film *Joker* (2019) — the teal shirt, the rust suit, the
character's classic purple and the mustard vest.

Usage, once at the top of a post or notebook::

    from investigating.style import apply_style, CATEGORICAL, LINE_STYLES, MARKERS
    apply_style()

Fonts: IBM Plex Sans / Mono, read from ``assets/fonts/`` at the repo root
(copy them from ibexQuant). If the folder is missing, charts fall back to
matplotlib's default font with a warning instead of failing.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Final

import matplotlib as mpl
from cycler import cycler
from matplotlib import font_manager

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

# --- Typography & geometry ---------------------------------------------------
FONT_FAMILY: Final[str] = "IBM Plex Sans"
FONT_DIRECTORY: Final[Path] = Path(__file__).resolve().parents[2] / "assets" / "fonts"

LINE_WIDTH: Final[float] = 1.6
AXIS_LINE_WIDTH: Final[float] = 0.7
GRID_LINE_WIDTH: Final[float] = 0.6
FIGURE_SIZE: Final[tuple[float, float]] = (9.0, 4.0)
SCREEN_DPI: Final[int] = 110


def register_fonts() -> bool:
    """Register the IBM Plex files found in assets/fonts (and its subfolders). Returns False if none were found."""
    files = sorted(FONT_DIRECTORY.rglob("*.ttf"))   # includes subfolders (ibm_plex_sans/, ibm_plex_mono/)
    for path in files:
        font_manager.fontManager.addfont(str(path))
    return bool(files)


def rc_params() -> dict[str, object]:
    """matplotlib settings of the house style."""
    return {
        "font.size": 9.0,
        "text.color": TEXT,
        "figure.figsize": FIGURE_SIZE,
        "figure.dpi": SCREEN_DPI,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": AXIS_LINE_WIDTH,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.labelcolor": TEXT,
        "axes.titlesize": 10.0,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.prop_cycle": cycler(color=list(CATEGORICAL)),
        "grid.color": GRID,
        "grid.linewidth": GRID_LINE_WIDTH,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "ytick.major.size": 0.0,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": LINE_WIDTH,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }


def apply_style() -> None:
    """Apply the house style for the rest of the session (notebooks, Quarto posts)."""
    # Silence matplotlib's per-weight font fallback messages (they would print into rendered posts)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    settings = rc_params()
    if register_fonts():
        settings["font.family"] = [FONT_FAMILY]
    else:
        warnings.warn(
            f"No fonts found in {FONT_DIRECTORY}; using matplotlib's default font. "
            "Copy the IBM Plex .ttf files there (see assets/fonts/README.md).",
            stacklevel=2,
        )
    mpl.rcParams.update(settings)