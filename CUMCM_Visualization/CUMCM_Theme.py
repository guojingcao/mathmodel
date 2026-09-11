"""Central visual tokens for the CUMCM Visualization V2 system."""
from __future__ import annotations
from typing import Literal

MM = 1 / 25.4
FIGSIZE_SINGLE = (89 * MM, 62 * MM)
FIGSIZE_DOUBLE = (178 * MM, 92 * MM)
FIGSIZE_WIDE = (178 * MM, 70 * MM)
PRIMARY, SECONDARY, ACCENT = "#5B7FA3", "#668B74", "#B6534B"
SUCCESS, WARNING, NEUTRAL = SECONDARY, "#C08B52", "#9AA0A6"
TEXT, DARK_GRAY, GRID = "#333333", "#555555", "#D9D9D9"
PALETTE = (PRIMARY, ACCENT, SECONDARY, WARNING, "#806A8A", "#36566F", NEUTRAL)
FONT_TICK, FONT_LABEL, FONT_LEGEND, FONT_ANNOTATION, FONT_PANEL = 7.8, 8.5, 7.6, 7.2, 8.6
LW_AXIS, LW_MAIN, LW_SECONDARY, LW_GRID, LW_REFERENCE = .8, 1.45, 1.0, .45, .8

def figure_size(width: Literal["single", "double", "wide"] = "single") -> tuple[float, float]:
    """Return a standard paper figure size in inches."""
    sizes = {"single": FIGSIZE_SINGLE, "double": FIGSIZE_DOUBLE, "wide": FIGSIZE_WIDE}
    if width not in sizes:
        raise ValueError("width must be 'single', 'double', or 'wide'")
    return sizes[width]
