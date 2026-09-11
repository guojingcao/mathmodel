"""Controlled, vector-first export for CUMCM figures."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
def save_cumcm(fig, path: str | Path, formats: Sequence[str] = ("pdf", "svg", "png"), dpi: int = 600) -> list[Path]:
    """Save a figure to PDF/SVG/PNG; ``path`` is a stem, optionally with a suffix."""
    target=Path(path); target.parent.mkdir(parents=True, exist_ok=True); stem=target.with_suffix("")
    allowed={"pdf","svg","png"}; result=[]
    for fmt in formats:
        fmt=fmt.lower().lstrip(".")
        if fmt not in allowed: raise ValueError("formats must be pdf, svg, or png")
        out=stem.with_suffix("."+fmt)
        fig.savefig(out, dpi=dpi if fmt == "png" else None, bbox_inches="tight", pad_inches=.035)
        result.append(out)
    return result
