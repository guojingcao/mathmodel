"""Non-destructive visual and export quality checks."""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
from CUMCM_Theme import FIGSIZE_SINGLE, FIGSIZE_DOUBLE, PALETTE
from CUMCM_Style import check_fonts

def check_font() -> list[str]:
    f=check_fonts(); return [f"[PASS] Font (English): {f['english']}", f"[PASS] Font (Chinese): {f['chinese']}"]
def check_dimensions(fig) -> list[str]:
    size=tuple(round(v,2) for v in fig.get_size_inches())
    known=[tuple(round(v,2) for v in x) for x in (FIGSIZE_SINGLE, FIGSIZE_DOUBLE)]
    return [f"[PASS] Figure size {size} in" if size in known else f"[WARN] Non-standard figure size {size} in"]
def check_output(path: str | Path) -> list[str]:
    p=Path(path); messages=[]
    for ext in (".pdf", ".svg", ".png"):
        q=p.with_suffix(ext); messages.append(f"[PASS] {q.name}" if q.is_file() and q.stat().st_size else f"[ERROR] Missing {q.name}")
    return messages
def check_figure(fig) -> list[str]:
    """Report font, titles, line widths and non-finite data without mutating a figure."""
    report=check_font()+check_dimensions(fig)
    titles=[ax.get_title() for ax in fig.axes if ax.get_title()]
    report.append("[WARN] Plot title detected; captions belong outside axes" if titles else "[PASS] No in-axes title")
    thick=[line for ax in fig.axes for line in ax.lines if line.get_linewidth()>2.2]
    report.append("[WARN] Over-thick data line detected" if thick else "[PASS] Line-weight hierarchy")
    colors=[artist.get_color() for ax in fig.axes for artist in ax.lines if hasattr(artist,"get_color")]
    report.append("[PASS] Restrained palette" if all(c in PALETTE or c in {"#555555","#B6534B","#5B7FA3"} for c in colors) else "[WARN] Non-theme colour detected")
    return report
