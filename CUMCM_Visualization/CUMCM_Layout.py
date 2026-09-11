"""Paper-panel layout utilities; panel labels are not paper figure numbers."""
from __future__ import annotations
import matplotlib.pyplot as plt
from CUMCM_Theme import FIGSIZE_DOUBLE, FONT_PANEL
from CUMCM_Style import set_cumcm_style, polish_axes

def cumcm_panel(figures=None, *, nrows: int = 2, ncols: int = 2, labels=None, figsize=FIGSIZE_DOUBLE):
    """Create a standard panel canvas.

    Existing Matplotlib figures cannot be safely merged across canvases; pass
    ``figures=None`` and draw into the returned axes. ``figures`` is accepted
    only as a count/intent marker for a clear API migration path.
    """
    if nrows < 1 or ncols < 1: raise ValueError("nrows and ncols must be positive")
    if figures is not None and len(figures) > nrows*ncols: raise ValueError("more figures than panel slots")
    set_cumcm_style(); fig, axes=plt.subplots(nrows,ncols,squeeze=False,figsize=figsize,layout="constrained")
    labels=labels or [f"({chr(97+i)})" for i in range(nrows*ncols)]
    if len(labels) < nrows*ncols: raise ValueError("labels must cover every panel")
    for i, ax in enumerate(axes.flat):
        polish_axes(ax); ax.text(-.02,1.03,labels[i],transform=ax.transAxes,fontsize=FONT_PANEL,ha="left",va="bottom")
    return fig, axes
