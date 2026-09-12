# -*- coding: utf-8 -*-
"""探针: 打印每个行间公式的 mathtext 完整报错, 定位真正的失败记号码。"""
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import md2docx as m                                  # noqa: E402

src = open(os.path.join("2026B_solution", "2026B_论文初稿.md"), encoding="utf-8").read()
eqs = re.findall(r"\$\$(.+?)\$\$", src, re.S)
print(f"共 {len(eqs)} 条行间公式")
for k, e in enumerate(eqs, 1):
    body = " ".join(e.split())
    cand, notes = m.sanitize_tex(body)
    try:
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${cand}$", fontsize=13)
        fig.savefig("nul", dpi=200, transparent=True, bbox_inches="tight")
        plt.close(fig)
        print(f"{k:2d} OK   {cand[:60]}")
    except Exception as ex:
        print(f"{k:2d} FAIL {str(ex)[:200]}")
        plt.close("all")
