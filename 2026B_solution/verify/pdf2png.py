# -*- coding: utf-8 -*-
"""把 B1-B4.pdf 转为 PNG 页面图, 供视觉理解"""
import pymupdf, os
base = r"D:\My_MathModeling_Project\.dsh-uploads\session-7d22bc9c-54ee-4595-ada8-a753f80b1539"
outdir = r"D:\My_MathModeling_Project\2026B_solution\verify\pdf_pages"
os.makedirs(outdir, exist_ok=True)
for name in ["9e025c3361abf70e-B1", "871e3e41b267d222-B2", "1af4754dc33575c6-B3", "3002103bb7978e35-B4"]:
    doc = pymupdf.open(os.path.join(base, name + ".pdf"))
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=130)
        out = os.path.join(outdir, f"{name.split('-')[-1]}_p{i+1}.png")
        pix.save(out)
        print("saved", out, pix.width, "x", pix.height)
    doc.close()
