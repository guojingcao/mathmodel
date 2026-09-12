# -*- coding: utf-8 -*-
"""把 2026B_论文初稿.md 转成 Word 投稿版(2026B_论文初稿.docx)。

设计取舍:
  * 行间公式($$...$$) -> matplotlib mathtext 渲染为 300 dpi 透明 PNG 居中插入
    (mathtext 是 LaTeX 子集渲染器, 论文用到的 \\frac/\\bigcap/\\sum/\\operatorname 等均支持;
     若个别公式渲染失败, 退化为 Unicode 文本并打印告警)。
  * 行内公式($...$) -> Unicode 转写(避免 Word 行内图片基线问题)。
  * 表格 -> 真 Word 表格(表头加粗+底纹, 全边框); 图件 -> 按"可视化"段落自动嵌入并编号;
  * 排版: A4, 页边距 2.5/2.5/3.0/2.5 cm, 正文宋体·Times New Roman 小四, 1.5 倍行距,
    标题黑体, 页脚页码。

用法: python md2docx.py [输入.md] [输出.docx]
"""
import hashlib
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from docx import Document                            # noqa: E402
from docx.enum.section import WD_SECTION             # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT       # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING   # noqa: E402
from docx.oxml import OxmlElement                    # noqa: E402
from docx.oxml.ns import qn                          # noqa: E402
from docx.shared import Cm, Pt, RGBColor             # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "2026B_solution", "word_assets")

# 图件说明(按文件名 -> 题号与标题)
FIG_CAP = {
    ("Problem1_Localization", "fig01_single_wedge"): "单站有界测向角域与半平面等价形式",
    ("Problem1_Localization", "fig02_feasible_region"): "可行域的内包/外包与网格叶格",
    ("Problem1_Localization", "fig03_number"): "观测点数对可行域面积的影响",
    ("Problem1_Localization", "fig04_error"): "测向误差界对可行域面积的影响",
    ("Problem1_Localization", "fig05_angle"): "交会角对定位误差的影响",
    ("Problem2_Observation_Optimization", "fig01_optimized_layout"): "优化后的观测布局",
    ("Problem2_Observation_Optimization", "fig02_layout_regions"): "四种布局的可行域对比",
    ("Problem2_Observation_Optimization", "fig03_number"): "点数对可行域面积的影响",
    ("Problem2_Observation_Optimization", "fig04_angle"): "交会角对定位误差的影响",
    ("Problem2_Observation_Optimization", "fig05_greedy_convergence"): "字典序贪心的收敛轨迹",
    ("Problem3_Robot", "fig01_time_vs_sources"): "38 局实机：虚拟时间随源数变化",
    ("Problem3_Robot", "fig02_phase_time"): "实机 36 局分阶段时间占比（合计 100 %）",
    ("Problem4_Robot", "fig01_mesh_ab"): "旧默认网格与最终最小覆盖设计的 A/B 对比",
    ("Problem4_Robot", "fig02_phase_time"): "最终 27 点设计 4 局分阶段时间（逐局闭合）",
}
PROB_NAME = {"Problem1_Localization": "问题一", "Problem2_Observation_Optimization": "问题二",
             "Problem3_Robot": "问题三", "Problem4_Robot": "问题四"}

# 行内 LaTeX -> Unicode(覆盖论文用到的符号)
UNI = {
    r"\Omega": "Ω", r"\omega": "ω", r"\theta": "θ", r"\delta": "δ", r"\gamma": "γ",
    r"\alpha": "α", r"\beta": "β", r"\mu": "μ", r"\sigma": "σ", r"\pi": "π",
    r"\lambda": "λ", r"\rho": "ρ", r"\tau": "τ", r"\phi": "φ", r"\varepsilon": "ε",
    r"\le": "≤", r"\ge": "≥", r"\ne": "≠", r"\approx": "≈", r"\equiv": "≡",
    r"\times": "×", r"\cdot": "·", r"\pm": "±", r"\to": "→", r"\Rightarrow": "⇒",
    r"\Longrightarrow": "⟹", r"\in": "∈", r"\notin": "∉", r"\subseteq": "⊆",
    r"\subset": "⊂", r"\cap": "∩", r"\cup": "∪", r"\bigcap": "⋂", r"\bigcup": "⋃",
    r"\setminus": "∖", r"\emptyset": "∅", r"\varnothing": "∅", r"\infty": "∞",
    r"\sum": "∑", r"\prod": "∏", r"\sqrt": "√", r"\lceil": "⌈", r"\rceil": "⌉",
    r"\lfloor": "⌊", r"\rfloor": "⌋", r"\dots": "…", r"\ldots": "…", r"\cdots": "⋯",
    r"\quad": " ", r"\qquad": "  ", r"\ ": " ", r"\%": "%", r"\lVert": "‖",
    r"\rVert": "‖", r"\Vert": "‖", r"\|": "‖", r"\angle": "∠", r"\perp": "⊥",
    r"\parallel": "∥", r"\min": "min", r"\max": "max", r"\arg": "arg", r"\log": "log",
    r"\exp": "exp", r"\sin": "sin", r"\cos": "cos", r"\tan": "tan", r"\det": "det",
    r"\mathbb{R}": "ℝ", r"\mathbb{S}^1": "S¹", r"\mathbb{S}": "S", r"\mathrm": "",
}
SUP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷",
       "8": "⁸", "9": "⁹", "-": "⁻", "n": "ⁿ", "k": "ᵏ", "m": "ᵐ", "T": "ᵀ", "*": "*"}
SUB = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇",
       "8": "₈", "9": "₉", "-": "₋", "i": "ᵢ", "j": "ⱼ", "n": "ₙ", "k": "ₖ", "m": "ₘ",
       "c": "꜀", "s": "ₛ", "t": "ₜ", "x": "ₓ", "a": "ₐ", "e": "ₑ", "o": "ₒ", "u": "ᵤ",
       "v": "ᵥ", "p": "ₚ", "R": "ᵣ", "in": "ᵢₙ", "out": "ₒᵤₜ", "ok": "ₒₖ",
       "fail": "բ", "clear": "꜀ₗ", "switch": "ₛᵥ"}


def uni_math(s):
    """行内公式 -> 尽量可读的 Unicode 文本。"""
    t = s
    t = re.sub(r"\\operatorname\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\mathbb\{([^}]*)\}", lambda m: {"R": "ℝ", "S": "S"}.get(m.group(1), m.group(1)), t)
    for k in sorted(UNI, key=len, reverse=True):
        t = t.replace(k, UNI[k])
    # 上下标
    t = re.sub(r"\^\{?([0-9nkmiT+\-*]+)\}?", lambda m: "".join(SUP.get(c, c) for c in m.group(1)), t)
    t = re.sub(r"_\{([A-Za-z0-9]+)\}", lambda m: "".join(SUB.get(c, c) for c in m.group(1)), t)
    t = re.sub(r"_([0-9A-Za-z])", lambda m: SUB.get(m.group(1), "_" + m.group(1)), t)
    t = t.replace("{", "").replace("}", "").replace("\\", "")
    return t.strip()


def sanitize_tex(tex):
    """把 LaTeX 映射到 mathtext 子集; 返回 (纯数学部分, 抽出的中文说明列表)。

    关键: mathtext 无法渲染中文, 故把 \\text{中文...} 抽出来, 由调用方作为正文附在公式后。
    """
    notes = []

    def take(m):
        s = m.group(1).strip()
        if s:
            notes.append(s)
        return r"\ "
    t = re.sub(r"\\text\{([^}]*)\}", take, tex)
    t = re.sub(r"\\mathrm\{([^}]*)\}", lambda m: (notes.append(m.group(1)), r"\ ")[1]
               if re.search(r"[\u4e00-\u9fff]", m.group(1)) else r"\mathrm{" + m.group(1) + "}",
               t)
    t = re.sub(r"\\operatorname\{([^}]*)\}", r"\\mathrm{\1}", t)
    # 去掉尺寸命令(必须加负向断言, 否则 \bigcap 里的 \big 也会被吃掉 -> 变成 cap)
    t = re.sub(r"\\(bigl|bigr|Bigl|Bigr|biggl|biggr|big|Big|left|right)(?![A-Za-z])\s*", "", t)
    # \overline\Omega 这类缺花括号的装饰命令: mathtext 要求 \overline{...}
    t = re.sub(r"\\(overline|underline|hat|bar|vec|tilde|widehat|widetilde)\s*"
               r"(\\[A-Za-z]+|[A-Za-z])", r"\\\1{\2}", t)
    t = t.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    # \frac12 -> \frac{1}{2}(mathtext 要求两个参数都带花括号)
    t = re.sub(r"\\frac\s*([0-9A-Za-z])\s*([0-9A-Za-z])(?![0-9A-Za-z])", r"\\frac{\1}{\2}", t)
    for a, b in ((r"\bigcap", r"\cap"), (r"\bigcup", r"\cup"),
                 (r"\qquad", r"\ \ \ \ "), (r"\quad", r"\ \ "),
                 (r"\lVert", "‖"), (r"\rVert", "‖"), (r"\Vert", "‖"), (r"\|", "‖"),
                 # mathtext 只认 \leq/\geq/\neq(不认 \le/\ge/\ne), 以及 \Longrightarrow
                 (r"\le", r"\leq"), (r"\ge", r"\geq"), (r"\ne", r"\neq"),
                 (r"\Longrightarrow", r"\Rightarrow"), (r"\longrightarrow", r"\rightarrow"),
                 (r"\to", r"\rightarrow"), (r"\bmod", r"\ \mathrm{mod}\ "),
                 (r"\max", r"\mathrm{max}"), (r"\min", r"\mathrm{min}"),
                 (r"\arg", r"\mathrm{arg}"), (r"\sup", r"\mathrm{sup}"),
                 (r"\inf", r"\mathrm{inf}"), (r"\det", r"\mathrm{det}"),
                 (r"\log", r"\mathrm{log}"), (r"\exp", r"\mathrm{exp}"),
                 (r"\dots", r"\ldots"), (r"\_", "-")):
        t = t.replace(a, b)
    t = re.sub(r"\\(mathbb|mathcal|mathrm|mathbf)\s+([A-Za-z])", r"\\\1{\2}", t)
    return t, notes


def render_display_eq(tex, idx):
    """行间公式 -> (PNG 路径或 None, 抽出的中文说明文本)。"""
    os.makedirs(ASSETS, exist_ok=True)
    body = tex.strip().replace("\n", " ")
    body = re.sub(r"\\label\{[^}]*\}", "", body)
    cand, notes = sanitize_tex(body)
    key = hashlib.md5((cand + body).encode("utf-8")).hexdigest()[:10]
    out = os.path.join(ASSETS, f"eq_{idx:02d}_{key}.png")
    note_txt = "　（" + "；".join(notes) + "）" if notes else ""
    if os.path.exists(out):
        return out, note_txt
    last = None
    for c in (cand, body):
        if not c.strip():
            continue
        # 长公式在 300 dpi 下可能触及 matplotlib 图像尺寸上限 -> 逐级降级
        for dpi, fs in ((300, 13), (200, 12), (150, 11), (110, 10)):
            try:
                fig = plt.figure(figsize=(0.01, 0.01))
                fig.text(0, 0, f"${c}$", fontsize=fs)
                fig.savefig(out, dpi=dpi, transparent=True,
                            bbox_inches="tight", pad_inches=0.03)
                plt.close(fig)
                return out, note_txt
            except Exception as e:
                last = e
                try:
                    plt.close("all")
                except Exception:
                    pass
    print(f"  [告警] 公式 {idx} 渲染失败({type(last).__name__}): {cand[:70]}")
    return None, ""


def set_font(run, ascii_font="Times New Roman", ea="宋体", size=None, bold=None,
             mono=False, color=None, ea_font=None):
    ea = ea_font or ea
    run.font.name = "Consolas" if mono else ascii_font
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    rf.set(qn("w:ascii"), "Consolas" if mono else ascii_font)
    rf.set(qn("w:hAnsi"), "Consolas" if mono else ascii_font)
    rf.set(qn("w:eastAsia"), ea)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color:
        run.font.color.rgb = color


def add_runs(par, text, base_size=12, ea="宋体", ascii_font="Times New Roman"):
    """解析行内标记: **粗体**、`代码`、$行内公式$。"""
    tokens = re.split(r"(\*\*[^*]+\*\*|`[^`]+`|\$[^$]+\$)", text)
    for tk in tokens:
        if not tk:
            continue
        if tk.startswith("**") and tk.endswith("**") and len(tk) > 4:
            r = par.add_run(tk[2:-2]); set_font(r, ascii_font, ea, base_size, bold=True)
        elif tk.startswith("`") and tk.endswith("`") and len(tk) > 2:
            r = par.add_run(tk[1:-1]); set_font(r, size=base_size - 1.5, mono=True,
                                                ea_font="Consolas")
        elif tk.startswith("$") and tk.endswith("$") and len(tk) > 2:
            r = par.add_run(uni_math(tk[1:-1]))
            set_font(r, "Cambria Math", ea, base_size)
        else:
            r = par.add_run(re.sub(r"\s+", " ", tk))
            set_font(r, ascii_font, ea, base_size)


def shade(cell, hexcolor):
    tcpr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), hexcolor)
    tcpr.append(sh)


def add_page_number(par):
    for instr in ("PAGE",):
        run = par.add_run()
        f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr
        f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
        run._r.append(f1); run._r.append(it); run._r.append(f2)
        set_font(run, size=10)


def parse_table(lines):
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue
        rows.append(cells)
    return rows


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "2026B_solution", "2026B_论文初稿.md")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        ROOT, "2026B_solution", "2026B_论文初稿.docx")
    lines = open(src, encoding="utf-8").read().splitlines()

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.5)
    sec.left_margin, sec.right_margin = Cm(3.0), Cm(2.5)
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    # 页脚页码
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(fp)

    fig_no = 0
    eq_no = 0
    cur_prob = None
    fig_done = {}
    prob_dir = {v: k for k, v in PROB_NAME.items()}
    def insert_figs(prob_key):
        """按题号把该问的全部图件按序插入(以题号为界, 避免按文件名匹配漏图)。"""
        nonlocal fig_no
        pdir = prob_dir.get(prob_key)
        if pdir is None:
            return
        for (prob, base), cap in FIG_CAP.items():
            if prob != pdir or base in fig_done.get(prob, set()):
                continue
            png = os.path.join(ROOT, "origin_figures", prob, base + ".png")
            if not os.path.exists(png):
                continue
            fig_no += 1
            fig_done.setdefault(prob, set()).add(base)
            print(f"  [插图 {fig_no}] {prob}/{base}")
            ip = doc.add_paragraph(); ip.alignment = WD_ALIGN_PARAGRAPH.CENTER
            ip.paragraph_format.space_before = Pt(6)
            ip.add_run().add_picture(png, width=Cm(14.5))
            cp = doc.add_paragraph(); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = cp.add_run(f"图 {fig_no}　{PROB_NAME[prob]}：{cap}")
            set_font(r, size=10.5, ea_font="宋体")
            cp.paragraph_format.space_after = Pt(8)

    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        ln = raw.rstrip()
        s = ln.strip()

        # ---- 代码块 ----
        if s.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.5)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            for b in buf:
                r = p.add_run(b + "\n"); set_font(r, size=9.5, mono=True, ea="Consolas")
            continue

        # ---- 行间公式 ----
        if s.startswith("$$"):
            eq_no += 1
            body = s[2:]
            if body.endswith("$$"):
                body = body[:-2].strip(); i += 1
            else:
                i += 1
                buf = []
                while i < n and "$$" not in lines[i]:
                    buf.append(lines[i]); i += 1
                if i < n:
                    buf.append(lines[i].split("$$")[0]); i += 1
                body = " ".join(buf).strip()
            png, note_txt = render_display_eq(body, eq_no)
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(6)
            if png:
                p.add_run().add_picture(png, height=Cm(0.8) if len(body) < 60 else Cm(1.0))
                if note_txt:
                    r = p.add_run(note_txt); set_font(r, size=10.5, ea_font="宋体")
            else:
                add_runs(p, uni_math(body), 11.5)
            continue

        # ---- 表格 ----
        if s.startswith("|"):
            blk = []
            while i < n and lines[i].strip().startswith("|"):
                blk.append(lines[i]); i += 1
            rows = parse_table(blk)
            if not rows:
                continue
            ncol = max(len(r) for r in rows)
            tb = doc.add_table(rows=0, cols=ncol)
            tb.style = "Table Grid"
            tb.alignment = WD_TABLE_ALIGNMENT.CENTER
            for ri, r in enumerate(rows):
                cells = tb.add_row().cells
                for ci in range(ncol):
                    txt = r[ci] if ci < len(r) else ""
                    par = cells[ci].paragraphs[0]
                    par.paragraph_format.space_before = Pt(1)
                    par.paragraph_format.space_after = Pt(1)
                    par.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                    add_runs(par, txt, 9.5)
                    if ri == 0:
                        for rr in par.runs:
                            rr.font.bold = True
                        shade(cells[ci], "EAF1F8")
            doc.add_paragraph().paragraph_format.space_after = Pt(4)
            continue

        # ---- 标题 ----
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            lvl, txt = len(m.group(1)), m.group(2)
            mm = re.search(r"问题([一二三四])", txt)
            md = re.match(r"^(\d)\.", txt)
            CH = {"3": "问题一", "4": "问题二", "5": "问题三", "6": "问题四"}
            newp = CH.get(md.group(1)) if (md and md.group(1) in CH) else (
                f"问题{mm.group(1)}" if mm else None)
            if newp:
                if cur_prob is not None and cur_prob != newp:
                    insert_figs(cur_prob)        # 章节收尾兜底: 上一问未插入的图件补在此
                cur_prob = newp
            if "可视化" in txt:
                insert_figs(cur_prob)
            if lvl == 1:
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(txt); set_font(r, "Times New Roman", "黑体", 20, bold=True)
            else:
                p = doc.add_paragraph()
                size = {2: 15, 3: 13, 4: 12}.get(lvl, 12)
                r = p.add_run(txt)
                set_font(r, "Times New Roman", "黑体", size, bold=True)
                p.paragraph_format.space_before = Pt(10 if lvl == 2 else 8)
                p.paragraph_format.space_after = Pt(4)
            if "可视化" in txt:
                insert_figs(cur_prob)
            i += 1
            continue

        # ---- 分割线 ----
        if re.fullmatch(r"-{3,}", s):
            i += 1
            continue

        # ---- 引用(说明/教训段) ----
        if s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip()); i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(4)
            add_runs(p, " ".join(buf), 10.5, ea="楷体")
            for rr in p.runs:
                rr.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
            continue

        # ---- 列表 ----
        m = re.match(r"^[-*]\s+(.*)$", s)
        if m:
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(0)
            add_runs(p, m.group(1), 12); i += 1; continue
        m = re.match(r"^(\d+)\.\s+(.*)$", s)
        if m:
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(0)
            add_runs(p, m.group(2), 12); i += 1; continue

        # ---- 空行 ----
        if not s:
            i += 1
            continue

        # ---- 普通段落 ----
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0.74)      # 首行缩进 2 字符
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(2)
        add_runs(p, s, 12)
        # 段落是"可视化"段: 该题号的**全部**图件按序插入
        if "可视化" in s and cur_prob is not None:
            insert_figs(cur_prob)
            i += 1
            continue
        i += 1

    insert_figs(cur_prob)                 # 文末兜底: 最后一问的图件
    doc.save(dst)
    print(f"已生成 {dst}")
    print(f"  行间公式 {eq_no} 个(渲染于 {ASSETS}); 插图 {fig_no} 张")
    return dst


if __name__ == "__main__":
    main()
