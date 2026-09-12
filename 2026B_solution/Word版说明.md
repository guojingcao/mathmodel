# Word 投稿版说明（2026B_论文初稿.docx）

## 文件

| 文件 | 说明 |
|---|---|
| `2026B_solution/2026B_论文初稿.docx` | **投稿用 Word 文档**（A4，正文宋体·Times New Roman 小四，1.5 倍行距，页脚页码） |
| `2026B_solution/word_assets/eq_*.png` | 12 个行间公式的 300 dpi 透明底图片（文档内嵌，此目录为可再生副本） |
| `md2docx.py` | 转换脚本，可重复运行：`python md2docx.py` |
| `eq_probe.py` | 公式渲染自检（逐条报告 mathtext 能否渲染，便于定位记号问题） |

## 转换规则

| Markdown | Word 中的呈现 |
|---|---|
| `# / ## / ###` | 标题（黑体；一级 20 pt、二级 15 pt、三级 13 pt） |
| 正文段落 | 宋体·Times New Roman 12 pt，首行缩进 2 字符，1.5 倍行距 |
| `**粗体**` | 加粗 |
| `` `代码` `` | Consolas 等宽字体 |
| `$行内公式$` | **Unicode 转写**（如 `\Omega`→Ω、`\le`→≤、`_i`→ᵢ），避免 Word 行内图片基线错位 |
| `$$\n行间公式$$` | **matplotlib mathtext 渲染为 300 dpi 透明 PNG**（居中插入）；公式内的中文（如 `\text{真实源必可被清除}`）自动抽出为图片后的正文括注 |
| 表格 | 真 Word 表格（表头加粗+浅蓝底纹，全边框，9.5 pt） |
| `> 引用` | 楷体 10.5 pt、左缩进（用于"修正说明/教训"等旁注） |
| ```` ``` ```` 代码块 | Consolas 9.5 pt，左缩进 |
| `origin_figures/**` 图件 | 按题号在"可视化"处**自动插入并编号**（图 1–图 14），图题居中 |

## 已知限制（投稿前请过目）

1. **行间公式是图片而非 Word 原生公式**：本机无 pandoc；mathtext 也不支持中文与部分命令，故采用"高分辨率图片 + 中文括注"的方案。若需**可编辑的原生公式**，两条路：① 装 pandoc 后 `pandoc 初稿.md -o 初稿.docx --mathml`（本机已确认 Office 的 `MML2OMML.XSL` 存在）；② 在 Word 里对着图片用公式编辑器重录（仅 12 条）。
2. 转换时对 LaTeX 做了**兼容映射**（`\le`→`\leq`、`\ge`→`\geq`、`\bigcap`→`\cap`、`\tfrac`→`\frac`、`\overline X`→`\overline{X}` 等），因为 mathtext 只支持 LaTeX 子集；若正文新增公式，请先跑 `python eq_probe.py` 确认能渲染。
3. 极长公式在 300 dpi 下会触发 matplotlib 图像尺寸上限，脚本会自动降到 200/150/110 dpi（清晰度仍满足打印）。
4. 表内文字为 9.5 pt；若排版要求更严，可在 Word 中统一调整表格样式。
5. **数据口径**：本文档内容与 `2026B_论文初稿.md` 完全一致——最终配置为 **27 点最小覆盖设计**、计时为**题设 5 s 口径**（逐局与逐动作残差为零）；历史版本（29/31 点）仅作追溯，不作为结论。

## 重新生成

```powershell
python md2docx.py                    # 读 2026B_solution/2026B_论文初稿.md, 覆盖生成 .docx
python eq_probe.py                   # 检查行间公式能否全部渲染
```
