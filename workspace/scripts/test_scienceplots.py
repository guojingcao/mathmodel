import matplotlib.pyplot as plt
import scienceplots

# ===== 1. 设置 SciencePlots 风格，但强制不使用 LaTeX =====
plt.style.use('science')
# 关键：关闭 LaTeX，改用 Matplotlib 自己的文本渲染器
plt.rcParams['text.usetex'] = False

# ===== 2. 指定中文字体（保证中文正常显示） =====
# 优先使用系统自带的微软雅黑或黑体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ===== 3. 其他常规设置 =====
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ===== 数据 =====
categories = ['A', 'B', 'C', 'D', 'E', 'F']
values = [128, 148, 115, 138, 75, 60]
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']

# ===== 绘图 =====
fig, ax = plt.subplots(figsize=(89/25.4, 89/25.4 * 0.75))
bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=0.8)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('赛题', fontsize=11, fontweight='bold')
ax.set_ylabel('论文数量（篇）', fontsize=11, fontweight='bold')
ax.set_ylim(0, 170)
ax.set_yticks(range(0, 171, 20))
ax.tick_params(direction='in', length=4, width=1.2)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

ax.set_title('图 1  2004–2023年各赛题优秀论文数量分布', fontsize=11, fontweight='bold', pad=12)

# ===== 保存 =====
plt.savefig('bar_chart_journal.png', dpi=300, bbox_inches='tight')
print("✅ 图片已保存为 bar_chart_journal.png（300 dpi）")