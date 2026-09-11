import pandas as pd
import os
from collections import Counter

# 获取脚本所在目录，向上两级到项目根目录，再进入 corpus/03_analysis
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # workspace 的上一级是项目根目录
csv_path = os.path.join(project_root, "corpus", "03_analysis", "all_papers_keyinfo.csv")

print(f"正在读取: {csv_path}")

# 读取CSV
df = pd.read_csv(csv_path, encoding='utf-8-sig')

# -------- 1. 基本统计 --------
print("\n" + "="*50)
print("【基本统计】")
print("="*50)
print(f"总论文数: {len(df)}")
print(f"年份范围: {df['year'].min()} - {df['year'].max()}")
print("\n各赛题数量:")
print(df['topic'].value_counts().sort_index())

print("\n各年份数量(全部):")
year_counts = df['year'].value_counts().sort_index()
for year, count in year_counts.items():
    print(f"  {year}: {count}")

# -------- 2. 关键词词频统计 --------
print("\n" + "="*50)
print("【高频关键词 TOP30】")
print("="*50)

all_keywords = []
for kw_str in df['keywords'].dropna():
    # 尝试多种分隔符
    found = False
    for sep in ['；', ';', ',', '，', '、']:
        if sep in kw_str:
            all_keywords.extend([k.strip() for k in kw_str.split(sep) if k.strip()])
            found = True
            break
    if not found:
        all_keywords.append(kw_str.strip())

keyword_freq = Counter(all_keywords).most_common(30)
for i, (kw, cnt) in enumerate(keyword_freq, 1):
    print(f"  {i:2}. {kw}: {cnt}")

# -------- 3. 核心模型名称高频统计 --------
print("\n" + "="*50)
print("【高频模型名称 TOP25】")
print("="*50)

model_terms = []
for m in df['core_model'].dropna():
    m_str = str(m)
    # 尝试按常见分隔符拆分
    found = False
    for sep in ['、', '，', ',', '；', '/']:
        if sep in m_str:
            for part in m_str.split(sep):
                part = part.strip()
                if len(part) > 2:
                    model_terms.append(part)
            found = True
            break
    if not found and len(m_str) > 2:
        model_terms.append(m_str)

model_freq = Counter(model_terms).most_common(25)
for i, (m, cnt) in enumerate(model_freq, 1):
    print(f"  {i:2}. {m}: {cnt}")

# -------- 4. 按赛题统计关键词差异 --------
print("\n" + "="*50)
print("【各赛题高频关键词 TOP5】")
print("="*50)

for topic in ['A', 'B', 'C', 'D', 'E', 'F']:
    subset = df[df['topic'] == topic]
    if len(subset) == 0:
        continue
    kw_list = []
    for kw_str in subset['keywords'].dropna():
        for sep in ['；', ';', ',', '，', '、']:
            if sep in kw_str:
                kw_list.extend([k.strip() for k in kw_str.split(sep) if k.strip()])
                break
        else:
            kw_list.append(kw_str.strip())
    freq = Counter(kw_list).most_common(5)
    print(f"\n{topic}题 ({len(subset)}篇):")
    for kw, cnt in freq:
        print(f"    {kw}: {cnt}")

# -------- 5. 保存统计报告 --------
output_path = os.path.join(project_root, "corpus", "03_analysis", "stats_report.txt")
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("语料库统计分析报告\n")
    f.write("="*60 + "\n\n")
    f.write(f"总论文数: {len(df)}\n")
    f.write(f"年份范围: {df['year'].min()} - {df['year'].max()}\n")
    f.write("\n高频关键词 TOP30:\n")
    for kw, cnt in keyword_freq:
        f.write(f"  {kw}: {cnt}\n")
    f.write("\n高频模型名称 TOP25:\n")
    for m, cnt in model_freq:
        f.write(f"  {m}: {cnt}\n")

print("\n" + "="*50)
print(f"✅ 统计报告已保存至: {output_path}")
print("="*50)