import pandas as pd
import os
import numpy as np

# 路径设置
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
csv_path = os.path.join(project_root, "corpus", "03_analysis", "all_papers_keyinfo.csv")
output_path = os.path.join(project_root, "corpus", "03_analysis", "sampled_papers.csv")

# 读取数据
df = pd.read_csv(csv_path, encoding='utf-8-sig')
print(f"总论文数: {len(df)}")

# 确保关键词列存在且为字符串
df['keywords'] = df['keywords'].fillna('').astype(str)
# 确保 core_model 列为字符串（空值转空字符串）
df['core_model'] = df['core_model'].fillna('').astype(str)

def contains_keywords(text, kw_list):
    """检查文本是否包含关键词列表中的任意一个"""
    if not text:
        return False
    text_lower = text.lower()
    for kw in kw_list:
        if kw.lower() in text_lower:
            return True
    return False

# -------- 按赛题方法论抽样 --------
sampled = []

# 1. A题 - 机理分析类
a_keywords = ['微分方程', '最小二乘', '机理', '拟合', '回归']
a_pool = df[(df['topic'] == 'A') & (df['keywords'].apply(lambda x: contains_keywords(x, a_keywords)))]
if len(a_pool) >= 5:
    sampled.append(a_pool.sample(5, random_state=42))
else:
    sampled.append(a_pool)
print(f"A题机理类: {len(a_pool)}篇候选，抽取{len(sampled[-1])}篇")

# 2. B题 - 优化类
b_keywords = ['遗传算法', '整数规划', '模拟退火', '贪心', '启发式', '动态规划']
b_pool = df[(df['topic'] == 'B') & (df['keywords'].apply(lambda x: contains_keywords(x, b_keywords)))]
if len(b_pool) >= 8:
    sampled.append(b_pool.sample(8, random_state=42))
else:
    sampled.append(b_pool)
print(f"B题优化类: {len(b_pool)}篇候选，抽取{len(sampled[-1])}篇")

# 3. C题 - 预测类
c_keywords = ['神经网络', '随机森林', '支持向量机', '预测', '回归', 'LSTM', 'XGBoost']
c_pool = df[(df['topic'] == 'C') & (df['keywords'].apply(lambda x: contains_keywords(x, c_keywords)))]
if len(c_pool) >= 8:
    sampled.append(c_pool.sample(8, random_state=42))
else:
    sampled.append(c_pool)
print(f"C题预测类: {len(c_pool)}篇候选，抽取{len(sampled[-1])}篇")

# 4. D题 - 评价类
d_keywords = ['主成分分析', '聚类', '综合评价', '层次分析', 'TOPSIS', '因子分析']
d_pool = df[(df['topic'] == 'D') & (df['keywords'].apply(lambda x: contains_keywords(x, d_keywords)))]
if len(d_pool) >= 6:
    sampled.append(d_pool.sample(6, random_state=42))
else:
    sampled.append(d_pool)
print(f"D题评价类: {len(d_pool)}篇候选，抽取{len(sampled[-1])}篇")

# 5. E题 - 医学统计类
e_keywords = ['机器学习', '分类', '预测', '医学', '临床', '诊断']
e_pool = df[(df['topic'] == 'E') & (df['keywords'].apply(lambda x: contains_keywords(x, e_keywords)))]
if len(e_pool) >= 6:
    sampled.append(e_pool.sample(6, random_state=42))
else:
    sampled.append(e_pool)
print(f"E题医学类: {len(e_pool)}篇候选，抽取{len(sampled[-1])}篇")

# 6. F题 - 信号/图像类
f_keywords = ['深度学习', '卷积神经网络', '雷达', '信号', '图像', 'CNN']
f_pool = df[(df['topic'] == 'F') & (df['keywords'].apply(lambda x: contains_keywords(x, f_keywords)))]
if len(f_pool) >= 5:
    sampled.append(f_pool.sample(5, random_state=42))
else:
    sampled.append(f_pool)
print(f"F题信号类: {len(f_pool)}篇候选，抽取{len(sampled[-1])}篇")

# 7. 纵向 - 跨年份演进
years = [2004, 2008, 2012, 2016, 2020]
year_pool = df[df['year'].isin(years)]
if len(year_pool) >= 10:
    sampled.append(year_pool.sample(10, random_state=42))
else:
    sampled.append(year_pool)
print(f"跨年份演进: {len(year_pool)}篇候选，抽取{len(sampled[-1])}篇")

# 合并并去重
result_df = pd.concat(sampled).drop_duplicates(subset=['file_name'])
print(f"\n最终抽样: {len(result_df)} 篇")

# 保存结果
result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"已保存至: {output_path}")

# 打印抽样清单（修复空值问题）
print("\n" + "="*60)
print("抽样清单 (按赛题排序):")
print("="*60)
for topic in ['A','B','C','D','E','F']:
    subset = result_df[result_df['topic'] == topic]
    if len(subset) > 0:
        print(f"\n【{topic}题】共 {len(subset)} 篇:")
        for _, row in subset.iterrows():
            model_name = row['core_model']
            # 如果模型名称为空或长度不足50，显示完整名称，否则截断
            if pd.isna(model_name) or len(str(model_name)) == 0:
                display_name = "（未命名模型）"
            else:
                model_str = str(model_name)
                display_name = model_str[:50] + "..." if len(model_str) > 50 else model_str
            print(f"  {row['file_name']}: {display_name}")