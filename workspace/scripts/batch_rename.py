import os
import re

raw_dir = "D:/My_MathModeling_Project/corpus/01_raw_pdfs/"

# 获取所有PDF文件
files = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]

print("即将重命名的文件预览：")
for fname in files:
    # 这里只是一个示例规则，实际使用时根据你的文件名规律修改
    # 假设原始文件名类似 "2023全国赛A题一等奖.pdf"
    new_name = fname.replace("全国赛", "CUMCM").replace("美赛", "MCM")
    # 更复杂的替换可写正则，例如提取年份、赛题等
    print(f"{fname} -> {new_name}")

# 确认后取消下面注释执行重命名
# for fname in files:
#     old_path = os.path.join(raw_dir, fname)
#     new_name = fname.replace("全国赛", "CUMCM").replace("美赛", "MCM")
#     new_path = os.path.join(raw_dir, new_name)
#     os.rename(old_path, new_path)
#     print(f"已重命名: {fname}")