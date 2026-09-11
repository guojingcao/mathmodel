import os
import re
import shutil

# 请确保此路径正确（建议使用绝对路径）
raw_root = "D:/My_MathModeling_Project/corpus/01_raw_pdfs/"

def find_year_and_topic_from_path(rel_path):
    """
    从相对路径（相对于 raw_root）中提取年份和赛题。
    返回 (year, topic)，year 为四位数字字符串或 "Unknown"，topic 为 A-F 或 "X"
    """
    parts = rel_path.split(os.sep)
    year = None
    topic = None

    # 从后往前遍历路径片段，优先匹配最近的年份和赛题文件夹
    for part in reversed(parts):
        if year is None:
            m = re.search(r'(\d{4})', part)
            if m:
                year = m.group(1)
                continue
        if topic is None:
            m = re.search(r'([A-F])\s*题?', part)  # 支持 A-F 和 "A题"
            if m:
                topic = m.group(1)
                # 不 break，继续向前找年份（可能年份在更前面）

    if year is None:
        year = "Unknown"
    if topic is None:
        # 如果确实没有赛题文件夹，标记为 X，但会继续处理
        topic = "X"

    return year, topic

def process_pdfs():
    if not os.path.exists(raw_root):
        print(f"错误：目录 {raw_root} 不存在，请检查路径。")
        return

    counter = {}
    total_moved = 0

    for root, dirs, files in os.walk(raw_root):
        # 只看当前目录下是否有 PDF 文件
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        if not pdf_files:
            continue

        # 计算相对于 raw_root 的路径
        rel_path = os.path.relpath(root, raw_root)
        if rel_path == '.':
            # 根目录本身有 PDF？不处理，避免混乱
            print("警告：根目录下存在 PDF，将忽略。")
            continue

        year, topic = find_year_and_topic_from_path(rel_path)
        if year == "Unknown":
            print(f"警告：无法从路径中提取年份，跳过文件夹: {root}")
            continue

        # 如果 topic 是 X，可以打印提示，但继续处理
        if topic == "X":
            print(f"提示：未找到赛题文件夹，将使用 'X' 作为赛题标记，文件夹: {root}")

        # 处理该文件夹下的所有 PDF
        for file in pdf_files:
            file_path = os.path.join(root, file)
            key = f"{year}_{topic}"
            counter[key] = counter.get(key, 0) + 1
            seq = str(counter[key]).zfill(3)  # 001, 002...
            new_name = f"{year}_{topic}_{seq}.pdf"
            new_path = os.path.join(raw_root, new_name)

            # 防重名（理论上不会，但安全起见）
            if os.path.exists(new_path):
                print(f"警告：目标文件 {new_path} 已存在，跳过 {file_path}")
                continue

            shutil.move(file_path, new_path)
            print(f"移动: {file_path} -> {new_path}")
            total_moved += 1

    print(f"\n✅ 全部处理完成！共移动了 {total_moved} 个文件。")
    print("📂 所有 PDF 现已集中在 raw_root 根目录，并已重命名。")
    print("🗑️  原文件夹（如 '2004年优秀论文'）现在应已为空，可以手动删除。")

if __name__ == "__main__":
    process_pdfs()