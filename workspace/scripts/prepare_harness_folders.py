import os
import pandas as pd
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

csv_path = os.path.join(project_root, "corpus", "03_analysis", "sampled_papers.csv")
df = pd.read_csv(csv_path, encoding='utf-8-sig')

BATCH_SIZE = 10
total = len(df)
num_batches = math.ceil(total / BATCH_SIZE)

output_root = os.path.join(project_root, "corpus", "03_analysis", "deep_results")
os.makedirs(output_root, exist_ok=True)

for batch_idx in range(1, num_batches + 1):
    batch_folder = os.path.join(output_root, f"batch_{batch_idx:02d}")
    os.makedirs(batch_folder, exist_ok=True)

    start = (batch_idx - 1) * BATCH_SIZE
    end = min(batch_idx * BATCH_SIZE, total)
    batch_df = df.iloc[start:end]

    # 生成该批的文件列表
    list_path = os.path.join(batch_folder, "file_list.txt")
    with open(list_path, 'w', encoding='utf-8') as f:
        f.write(f"批次 {batch_idx}（共 {len(batch_df)} 篇）\n")
        for _, row in batch_df.iterrows():
            f.write(f"{row['file_name']}\n")

    print(f"创建 {batch_folder}，包含 {len(batch_df)} 篇")

print("所有批次文件夹创建完成。")