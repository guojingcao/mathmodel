import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(workspace_dir)

extracted_root = os.path.join(project_root, "corpus", "02_extracted")
txt_output_dir = os.path.join(project_root, "corpus", "03_analysis", "raw_texts")
os.makedirs(txt_output_dir, exist_ok=True)

print(f"提取根目录: {extracted_root}")
print(f"文本输出目录: {txt_output_dir}")

if not os.path.exists(extracted_root):
    print(f"错误：{extracted_root} 不存在，请先运行 batch_extract.py")
    exit(1)

success_count = 0
for item in os.listdir(extracted_root):
    item_path = os.path.join(extracted_root, item)
    if not os.path.isdir(item_path):
        continue

    json_path = os.path.join(item_path, "content_list.json")
    if not os.path.exists(json_path):
        print(f"跳过 {item}：未找到 content_list.json")
        continue

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        text_parts = []
        for block in data:
            if isinstance(block, dict) and block.get("type") == "text":
                content = block.get("content", "")
                if content:
                    text_parts.append(content)
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])

        full_text = "\n\n".join(text_parts)
        txt_path = os.path.join(txt_output_dir, f"{item}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(full_text)

        print(f"已生成: {txt_path} (共 {len(text_parts)} 个文本块)")
        success_count += 1
    except Exception as e:
        print(f"处理 {item} 时出错: {e}")

print(f"\n解析完成！成功生成 {success_count} 个 txt 文件。")