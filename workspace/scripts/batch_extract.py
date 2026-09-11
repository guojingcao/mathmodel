import os
import fitz  # PyMuPDF

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(workspace_dir)

raw_dir = os.path.join(project_root, "corpus", "01_raw_pdfs")
txt_dir = os.path.join(project_root, "corpus", "03_analysis", "raw_texts")
os.makedirs(txt_dir, exist_ok=True)

print(f"读取目录: {raw_dir}")
print(f"输出目录: {txt_dir}")

pdf_files = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]
print(f"找到 {len(pdf_files)} 个PDF文件")

for idx, pdf in enumerate(pdf_files, 1):
    pdf_path = os.path.join(raw_dir, pdf)
    base_name = os.path.splitext(pdf)[0]
    txt_path = os.path.join(txt_dir, f"{base_name}.txt")

    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"=== 第{page_num}页 ===\n{text}")
        doc.close()

        if text_parts:
            full_text = "\n\n".join(text_parts)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"[{idx}/{len(pdf_files)}] ✓ {base_name}.txt (共{len(text_parts)}页)")
        else:
            print(f"[{idx}/{len(pdf_files)}] ⚠ {base_name} 提取为空（可能为扫描件）")
    except Exception as e:
        print(f"[{idx}/{len(pdf_files)}] ✗ {pdf} 出错: {e}")

print("\n全部完成！请检查 corpus/03_analysis/raw_texts/ 目录。")