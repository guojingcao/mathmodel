from pathlib import Path

from PIL import Image, ImageDraw


qa = Path(__file__).resolve().with_name("qa_render")
pages = sorted(qa.glob("page-*.png"))
for start in range(0, len(pages), 3):
    group = pages[start : start + 3]
    opened = [Image.open(path).convert("RGB") for path in group]
    thumb_w = 700
    resized = []
    for image in opened:
        h = round(image.height * thumb_w / image.width)
        resized.append(image.resize((thumb_w, h)))
    canvas_h = max(img.height for img in resized) + 50
    canvas = Image.new("RGB", (thumb_w * len(resized), canvas_h), "#c8c8c8")
    draw = ImageDraw.Draw(canvas)
    for idx, (img, path) in enumerate(zip(resized, group)):
        x = idx * thumb_w
        canvas.paste(img, (x, 36))
        draw.text((x + 10, 10), path.stem, fill="black")
    out = qa / f"montage-{start + 1:02d}-{start + len(group):02d}.png"
    canvas.save(out)
