import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageOps

# --- 配置 ---
FONT_PATH = "Y145m-BGvd.ttf"
NORMAL_FONT_PATH = "arial.ttf"
OUTPUT_DIR = "cad_enhanced_dataset"
NUM_IMAGES = 200
CANVAS_SIZE = 1024

GDT_SYMBOLS = {
    "parallelism": "h", "perpendicularity": "o", "flatness": "i",
    "circularity": "c", "concentricity": "e", "cylindricity": "b",
    "phi": "d"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_overlapping(new_box, existing_boxes):
    for box in existing_boxes:
        if not (new_box[2] < box[0] or new_box[0] > box[2] or
                new_box[3] < box[1] or new_box[1] > box[3]):
            return True
    return False


def generate_enhanced_image(file_name):
    # 1. 创建底图并画背景干扰线
    img = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # 画背景干扰
    for _ in range(20):
        x1, y1 = random.randint(0, CANVAS_SIZE), random.randint(0, CANVAS_SIZE)
        x2, y2 = random.randint(0, CANVAS_SIZE), random.randint(0, CANVAS_SIZE)
        draw.line([x1, y1, x2, y2], fill=(235, 235, 235), width=1)

    existing_boxes = []
    items_drawn = 0
    attempts = 0

    while items_drawn < 15 and attempts < 150:
        attempts += 1

        # 随机参数：线宽、字号、旋转角度
        stroke_w = random.randint(1, 3)  # 线宽增强
        font_size = random.randint(35, 55)  # 尺寸增强
        angle = random.randint(-45, 45)  # 旋转增强（对平行/垂直度至关重要）

        gdt_font = ImageFont.truetype(FONT_PATH, font_size)
        num_font = ImageFont.truetype(NORMAL_FONT_PATH, int(font_size * 0.85))

        # 2. 创建一个临时小画布（RGBA支持透明，方便旋转）
        # 我们先在小画布上画标注，再旋转，最后贴到底图
        temp_w, temp_h = 400, 150
        temp_img = Image.new('RGBA', (temp_w, temp_h), (255, 255, 255, 0))
        temp_draw = ImageDraw.Draw(temp_img)

        item_type = random.choice(["linear", "angle", "gdt_frame"])

        if item_type == "gdt_frame":
            # --- 合成公差框 ---
            sym = GDT_SYMBOLS[random.choice(list(GDT_SYMBOLS.keys()))]
            tol = f"{random.uniform(0.01, 0.1):.3f}"
            datum = random.choice(["A", "B", "C", ""])
            w1, w2, w3 = 60, 120, 60
            box_w = w1 + w2 + (w3 if datum else 0)
            box_h = font_size + 20

            # 绘制白色遮罩底色
            temp_draw.rectangle([0, 0, box_w, box_h], fill=(255, 255, 255, 255))
            # 绘框
            temp_draw.rectangle([0, 0, box_w, box_h], outline="black", width=stroke_w)
            temp_draw.line([w1, 0, w1, box_h], fill="black", width=stroke_w)
            if datum:
                temp_draw.line([w1 + w2, 0, w1 + w2, box_h], fill="black", width=stroke_w)
            # 填字
            temp_draw.text((10, 5), sym, font=gdt_font, fill="black")
            temp_draw.text((w1 + 10, 15), tol, font=num_font, fill="black")
            if datum:
                temp_draw.text((w1 + w2 + 10, 15), datum, font=num_font, fill="black")

            actual_w, actual_h = box_w, box_h

        else:
            # --- 线性/角度尺寸 ---
            text = f"Φ {random.uniform(1, 500):.2f}" if item_type == "linear" else f"{random.uniform(1, 179):.1f}°"
            bbox = temp_draw.textbbox((0, 0), text, font=num_font)
            actual_w, actual_h = bbox[2] - bbox[0] + 10, bbox[3] - bbox[1] + 10
            # 遮罩与填字
            temp_draw.rectangle([0, 0, actual_w, actual_h], fill=(255, 255, 255, 255))
            temp_draw.text((5, 5), text, font=num_font, fill="black")

        # 3. 旋转小画布
        rotated_item = temp_img.crop((0, 0, actual_w + 2, actual_h + 2))  # 裁剪出有内容的部分
        rotated_item = rotated_item.rotate(angle, expand=True, resample=Image.BICUBIC)

        # 4. 碰撞检测与粘贴
        rw, rh = rotated_item.size
        x_pos = random.randint(0, CANVAS_SIZE - rw)
        y_pos = random.randint(0, CANVAS_SIZE - rh)

        new_bbox = [x_pos, y_pos, x_pos + rw, y_pos + rh]

        if not is_overlapping(new_bbox, existing_boxes):
            # 将旋转后的组件贴到底图
            img.paste(rotated_item, (x_pos, y_pos), rotated_item)
            existing_boxes.append(new_bbox)
            items_drawn += 1

    img.save(os.path.join(OUTPUT_DIR, file_name))


# 运行
for i in range(NUM_IMAGES):
    generate_enhanced_image(f"cad_aug_{i:03d}.jpg")
print(f"增强版数据集已生成在 {OUTPUT_DIR}")