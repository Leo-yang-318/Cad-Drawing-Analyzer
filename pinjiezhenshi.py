import os
import random
import glob
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

# --- 配置 ---
SNIPPET_DIR = "raw_data"  # 你的截图文件夹路径
OUTPUT_DIR = "real_composed_dataset"  # 输出文件夹
NUM_IMAGES = 100  # 想合成的大图数量
CANVAS_SIZE = 1024  # 大图分辨率
SNIPPETS_PER_IMAGE = 10  # 每张大图里放多少个截图

os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_overlapping(new_box, existing_boxes):
    for box in existing_boxes:
        if not (new_box[2] < box[0] or new_box[0] > box[2] or
                new_box[3] < box[1] or new_box[1] > box[3]):
            return True
    return False


def get_all_snippets(directory):
    # 支持 png, jpg
    return glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.jpg"))


def process_snippet(img_path):
    """对单个截图进行增强处理"""
    snippet = Image.open(img_path).convert("RGBA")

    # 1. 随机缩放 (0.8倍 到 1.5倍)
    scale = random.uniform(0.8, 1.5)
    nw, nh = int(snippet.width * scale), int(snippet.height * scale)
    snippet = snippet.resize((nw, nh), Image.Resampling.LANCZOS)

    # 2. 随机旋转 (-45度 到 45度)
    # CAD里很多截图是白底，旋转时要填充白色
    angle = random.randint(-45, 45)
    snippet = snippet.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255, 0))

    # 3. 随机亮度/对比度 (模拟不同显示器或扫描效果)
    enhancer = ImageEnhance.Contrast(snippet)
    snippet = enhancer.enhance(random.uniform(0.8, 1.2))

    return snippet


def compose_image(index, all_snippet_paths):
    # 创建纯白底图
    canvas = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 画一些背景干扰线 (模拟真实图纸的其他线条)
    for _ in range(15):
        x1, y1 = random.randint(0, CANVAS_SIZE), random.randint(0, CANVAS_SIZE)
        x2, y2 = random.randint(0, CANVAS_SIZE), random.randint(0, CANVAS_SIZE)
        draw.line([x1, y1, x2, y2], fill=(235, 235, 235), width=1)

    existing_boxes = []
    selected_paths = random.sample(all_snippet_paths, min(len(all_snippet_paths), SNIPPETS_PER_IMAGE))

    for p in selected_paths:
        snippet = process_snippet(p)
        sw, sh = snippet.size

        # 尝试寻找不重叠的位置
        for _ in range(50):
            x = random.randint(0, CANVAS_SIZE - sw)
            y = random.randint(0, CANVAS_SIZE - sh)
            new_box = [x, y, x + sw, y + sh]

            if not is_overlapping(new_box, existing_boxes):
                # 粘贴 (使用Alpha通道作为Mask，保留旋转后的透明边缘)
                canvas.paste(snippet, (x, y), snippet)
                existing_boxes.append(new_box)
                break

    canvas.save(os.path.join(OUTPUT_DIR, f"real_mix_{index:03d}.jpg"))


# 主程序
snippet_list = get_all_snippets(SNIPPET_DIR)
if not snippet_list:
    print("错误：在指定文件夹内没找到图片文件！")
else:
    for i in range(NUM_IMAGES):
        compose_image(i, snippet_list)
    print(f"完成！已合成 {NUM_IMAGES} 张真实感大图，请前往 {OUTPUT_DIR} 查看。")