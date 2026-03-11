import os
from PIL import Image, ImageDraw, ImageFont

# 1. 配置路径（确保文件名和路径正确）
font_path = "Y145m-BGvd.ttf"
output_map_name = "symbol_map.png"


def create_font_map():
    if not os.path.exists(font_path):
        print(f"错误：找不到字体文件 {font_path}")
        return

    # 创建大画布
    img = Image.new('RGB', (1200, 600), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 加载字体
    try:
        gdt_font = ImageFont.truetype(font_path, 40)
        default_font = ImageFont.load_default()  # 用于标出原始字母
    except Exception as e:
        print(f"字体加载失败: {e}")
        return

    # 遍历所有常见的键盘字符
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;':\",.<>/?"

    for i, char in enumerate(chars):
        row = i // 12
        col = i % 12
        x, y = col * 100 + 20, row * 80 + 20

        # 画出原始键盘字母（蓝色）
        draw.text((x, y), char, fill="blue")
        # 画出对应的 CAD 符号（黑色）
        draw.text((x, y + 30), char, font=gdt_font, fill="black")

    img.save(output_map_name)
    print(f"对照表已生成: {output_map_name}。请打开查看符号对应的字母。")


create_font_map()