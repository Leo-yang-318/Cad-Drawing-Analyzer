import cv2
import numpy as np
import os
import random

# ================= 配置区域 =================
INPUT_DIR = "raw_data"  # 你存放那100张图的文件夹
OUTPUT_DIR = "augmented_dataset"  # 增强后的图片存放处
START_ID = 101  # 起始编号
# ===========================================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def add_gaussian_noise(image):
    """添加高斯噪声"""
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss * 20
    return np.clip(noisy, 0, 255).astype(np.uint8)


def adjust_brightness_contrast(image):
    """随机调节亮度和对比度"""
    brightness = random.uniform(0.7, 1.3)
    contrast = random.uniform(0.7, 1.3)
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


def add_blur(image):
    """随机模糊"""
    kernel_size = random.choice([3, 5])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def random_padding(image):
    """随机增加白色边框（模拟偏移）"""
    h, w = image.shape[:2]
    pad_top = random.randint(0, 10)
    pad_bot = random.randint(0, 10)
    pad_left = random.randint(0, 10)
    pad_right = random.randint(0, 10)
    # 使用白色填充
    return cv2.copyMakeBorder(image, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])


# 获取所有图片
files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
current_id = START_ID

print(f"开始处理，原始图片数量: {len(files)}")

for file_name in files:
    img_path = os.path.join(INPUT_DIR, file_name)
    img = cv2.imread(img_path)
    if img is None: continue

    # 每张原始图片生成 3 张增强后的图片
    for i in range(3):
        aug_img = img.copy()

        # 随机组合增强算法
        ops = [add_gaussian_noise, adjust_brightness_contrast, add_blur, random_padding]
        random.shuffle(ops)

        # 随机执行其中1-2种增强
        for op in ops[:random.randint(1, 2)]:
            aug_img = op(aug_img)

        # 统一缩放回合理尺寸（可选，如果你的切片大小差异太大）
        # aug_img = cv2.resize(aug_img, (width, height))

        # 保存图片
        save_name = f"{current_id}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), aug_img)

        print(f"生成图片: {save_name} (源自 {file_name})")
        current_id += 1

print(f"处理完成！最终数据集总数: {current_id - START_ID} 张，起始编号: {START_ID}")