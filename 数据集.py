import os
import shutil
import random

# --- 配置区域 ---
# 1. 之前真实截图拼成的100张图文件夹
SOURCE_REAL = "real_composed_dataset"
# 2. 之前仿真的200张增强图文件夹
SOURCE_SYNTH = "cad_enhanced_dataset"
# 3. 最终存放300张图的新文件夹
DEST_FOLDER = "final_dataset_for_labeling"

# 创建目标文件夹
if not os.path.exists(DEST_FOLDER):
    os.makedirs(DEST_FOLDER)


def collect_and_shuffle():
    # 获取两个文件夹下所有的图片路径
    real_files = [os.path.join(SOURCE_REAL, f) for f in os.listdir(SOURCE_REAL)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    synth_files = [os.path.join(SOURCE_SYNTH, f) for f in os.listdir(SOURCE_SYNTH)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    all_files = real_files + synth_files

    print(f"检测到真实图片: {len(real_files)} 张")
    print(f"检测到仿真图片: {len(synth_files)} 张")
    print(f"总计: {len(all_files)} 张")

    # 随机打乱顺序
    random.shuffle(all_files)

    # 开始复制并重命名
    for i, file_path in enumerate(all_files):
        # 获取原文件后缀名
        ext = os.path.splitext(file_path)[1]
        # 统一命名格式：cad_train_0001.jpg
        new_name = f"cad_train_{i + 1:04d}{ext}"
        dest_path = os.path.join(DEST_FOLDER, new_name)

        shutil.copy(file_path, dest_path)

        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1} 张...")

    print(f"\n恭喜！所有图片已整合至: {DEST_FOLDER}")
    print("现在你可以打开 LabelImg，选择该文件夹开始标注了。")


if __name__ == "__main__":
    collect_and_shuffle()