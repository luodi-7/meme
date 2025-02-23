from imgutils.generic import YOLOModel
from PIL import Image
import cv2
import os
import numpy as np

# 输入和输出目录
input_dir = "/mnt/afs/xueyingyi/image_vague/image"  # 需要处理的图片目录
mask_demo_dir = "/mnt/afs/xueyingyi/image_vague/mask_demo"  # 已处理的掩码图片目录
mask_add_dir = "/mnt/afs/xueyingyi/image_vague/mask_add"  # 新处理的掩码图片保存目录

# 确保输出目录存在
os.makedirs(mask_add_dir, exist_ok=True)

# 获取目录中所有图片文件
image_files = set(f for f in os.listdir(input_dir) if f.endswith(".jpg"))
mask_demo_files = set(f for f in os.listdir(mask_demo_dir) if f.endswith(".jpg"))

# 筛选出需要处理的图片（在 input_dir 但不在 mask_demo_dir 中）
files_to_process = image_files - mask_demo_files

print(f"[INFO] Found {len(files_to_process)} images to process.")

# 加载 YOLO 模型
model = YOLOModel('datacollection/anime_textblock_detection_test', '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')

# 用于存储无法处理的文件名
failed_files = []

# 遍历每张图片进行处理
for filename in files_to_process:
    input_path = os.path.join(input_dir, filename)
    mask_path = os.path.join(mask_add_dir, filename)  # 新掩码图片路径

    try:
        # 尝试加载原图像
        image = cv2.imread(input_path)

        # 如果读取失败，则记录并跳过
        if image is None:
            print(f"[ERROR] Failed to load image: {input_path}")
            failed_files.append(filename)
            continue

        # 调用 YOLO 模型进行文本检测
        detections = model.predict(input_path, 'v1_735k_s')
        print(detections)

        # 复制原图以进行处理
        orig = image.copy()

        # 获取图像的高和宽
        H, W = image.shape[:2]

        # 使用高斯模糊生成模糊背景
        blurred_image = cv2.GaussianBlur(orig, (89, 89), 0)

        # 创建掩码图像（全黑，表示初始时不需要修复）
        mask = np.zeros((H, W), dtype=np.uint8)  # 单通道图像，初始值为 0（黑色）

        # 遍历 OCR 识别结果
        for detection in detections:
            # 解析检测框坐标和文本
            (startX, startY, endX, endY) = detection[0]

            # 在模糊图像中替换原图的对应区域
            orig[startY:endY, startX:endX] = blurred_image[startY:endY, startX:endX]

            # 在掩码图像中将对应区域设置为白色（255）
            mask[startY:endY, startX:endX] = 255

        # 保存掩码图像
        cv2.imwrite(mask_path, mask)

        print(f"[INFO] Saved mask image to {mask_path}")

    except Exception as e:
        # 捕获其他异常（例如文件损坏）
        print(f"[ERROR] Error processing image {filename}: {str(e)}")
        failed_files.append(filename)

# 打印无法处理的文件名
if failed_files:
    print("\n[INFO] The following files could not be processed:")
    for filename in failed_files:
        print(f"- {filename}")
else:
    print("\n[INFO] All files were processed successfully.")