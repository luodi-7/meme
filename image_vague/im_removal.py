import cv2
import os
# from imgutils.ocr import detect_text_with_ocr
from imgutils.ocr import ocr
import numpy as np
input_dir = "/mnt/afs/xueyingyi/image_vague/meme_in_russia"
output_dir = "/mnt/afs/xueyingyi/image_vague/image_origin_replace"
# 确保输出目录存在

os.makedirs(output_dir, exist_ok=True)

# 获取目录中所有图片文件
image_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]

# 筛选文件名，只保留不含 "_数字.jpg" 的文件
filtered_files = [f for f in image_files if "_0" not in f and "_1" not in f and "_2" not in f]

print(f"[INFO] Found {len(filtered_files)} images to process.")

# 遍历每张图片进行处理
for filename in filtered_files:
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    # 调用 OCR 检测函数
    # detections = detect_text_with_ocr(image_path)
    # detections = ocr(input_path, recognize_model='cyrillic_PP-OCRv3_rec')
    detections = ocr(input_path)
    # 加载原图像
    image = cv2.imread(input_path)

    # 如果读取失败，则跳过
    if image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        exit()

    # 复制原图以进行处理
    orig = image.copy()

    # 获取图像的高和宽
    H, W = image.shape[:2]

    # 使用高斯模糊生成模糊背景
    blurred_image = cv2.GaussianBlur(orig, (89, 89), 0)

    # 遍历OCR识别结果
    for detection in detections:
        # 解析检测框坐标和文本
        (startX, startY, endX, endY) = detection[0]
    
        # 从原图和模糊图像中获取对应区域
        orig[startY:endY, startX:endX] = blurred_image[startY:endY, startX:endX]

    
    # 保存处理后的图像
    cv2.imwrite(output_path, orig)

    print(f"[INFO] Saved processed image to {output_path}")
