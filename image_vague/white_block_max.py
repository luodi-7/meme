from imgutils.generic import YOLOModel
import cv2
import os
import numpy as np

# 输入和输出目录
input_dir = "/mnt/afs/xueyingyi/image_vague/image"
output_dir = "/mnt/afs/xueyingyi/image_vague/big_mask"  # 保存处理后的图片

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取目录中所有图片文件
image_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]

# 筛选文件名，只保留不含 "_数字.jpg" 的文件
filtered_files = [f for f in image_files if "_0" not in f and "_1" not in f and "_2" not in f]

print(f"[INFO] Found {len(filtered_files)} images to process.")

# 加载 YOLO 模型
model = YOLOModel('datacollection/anime_textblock_detection_test', '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')

# 遍历每张图片进行处理
for filename in filtered_files:
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)  # 处理后的图片保存路径

    # 调用 YOLO 模型进行文本检测
    detections = model.predict(input_path, 'v1_735k_s')
    print(detections)

    # 加载原图像
    image = cv2.imread(input_path)

    # 如果读取失败，则跳过
    if image is None:
        print(f"[ERROR] Failed to load image: {input_path}")
        continue

    # 复制原图以进行处理
    orig = image.copy()

    # 获取图像的高和宽
    H, W = image.shape[:2]

    # 用于记录大于 7900 的 detection 数量
    big_mask_count = 0

    # 遍历 OCR 识别结果
    for i, detection in enumerate(detections):
        # 解析检测框坐标和文本
        (startX, startY, endX, endY) = detection[0]

        # 计算当前检测框的面积
        area = (endX - startX) * (endY - startY)
        print(f"[INFO] Detection {i}: Mask area is {area} pixels")

        # 计算宽高比（横向长度 / 纵向长度）
        width = endX - startX
        height = endY - startY
        aspect_ratio = width / height if height != 0 else 0  # 避免除以零

        # 如果面积大于 7900 且宽高比小于 3，则在原图上将该区域替换为白色
        if area > 7900 and aspect_ratio < 3:
            big_mask_count += 1

            # 在原图上将该区域替换为白色
            orig[startY:endY, startX:endX] = 255  # 255 表示白色

            print(f"[INFO] Detection {i}: Replaced area with white in the image.")
        else:
            print(f"[INFO] Detection {i}: Skipped due to area {area} or aspect ratio {aspect_ratio}.")

    # 如果有大于 7900 的区域，则保存处理后的图片
    if big_mask_count > 0:
        base_name, ext = os.path.splitext(filename)
        if big_mask_count > 1:
            # 如果有多个大于 7900 的区域，添加后缀
            output_path = os.path.join(output_dir, f"{base_name}_{big_mask_count}{ext}")
        else:
            output_path = os.path.join(output_dir, f"{base_name}{ext}")

        cv2.imwrite(output_path, orig)
        print(f"[INFO] Saved processed image to {output_path}")
    else:
        print(f"[INFO] No large masks found in {filename}. Skipping.")