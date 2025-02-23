from imgutils.generic import YOLOModel
from PIL import Image
import cv2
import os
import numpy as np

# 输入和输出目录
input_dir = "/mnt/afs/niuyazhe/data/lister/meme/quickmeme_images"

mask_dir = "/mnt/afs/xueyingyi/image_vague/mask_quickmeme"  # 掩码图片保存目录

# 确保掩码目录存在

os.makedirs(mask_dir, exist_ok=True)

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
    mask_path = os.path.join(mask_dir, filename)  # 掩码图片路径

    try:
        # 调用 YOLO 模型进行文本检测
        detections = model.predict(
            input_path, 
            model_name='y1_735k_bs512_s_yv11',  # 指定模型名称
            iou_threshold=0.35,                 # 设置 IOU 阈值
            conf_threshold=0.3                  # 设置置信度阈值
        )
    except Exception as e:
        # 如果模型检测失败，打印错误信息并跳过当前图片
        print(f"[ERROR] Failed to process image {filename} with model: {e}")
        continue

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