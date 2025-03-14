from imgutils.generic import YOLOModel
from PIL import Image
import cv2
import os
import json
import numpy as np
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch

input_dir = "/mnt/afs/xueyingyi/image_vague/image"
output_dir = "/mnt/afs/xueyingyi/image_vague/image_ocr_replace"
jsonl_file = "/mnt/afs/xueyingyi/image_vague/loc_dection/dections_Eimage.jsonl"  # JSONL 文件路径
error_log_file = "/mnt/afs/xueyingyi/image_vague/error_log_1.txt"  # 错误日志文件路径

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取目录中所有图片文件
image_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]

# 筛选文件名，只保留不含 "_数字.jpg" 的文件
filtered_files = [f for f in image_files if "_0" not in f and "_1" not in f and "_2" not in f]

print(f"[INFO] Found {len(filtered_files)} images to process.")
model = YOLOModel('datacollection/anime_textblock_detection_test', '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')

# 初始化 PaliGemma 模型
model_id = "google/paligemma-3b-mix-224"
device = "cuda:0"
dtype = torch.bfloat16

paligemma_model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16"
).eval()
processor = AutoProcessor.from_pretrained(model_id)

def recognize_text(image):
    # 将灰度图像转换为RGB格式
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Instruct the model to create a caption in Spanish
    prompt = "text reading"
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(paligemma_model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = paligemma_model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded

# 打开 JSONL 文件以追加写入
with open(jsonl_file, "w", encoding="utf-8") as f, open(error_log_file, "w", encoding="utf-8") as error_log:
    # 遍历每张图片进行处理
    for filename in filtered_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            # 调用 OCR 检测函数
            detections = model.predict(
                input_path, 
                model_name='y1_735k_bs512_s_yv11',  # 指定模型名称
                iou_threshold=0.35,                 # 设置 IOU 阈值
                conf_threshold=0.3                  # 设置置信度阈值
            )

            # 加载原图像
            image = cv2.imread(input_path)

            # 如果读取失败，则跳过
            if image is None:
                print(f"[ERROR] Failed to load image: {input_path}")
                error_log.write(f"Failed to load image: {input_path}\n")
                continue

            # 复制原图以进行处理
            orig = image.copy()

            # 获取图像的高和宽
            H, W = image.shape[:2]

            # 使用高斯模糊生成模糊背景
            blurred_image = cv2.GaussianBlur(orig, (89, 89), 0)

            # 遍历OCR识别结果
            detection_info = []  # 用于存储当前图片的检测信息
            for detection in detections:
                # 解析检测框坐标和文本
                (startX, startY, endX, endY) = detection[0]
            
                # 从原图和模糊图像中获取对应区域
                orig[startY:endY, startX:endX] = blurred_image[startY:endY, startX:endX]

                # 裁剪出检测框区域
                cropped_image = Image.fromarray(image[startY:endY, startX:endX])

                # 使用 PaliGemma 识别文字
                recognized_text = recognize_text(cropped_image)

                # 将检测框信息添加到 detection_info 中按照paligemma顺序[y_min, x_min, y_max, x_max]
                detection_info.append({
                    "bbox": [int(startY), int(startX), int(endY), int(endX)],
                    "text": recognized_text  # 使用 PaliGemma 识别的文字
                })

            # 保存处理后的图像
            cv2.imwrite(output_path, orig)

            # 将图片路径和检测信息写入 JSONL 文件
            record = {
                "image_path": input_path,
                "detections": detection_info
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"[INFO] Processed and saved image: {output_path}")

        except Exception as e:
            # 捕获异常并记录错误信息
            print(f"[ERROR] Failed to process image: {input_path}, Error: {str(e)}")
            error_log.write(f"Failed to process image: {input_path}, Error: {str(e)}\n")
            continue

print(f"[INFO] All detections saved to {jsonl_file}")
print(f"[INFO] Error log saved to {error_log_file}")