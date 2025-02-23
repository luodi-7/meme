from imgutils.generic import YOLOModel
from PIL import Image
import cv2
import os
from imgutils.ocr import ocr
import numpy as np
import json

input_dir = "/mnt/afs/xueyingyi/image_vague/ocr_failure"
output_dir = "/mnt/afs/xueyingyi/image_vague/ocr_retest"
output_jsonl_path = "/mnt/afs/xueyingyi/image_vague/loc_dection/dections_new.jsonl"  # JSONL 文件路径

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取目录中所有图片文件
image_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]

# 筛选文件名，只保留不含 "_数字.jpg" 的文件
filtered_files = [f for f in image_files if "_0" not in f and "_1" not in f and "_2" not in f]

print(f"[INFO] Found {len(filtered_files)} images to process.")
model = YOLOModel('datacollection/anime_textblock_detection_test', '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')

# 打开 JSONL 文件准备写入
with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:
    # 遍历每张图片进行处理
    for filename in filtered_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

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

        # 第一次 YOLO 检测（较低阈值）
        first_detections = model.predict(
            input_path, 
            model_name='y1_735k_bs512_s_yv11',  # 指定模型名称
            iou_threshold=0.35,                 # 设置较低的 IOU 阈值
            conf_threshold=0.3                  # 设置较低的置信度阈值
        )

        # 存储最终的检测结果
        final_detections = []

        # 遍历第一次检测结果
        for detection in first_detections:
            # 解析检测框坐标、标签和置信度
            bbox, label, confidence = detection  # 假设 detection 是 ((x1, y1, x2, y2), 'text_block', confidence)
            (startX, startY, endX, endY) = bbox  # 提取检测框坐标

            # 提取第一次检测的区域
            region = orig[startY:endY, startX:endX]

            # 将区域保存为临时文件（因为 YOLO 模型可能需要文件路径）
            temp_path = os.path.join(output_dir, "temp_region.jpg")
            cv2.imwrite(temp_path, region)

            # 第二次 YOLO 检测（更高阈值）
            second_detections = model.predict(
                temp_path, 
                model_name='y1_735k_bs512_s_yv11',  # 指定模型名称
                iou_threshold=0.30,                  # 设置更高的 IOU 阈值
                conf_threshold=0.2                  # 设置更高的置信度阈值
            )

            # 遍历第二次检测结果
            for second_detection in second_detections:
                print(second_detections)
                # 解析第二次检测框坐标、标签和置信度
                second_bbox, second_label, second_confidence = second_detection
                (region_startX, region_startY, region_endX, region_endY) = second_bbox

                # 将区域坐标转换为原图坐标
                final_startX = startX + region_startX
                final_startY = startY + region_startY
                final_endX = startX + region_endX
                final_endY = startY + region_endY

                # 提取最终框的区域
                final_region = orig[final_startY:final_endY, final_startX:final_endX]

                # 将图像区域转换为 PIL 图像
                final_region_pil = Image.fromarray(cv2.cvtColor(final_region, cv2.COLOR_BGR2RGB))

                # 使用 OCR 提取文本
                ocr_result = ocr(final_region_pil)

                # 如果 OCR 识别到了文本，则保存结果
                if ocr_result:
                    # 解析 OCR 结果（假设 ocr_result 是 [((x1, y1, x2, y2), 'text', confidence), ...]）
                    for ocr_detection in ocr_result:
                        ocr_bbox, ocr_text, ocr_confidence = ocr_detection
                        final_detections.append({
                            "bbox": [final_startX, final_startY, final_endX, final_endY],
                            "text": ocr_text  # 使用 OCR 识别出的文本
                        })
                else:
                    final_detections.append({
                        "bbox": [final_startX, final_startY, final_endX, final_endY],
                        "text": ""  # 如果没有识别到文本，则留空
                    })

                # 将最终框的区域替换为模糊背景
                orig[final_startY:final_endY, final_startX:final_endX] = blurred_image[final_startY:final_endY, final_startX:final_endX]

            # 删除临时文件
            os.remove(temp_path)

        # 保存处理后的图像
        cv2.imwrite(output_path, orig)

        # 将结果格式化为 JSON 结构
        result = {
            "image_path": input_path,
            "detections": final_detections
        }

        # 将结果写入 JSONL 文件（每行一个 JSON 对象）
        jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"[INFO] Processed and saved results for {filename}")

print(f"[INFO] All results saved to {output_jsonl_path}")