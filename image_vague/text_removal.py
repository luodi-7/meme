import os
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
import time

# 指定输入和输出目录
# input_dir = "/mnt/afs/niuyazhe/data/meme/data/Eimages/Eimages/Eimages"
# output_dir = "/mnt/afs/xueyingyi/image_vague/image_after"
input_dir = "/mnt/afs/xueyingyi/image_vague/image"
output_dir = "/mnt/afs/xueyingyi/image_vague/image_test"
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

    # 加载输入图像
    image = cv2.imread(input_path)

    # 如果读取失败，则跳过该文件
    if image is None:
        print(f"[ERROR] Failed to load image: {input_path}")
        continue  # 跳过当前文件，继续下一个
    
    orig = image.copy()
    (H, W) = image.shape[:2]

    # 设置新的宽高
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # 加载模糊背景
    blurred_image = cv2.GaussianBlur(orig, (89, 89), 0)

    # 定义 EAST 模型的输出层名称
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    # 加载 EAST 模型
    print(f"[INFO] Processing {filename}...")
    net = cv2.dnn.readNet("/mnt/afs/xueyingyi/image_vague/frozen_east_text_detection.pb")

    # 构建图像 Blob
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # 提取分数和几何信息
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # 遍历每一行
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.09:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # 非极大值抑制
    boxes = non_max_suppression(np.array(rects), probs=confidences)


    # 设置扩展比例
    expand_ratio = 0.30  # 扩展30%

    # 遍历每一个bounding box
    for (startX, startY, endX, endY) in boxes:
        
        
        
        
        startX = max(0, int(startX * rW))
        startY = max(0, int(startY * rH))
        endX = min(orig.shape[1], int(endX * rW))
        endY = min(orig.shape[0], int(endY * rH))
        # 计算纵向和横向扩展的范围
        width = endX - startX
        height = endY - startY
    
        # 横向扩展
        new_width = width * (1 + 2 * expand_ratio)  # 扩展10%向左和向右
        offsetX = (new_width - width) // 2  # 计算左侧和右侧的扩展量
    
        # 纵向扩展
        new_height = height * (1 + 2 * expand_ratio)  # 扩展10%向上和向下
        offsetY = (new_height - height) // 2  # 计算上方和下方的扩展量

        # 重新计算边框位置
        startX = max(0, int(startX - offsetX))  # 向左扩展
        startY = max(0, int(startY - offsetY))  # 向上扩展
        endX = min(orig.shape[1], int(endX + offsetX))  # 向右扩展
        endY = min(orig.shape[0], int(endY + offsetY))  # 向下扩展

        # 替换图像中的文字区域
        orig[startY:endY, startX:endX] = blurred_image[startY:endY, startX:endX]
    # 保存结果
    cv2.imwrite(output_path, orig)
    print(f"[INFO] Saved processed image to {output_path}")


    # # 遍历检测框并处理图像
    # for (startX, startY, endX, endY) in boxes:
    #     startX = max(0, int(startX * rW))
    #     startY = max(0, int(startY * rH))
    #     endX = min(orig.shape[1], int(endX * rW))
    #     endY = min(orig.shape[0], int(endY * rH))
    #     orig[startY:endY, startX:endX] = blurred_image[startY:endY, startX:endX]

    # # 保存结果
    # cv2.imwrite(output_path, orig)
    # print(f"[INFO] Saved processed image to {output_path}")
