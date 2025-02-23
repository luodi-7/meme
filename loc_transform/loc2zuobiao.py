from PIL import Image, ImageDraw
import re

# 解析输出中的坐标
def parse_bbox_coordinates(output):
    # 正则表达式匹配 <loc[value]> 格式的坐标
    loc_pattern = r"<loc(\d+)>"
    matches = re.findall(loc_pattern, output)
    
    # 将匹配到的值转换为整数
    coords = [int(m) for m in matches]
    
    # 每 4 个值表示一个边界框 (y_min, x_min, y_max, x_max)
    bboxes = []
    for i in range(0, len(coords), 4):
        if i + 4 <= len(coords):
            bbox = coords[i:i+4]
            bboxes.append(bbox)
    
    return bboxes

# 将归一化坐标转换为实际坐标
def normalize_coordinates(bbox, image_width, image_height):
    y_min, x_min, y_max, x_max = bbox
    y_min = (y_min / 1024) * image_height
    x_min = (x_min / 1024) * image_width
    y_max = (y_max / 1024) * image_height
    x_max = (x_max / 1024) * image_width
    return [y_min, x_min, y_max, x_max]

# 示例 decoded 结果
decoded = '<loc0104><loc0057><loc0843><loc0976> text loaction'

# 加载图片并转换为 RGB 模式
image_path = "/mnt/afs/xueyingyi/image_vague/ocr_failure/image_ (2437).jpg"
image = Image.open(image_path).convert("RGB")  # 转换为 RGB 模式
image_width, image_height = image.size

# 解析并转换坐标
bboxes = parse_bbox_coordinates(decoded)
for bbox in bboxes:
    normalized_bbox = normalize_coordinates(bbox, image_width, image_height)
    print("Detected Bounding Box:", normalized_bbox)

    # 在图片上绘制边界框
    draw = ImageDraw.Draw(image)
    y_min, x_min, y_max, x_max = normalized_bbox
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

# 保存或显示图片
output_image_path = "/mnt/afs/xueyingyi/image_vague/ocr_failure/image_ (2437)_with_bbox.jpg"
image.save(output_image_path)
print(f"Image with bounding box saved to: {output_image_path}")

# 如果需要显示图片
image.show()