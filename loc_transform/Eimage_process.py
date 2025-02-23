import json
from PIL import Image

# 将 bbox 坐标转换为 <loc> 标签格式
def bbox_to_loc(bbox, image_width, image_height):
    y_min, x_min, y_max, x_max = bbox
    
    # 将坐标归一化到 [0, 1024] 范围内
    y_min_norm = int((y_min / image_height) * 1024)
    x_min_norm = int((x_min / image_width) * 1024)
    y_max_norm = int((y_max / image_height) * 1024)
    x_max_norm = int((x_max / image_width) * 1024)
    
    # 格式化为 <loc> 标签
    loc_str = f"<loc{y_min_norm:04}><loc{x_min_norm:04}><loc{y_max_norm:04}><loc{x_max_norm:04}>"
    return loc_str

# 输入文件和输出文件路径
input_jsonl_path = "/mnt/afs/xueyingyi/image_vague/loc_dection/dections_NEW.jsonl"
output_jsonl_path = "/mnt/afs/xueyingyi/loc_transform/data_processed/transformed_data.jsonl"
empty_detections_jsonl_path = "/mnt/afs/xueyingyi/loc_transform/data_processed/empty_detections.jsonl"

# 打开输入文件和两个输出文件
with open(input_jsonl_path, "r") as input_file, \
     open(output_jsonl_path, "w") as output_file, \
     open(empty_detections_jsonl_path, "w") as empty_file:
    
    # 逐行读取输入文件
    for line in input_file:
        # 解析 JSON 数据
        data = json.loads(line.strip())
        
        # 加载图片以获取尺寸
        try:
            image = Image.open(data["image_path"])
            image_width, image_height = image.size
        except FileNotFoundError:
            print(f"Image not found: {data['image_path']}")
            continue

        # 如果 detections 为空，写入空数据文件
        if not data["detections"]:
            empty_file.write(json.dumps(data) + "\n")
            continue

        # 转换 detections 中的 bbox 为 <loc> 标签格式，并保留原始 bbox
        transformed_detections = []
        for detection in data["detections"]:
            bbox = detection["bbox"]
            loc_str = bbox_to_loc(bbox, image_width, image_height)
            
            # 保留原始 bbox 和转换后的 <loc> 标签
            transformed_detection = {
                "bbox": bbox,  # 原始 bbox
                "loc": loc_str,  # 转换后的 <loc> 标签
                "text": detection.get("text", "")  # 保留原始 text 字段
            }
            transformed_detections.append(transformed_detection)

        # 构建新的数据格式
        transformed_data = {
            "image_path": data["image_path"],
            "detections": transformed_detections
        }

        # 写入目标文件
        output_file.write(json.dumps(transformed_data, ensure_ascii=False) + "\n")

print(f"Transformed data saved to: {output_jsonl_path}")
print(f"Empty detections data saved to: {empty_detections_jsonl_path}")