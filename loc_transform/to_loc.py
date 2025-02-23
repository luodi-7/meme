from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch
import re

# 模型和处理器初始化
model_id = "google/paligemma-3b-mix-224"
device = "cuda:0"
dtype = torch.bfloat16

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()
processor = AutoProcessor.from_pretrained(model_id)

# 加载图像
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

# 设置提示词（例如检测汽车）
prompt = "detect car"

# 准备模型输入
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

# 生成输出
with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=False)  # 注意：不要跳过特殊标记
    print("Model Output:", decoded)

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

# 解析并转换坐标
bboxes = parse_bbox_coordinates(decoded)
image_width, image_height = image.size
for bbox in bboxes:
    normalized_bbox = normalize_coordinates(bbox, image_width, image_height)
    print("Detected Bounding Box:", normalized_bbox)