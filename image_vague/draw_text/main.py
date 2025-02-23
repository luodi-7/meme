from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import os
import uuid

app = FastAPI()

# 假设字体文件存储在某个目录下
FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
# 假设生成的图片存储在某个目录下
IMAGE_DIR = "/mnt/afs/xueyingyi/image_vague/demo/"

class ImageRequest(BaseModel):
    uid: str
    base_image: int
    font_type: str
    detections: List[Tuple[int, int, int, int]]
    texts: List[str]

@app.post("/deal_with_post_base_image")
async def deal_with_post_base_image(request: ImageRequest):
    uid = request.uid
    base_image = request.base_image
    font_type = request.font_type
    detections = request.detections
    texts = request.texts

    # 检查uid是否有效（这里假设uid是有效的）
    if not uid:
        raise HTTPException(status_code=400, detail="Invalid uid")

    # 加载底图（假设底图路径是根据base_image生成的）
    image_path = f"{IMAGE_DIR}image_{base_image}.jpg"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Base image not found")

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 加载字体
    font_path = os.path.join(FONT_DIR, font_type)
    if not os.path.exists(font_path):
        raise HTTPException(status_code=404, detail="Font not found")

    # 遍历检测框和文本
    for detection, text in zip(detections, texts):
        (startX, startY, endX, endY) = detection

        # 计算文本框的宽度和高度
        box_width = endX - startX
        box_height = endY - startY

        # 动态调整字体大小以适应文本框
        font_size = 1  # 初始字体大小
        max_font_size = 1000  # 最大字体大小

        # 计算文本的边界框
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 逐步增加字体大小，直到文本超出文本框或达到最大字体大小
        while text_width < box_width and text_height < box_height and font_size < max_font_size:
            font_size += 1
            font = ImageFont.truetype(font_path, font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

        # 如果字体大小超过最大值，回退到最大值
        if font_size >= max_font_size:
            font_size = max_font_size
            font = ImageFont.truetype(font_path, font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            # 回退到上一个合适的字体大小
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

        # 计算文本位置（居中）
        text_x = startX + (box_width - text_width) // 2
        text_y = startY + (box_height - text_height) // 2

        # 设置字体颜色（与周围区分开）
        # 获取文本框区域的平均颜色
        box_region = image.crop((startX, startY, endX, endY))
        average_color = box_region.resize((1, 1)).getpixel((0, 0))

        # 计算反色
        inverse_color = tuple(255 - c for c in average_color[:3])  # 只处理 RGB，忽略 Alpha 通道

        # 在图片上绘制文本
        draw.text((text_x, text_y), text, font=font, fill=inverse_color)

    # 生成唯一的文件名
    output_filename = f"output_image_{uuid.uuid4()}.jpg"
    output_path = os.path.join(IMAGE_DIR, output_filename)
    image.save(output_path)

    # 生成图片的URL（假设图片可以通过某个URL访问）
    image_url = f"http://yourdomain.com/images/{output_filename}"

    return {
        "code": 0,
        "ret": True,
        "error_msg": "",
        "image_url": image_url
    }

#测试方法：
#但是好像在sensorcore跑不通
#启动端口：
# uvicorn main:app --reload --port 8001
#发送请求：
# curl -X POST "https://vscode-a9cca469-ea5f-4e0e-a403-0eb4414d66d2.aicl-proxy.cn-sh-01.sensecore.cn:33080/proxy/8001/deal_with_post_base_image" \
# -H "Content-Type: application/json" \
# -d '{
#   "uid": "12345",
#   "base_image": 1,
#   "font_type": "DejaVuSans.ttf",
#   "detections": [[111, 9, 508, 68], [120, 258, 503, 333], [549, 331, 625, 343]],
#   "texts": ["ANIMATED MEMES", "WAIT FOR IT...", ""]
# }'