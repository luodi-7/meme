from PIL import Image, ImageDraw, ImageFont
import os
import uuid
from typing import List, Tuple, Optional

def generate_image_with_text(
    uid: str,
    base_image: int,
    font_type: str,
    detections: List[Tuple[int, int, int, int]],
    texts: List[str],
    output_dir: str = "/mnt/afs/xueyingyi/image_vague/draw_text/ouput",
    font_sizes: Optional[List[int]] = None,
    font_colors: Optional[List[Tuple[int, int, int]]] = None,
    alignments: Optional[List[str]] = None,
    bold: bool = False,
    italic: bool = False,
):
    """
    在底图上添加文本并保存生成的图片。支持用户自定义字体大小、颜色、对齐方式等。

    Args:
        uid (str): 用户 ID。
        base_image (int): 底图 ID。
        font_type (str): 字体文件名（例如 "DejaVuSans.ttf"）。
        detections (list): 文本框的位置信息，格式为 [(startX, startY, endX, endY), ...]。
        texts (list): 每个文本框对应的文本。
        output_dir (str): 保存生成图片的目录。
        font_sizes (list): 每个文本框的字体大小。如果为 None，则动态调整字体大小。
        font_colors (list): 每个文本框的字体颜色，格式为 [(R, G, B), ...]。如果为 None，则使用反色。
        alignments (list): 每个文本框的对齐方式（"left", "center", "right"）。如果为 None，则默认居中。
        bold (bool): 是否加粗字体。
        italic (bool): 是否斜体字体。

    Returns:
        str: 生成的图片路径。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载底图（假设底图路径是根据 base_image 生成的）
    image_path = f"/mnt/afs/xueyingyi/image_vague/draw_text/demo/image_ ({base_image}).jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Base image not found: {image_path}")

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 加载字体（假设字体文件在 fonts 目录下）
    font_path = os.path.join("/usr/share/fonts/truetype/dejavu", font_type)
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font not found: {font_path}")

    # 初始化默认值
    if font_sizes is None:
        font_sizes = [None] * len(texts)  # 动态调整字体大小
    if font_colors is None:
        font_colors = [None] * len(texts)  # 使用反色
    if alignments is None:
        alignments = ["center"] * len(texts)  # 默认居中

    # 遍历检测框和文本
    for i, (detection, text) in enumerate(zip(detections, texts)):
        (startX, startY, endX, endY) = detection

        # 计算文本框的宽度和高度
        box_width = endX - startX
        box_height = endY - startY

        # 动态调整字体大小（如果未指定字体大小）
        if font_sizes[i] is None:
            font_size = 1  # 初始字体大小
            max_font_size = 1000  # 最大字体大小

            # 计算文本的边界框
            font = ImageFont.truetype(font_path, font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # 逐步增加字体大小，直到文本超出文本框或达到最大字体大小
            while text_width < box_width*0.9 and text_height < box_height*0.9 and font_size < max_font_size:
                font_size += 1
                font = ImageFont.truetype(font_path, font_size)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

            # 如果字体大小超过最大值，回退到最大值
            if font_size >= max_font_size:
                font_size = max_font_size
            else:
                # 回退到上一个合适的字体大小
                font_size -= 1
        else:
            font_size = font_sizes[i]

        # 加载字体（支持加粗和斜体）
        try:
            # 根据 bold 和 italic 参数选择字体文件
            if bold and italic:
                font_path = os.path.join("fonts", font_type.replace(".ttf", "-BoldItalic.ttf"))
            elif bold:
                font_path = os.path.join("fonts", font_type.replace(".ttf", "-Bold.ttf"))
            elif italic:
                font_path = os.path.join("fonts", font_type.replace(".ttf", "-Italic.ttf"))
            else:
                font_path = os.path.join("fonts", font_type)

            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"加载字体失败: {e}")
            font = ImageFont.load_default()

        # 计算文本位置
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        if alignments[i] == "left":
            text_x = startX
        elif alignments[i] == "right":
            text_x = endX - text_width
        else:  # 默认居中
            text_x = startX + (box_width - text_width) // 2

        text_y = startY + (box_height - text_height) // 2

        # 设置字体颜色
        if font_colors[i] is None:
            # 获取文本框区域的平均颜色
            box_region = image.crop((startX, startY, endX, endY))
            average_color = box_region.resize((1, 1)).getpixel((0, 0))
            # 计算反色
            font_color = tuple(255 - c for c in average_color[:3])  # 只处理 RGB，忽略 Alpha 通道
        else:
            font_color = font_colors[i]

        # 在图片上绘制文本
        draw.text((text_x, text_y), text, font=font, fill=font_color)

    # 生成唯一的文件名
    output_filename = f"/mnt/afs/xueyingyi/image_vague/draw_text/ouput/output_image_{uid}_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    image.save(output_path)

    print(f"图片已保存到: {output_path}")
    return output_path


# 示例调用
if __name__ == "__main__":
    # 输入参数
    uid = "12345"
    base_image = 0
    font_type = "DejaVuSans.ttf"
    detections = [(0, 1, 496, 151), (138, 417, 373, 470), [407, 481, 499, 499]]
    texts = ["that moment after you throw up and your friend asks you \"you good bro?\"","i'm fuckin lit","irunny.co"]

    # 自定义参数（可选）
    font_sizes = None  # 每个文本框的字体大小
    font_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 每个文本框的字体颜色
    alignments = ["left", "center", "right"]  # 每个文本框的对齐方式
    bold = True  # 是否加粗
    italic = False  # 是否斜体

    # 调用函数生成图片
    try:
        output_path = generate_image_with_text(
            uid, base_image, font_type, detections, texts,
            font_sizes=font_sizes, font_colors=font_colors, alignments=alignments,
            bold=bold, italic=italic
        )
        print(f"生成的图片路径: {output_path}")
    except Exception as e:
        print(f"生成图片时出错: {e}")