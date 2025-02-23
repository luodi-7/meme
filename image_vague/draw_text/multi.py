import json
import os
import random
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional

def is_color_close_to_black(color, threshold=0.5):  
    """  
    判断颜色是否接近黑色  

    Args:  
        color: 颜色，可以是颜色名称字符串，也可以是 RGB 元组  
        threshold: 亮度阈值，0 到 1 之间，值越小越接近黑色  

    Returns:  
        True 如果颜色接近黑色，否则 False  
    """  
    try:  
        # 将颜色转换为 RGB 元组  
        rgb = color 
    except ValueError:  
        print(f"Invalid color format: {color}")  
        return False  

    # 计算颜色的亮度 (Luma)  
    # 亮度计算公式: Y = 0.299 * R + 0.587 * G + 0.114 * B  
    luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]  

    # 将亮度值归一化到 0 到 1 之间  
    normalized_luma = luma / 255.0  

    # 如果亮度低于阈值，则认为颜色接近黑色  
    return normalized_luma < threshold

def draw_multiline_text(draw, position, text, font, max_width, fill, line_spacing=5):  
    """  
    在图像上绘制多行文本  

    Args:  
        draw: ImageDraw 对象  
        position: 文本起始位置 (x, y)  
        text: 要绘制的文本  
        font: 使用的字体  
        max_width: 最大行宽  
        fill: 字体颜色  
        line_spacing: 行间距  
    """  
    lines = []  
    words = text.split()  
    current_line = ""  

    for word in words:  
        # 检查添加下一个单词后行的宽度  
        test_line = f"{current_line} {word}".strip()  # 包含空格  
        if draw.textsize(test_line, font=font)[0] <= max_width:  
            current_line = test_line  
        else:  
            if current_line:  
                lines.append(current_line)  
            current_line = word  

    if current_line:  
        lines.append(current_line)  

    # 在图像上绘制每一行文字  
    y_offset = 0  
    for line in lines:  
        draw.text((position[0], position[1] + y_offset), line, font=font, fill=fill) 
        print(font.size)

        y_offset += font.getsize(line)[1] + line_spacing  # 获取行高并增加行间距

def draw_multiline_text_with_outline(draw, position, text, font, max_width, fill,  
                                     outline_color="black", outline_width=2, line_spacing=5,  
                                     alignment="center"):  # 默认居中 
    """  
    绘制带描边的多行文本，支持左对齐、右对齐和居中对齐。  
    """  
    lines = []  
    words = text.split()  
    current_line = ""  

    for word in words:  
        test_line = f"{current_line} {word}".strip()  
        if draw.textsize(test_line, font=font)[0] <= max_width:  
            current_line = test_line  
        else:  
            if current_line:  
                lines.append(current_line)  
            current_line = word  

    if current_line:  
        lines.append(current_line)  

    x, y = position  
    y_offset = 0  
    for line in lines:  
        line_width = draw.textsize(line, font=font)[0]  
        if alignment == "center":  
            x_offset = (max_width - line_width) / 2  
        elif alignment == "right":  
            x_offset = max_width - line_width  
        else:  # 默认或 "left"  
            x_offset = 0  

        x_position = x + x_offset  # 计算实际的 x 坐标  

        # 绘制描边  
        for dx, dy in [(0, -outline_width), (0, outline_width),  
                       (-outline_width, 0), (outline_width, 0),  
                       (-outline_width, -outline_width), (-outline_width, outline_width),  
                       (outline_width, -outline_width), (outline_width, outline_width)]:  
            draw.text((x_position + dx, y + y_offset + dy), line, font=font, fill=outline_color)  

        # 绘制文本  
        draw.text((x_position, y + y_offset), line, font=font, fill=fill)  
        y_offset += font.getsize(line)[1] + line_spacing  




def get_contrasting_color(color):  
    """  
    根据给定的背景颜色计算反色，并进一步增强与背景颜色的对比度。  
    Args:  
        color: RGB 元组，例如 (255, 255, 255)  
    Returns:  
        选择的颜色元组。  
    """  
    # 计算颜色的亮度（luminance）  
    def calculate_luminance(color):  
        r, g, b = color  
        r = r / 255.0  
        g = g / 255.0  
        b = b / 255.0  
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4  
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4  
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4  
        return 0.2126 * r + 0.7152 * g + 0.0722 * b  

    # 计算对比度  
    def calculate_contrast(color1, color2):  
        luminance1 = calculate_luminance(color1)  
        luminance2 = calculate_luminance(color2)  
        if luminance1 > luminance2:  
            return (luminance1 + 0.05) / (luminance2 + 0.05)  
        else:  
            return (luminance2 + 0.05) / (luminance1 + 0.05)  

    # 计算反色  
    inverted_color = tuple(255 - c for c in color[:3])  # 只处理 RGB  

    # 计算反色与背景颜色的对比度  
    contrast = calculate_contrast(color, inverted_color)  

    # 如果对比度不足，调整反色的亮度以增强对比度  
    min_contrast = 4.5  # WCAG 标准的最小对比度  
    if contrast < min_contrast:  
        background_luminance = calculate_luminance(color)  
        if background_luminance > 0.5:  # 背景较亮，使用黑色  
            inverted_color = (0, 0, 0)  
        else:  # 背景较暗，使用白色  
            inverted_color = (255, 255, 255)  

    return inverted_color  



def generate_image_with_text(
    image_path: str,  # 直接传入模糊后的图像路径
    detections: List[Tuple[int, int, int, int]],
    texts: List[str],
    output_dir: str,
    output_filename: str,  # 明确指定输出文件名
    font_type: str = "DejaVuSans.ttf",
    font_sizes: Optional[List[int]] = None,
    font_colors: Optional[List[Tuple[int, int, int]]] = None,
    alignments: Optional[List[str]] = None,
    bold: bool = False,
    italic: bool = False
) -> str:
    """修改后的函数（直接使用图像路径和输出文件名）"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模糊后的底图
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # 字体路径处理
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
        (startY, startX, endY, endX) = detection
        startX=(startX/1000)*image.width
        endX=(endX/1000)*image.width
        startY=(startY/1000)*image.height
        endY=(endY/1000)*image.height
        box_width = endX - startX  
        box_height = endY - startY  

        # 计算文本框的宽度和高度  
        box_width = endX - startX  
        box_height = endY - startY 
         # 检查并调整 x 坐标  
        if startX < 3:  
            startX = 3  
            if endX <= startX:  # 确保框存在  
                endX = startX + 3  
        elif endX > image.width - 3:  
            endX = image.width - 3  
            if startX >= endX:  # 确保框存在  
                startX = endX - 3  

        # 检查并调整 y 坐标  
        if startY < 3:  
            startY = 3  
            if endY <= startY:  # 确保框存在  
                endY = startY + 3  
        elif endY > image.height - 3:  
            endY = image.height - 3  
            if startY >= endY:  # 确保框存在  
                startY = endY - 3 

        # 动态调整字体大小（如果未指定字体大小）  
        if font_sizes[i] is None:  
            font_size = 1  # 初始字体大小  
            max_font_size = min(box_width, box_height) * 2  # 最大字体大小（基于文本框尺寸）  

            # 逐步增加字体大小，直到文本超出文本框或达到最大字体大小  
            while font_size < max_font_size:  
                font = ImageFont.truetype(font_path, font_size)  
                lines = []  
                current_line = ""  
                words = text.split()  
                
                for word in words:  
                    test_line = f"{current_line} {word}".strip()  
                    if draw.textsize(test_line, font=font)[0] <= box_width:  
                        current_line = test_line  
                    else:  
                        if current_line:  
                            lines.append(current_line)  
                        current_line = word  

                if current_line:  
                    lines.append(current_line)  

                # 计算文本的总高度和每行最大宽度
                text_width = max(draw.textsize(line, font=font)[0] for line in lines)
                text_height = sum(font.getsize(line)[1] for line in lines)

                if text_width > box_width or text_height > box_height:  
                    break

                font_size += 1  

            # 退回到最后一个合适的字体大小  
            font_size -= int(font_size/6)
        else:  
            font_size = font_sizes[i]  

        # 加载字体（支持加粗和斜体）  
        try:  
            if bold and italic:  
                font_path_variant = os.path.join("fonts", font_type.replace(".ttf", "-BoldItalic.ttf"))  
            elif bold:  
                font_path_variant = os.path.join("fonts", font_type.replace(".ttf", "-Bold.ttf"))  
            elif italic:  
                font_path_variant = os.path.join("fonts", font_type.replace(".ttf", "-Italic.ttf"))  
            else:  
                font_path_variant = font_path  

            font = ImageFont.truetype(font_path_variant, font_size)  
        except Exception as e:  
            print(f"加载字体失败: {e}")  
            font = ImageFont.load_default()  

        # 计算文本位置并绘制文本  
        if font_colors[i] is None:  
            # 获取文本框区域的平均颜色  
            box_region = image.crop((startX, startY, endX, endY))  
            average_color = box_region.resize((1, 1)).getpixel((0, 0))  
            # 获取与背景颜色对比的字体颜色  
            font_color = get_contrasting_color(average_color)  
        else:  
            font_color = font_colors[i] 
        if is_color_close_to_black(font_color):  
            outline_color = (255,255,255)  
        else:  
            outline_color = (0,0,0)


        # 重新计算文本并缩小字体直到适应文本框
        lines = []  
        current_line = ""  
        words = text.split()  
        for word in words:  
            test_line = f"{current_line} {word}".strip()  
            if draw.textsize(test_line, font=font)[0] <= box_width:  
                current_line = test_line  
            else:  
                if current_line:  
                    lines.append(current_line)  
                current_line = word  

        if current_line:  
            lines.append(current_line)  

        # 计算每行文本的最大宽度和总高度
        text_width = max(draw.textsize(line, font=font)[0] for line in lines)
        text_height = sum(font.getsize(line)[1] for line in lines)

        while text_width > box_width or text_height > box_height:
            font_size -= 1  # 缩小字体
            font = ImageFont.truetype(font_path_variant, font_size)
            lines = []  
            current_line = ""  
            for word in words:
                test_line = f"{current_line} {word}".strip()
                if draw.textsize(test_line, font=font)[0] <= box_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)

            text_width = max(draw.textsize(line, font=font)[0] for line in lines)
            text_height = sum(font.getsize(line)[1] for line in lines)
            if font.size <= 6:  
                raise ValueError(f"Font size is too small: {font.size}. Cannot process image：{output_filename}")

        # 在给定文本框内绘制多行文本  
        draw_multiline_text_with_outline(draw, (startX, startY), text, font, box_width, font_color,outline_color=outline_color,alignment=alignments[i])

    # 直接使用指定的输出文件名
    output_path = os.path.join(output_dir, output_filename)
    image.save(output_path)
    return output_path

def irrelevant_text():
    with open("/mnt/afs/niuyazhe/data/lister/meme/memetrash/memetext2.json", "r", encoding="utf-8") as f:
        memes = json.load(f)
    
    # 收集所有文本用于随机替换
    all_texts = [
        detection["text"]
        for meme in memes
        for detection in meme["detections"]
    ]
    
    irrelevant_meme = []
    for meme in memes:
        original_filename = os.path.basename(meme["image_path"])
        blurr_image_path = os.path.join("/mnt/afs/niuyazhe/data/lister/meme/memetrash/Eimages_blurr", original_filename)
        output_dir = "/mnt/afs/niuyazhe/data/lister/meme/memetrash/Eimages_irrelevant_new"
        
        # 准备参数
        converted_detections = []
        new_texts = []
        new_detections_json = []
        
        # 遍历每个检测框
        for detection in meme["detections"]:
            y1, x1, y2, x2 = detection["bbox"]
            converted_detections.append((y1, x1, y2, x2))  # 转换为 (startY, startX, endY, endX)
            
            # 随机选择新文本
            new_text = random.choice(all_texts)
            new_texts.append(new_text)
            new_detections_json.append({
                "bbox": detection["bbox"],  # 保持原始格式
                "text": new_text
            })
        
        # 调用修改后的函数
        output_path = generate_image_with_text(
            image_path=blurr_image_path,
            detections=converted_detections,
            texts=new_texts,
            output_dir=output_dir,
            output_filename=original_filename,  # 保持文件名一致
            font_type="DejaVuSans.ttf",
            bold=True
        )
        
        # 记录元数据
        irrelevant_meme.append({
            "original_image": meme["image_path"],
            "blurred_image": blurr_image_path,
            "new_image": output_path,
            "detections": new_detections_json
        })
    
    # 保存结果
    with open("/mnt/afs/niuyazhe/data/lister/meme/memetrash/irrelevantmeme.json", "w", encoding="utf-8") as f:
        json.dump(irrelevant_meme, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    irrelevant_text()