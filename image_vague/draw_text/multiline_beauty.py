from PIL import Image, ImageDraw, ImageFont,ImageColor  
import os  
import uuid  
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
    uid: str,  
    base_image: str,  
    font_type: str,  
    detections: List[Tuple[int, int, int, int]],  
    texts: List[str],  
    output_dir: str = "/mnt/afs/xueyingyi/image_vague/draw_text/output",  
    font_sizes: Optional[List[int]] = None,  
    font_colors: Optional[List[Tuple[int, int, int]]] = None,
    outline_colors: Optional[List[Tuple[int, int, int]]] = None,  
    outline_width: Optional[int] = 2,
    alignments: Optional[List[str]] = None,  
    bold: bool = False,  
    italic: bool = False,  
):  
    """  
    在底图上添加文本并保存生成的图片。支持用户自定义字体大小、颜色、对齐方式等。  
    """  
    # 确保输出目录存在  
    os.makedirs(output_dir, exist_ok=True)  

    # 加载底图（假设底图路径是根据 base_image 生成的）  
    image_path = base_image  
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
    if outline_colors is None:  
        outline_colors = [None] * len(texts)  # 使用反色 
    if alignments is None:  
        alignments = ["center"] * len(texts)  # 默认居中  

    # 遍历检测框和文本  
    for i, (detection, text) in enumerate(zip(detections, texts)):  
        (startX, startY, endX, endY) = detection  

        # 计算文本框的宽度和高度  
        
        startX=(startX/1000)*image.width
        endX=(endX/1000)*image.width
        startY=(startY/1000)*image.height
        endY=(endY/1000)*image.height
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

        # draw.rectangle([startX, startY, endX, endY], outline="red", width=2)  

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
            font_size -= int(font_size/5)
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
        #描边颜色
        if outline_colors[i] is None:
            # 判断 font_color 是否更接近黑色
            if is_color_close_to_black(font_color):  
                outline_color = (255,255,255)  
            else:  
                outline_color = (0,0,0)
        else:  
            outline_color = outline_colors[i]

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

        # 在给定文本框内绘制多行文本  
        draw_multiline_text_with_outline(draw, (startX, startY), text, font, box_width, font_color, outline_color=outline_color, outline_width=outline_width, alignment=alignments[i])  

    # 生成唯一的文件名  
    output_filename = f"output_image_{uid}_{uuid.uuid4().hex}.jpg"  
    output_path = os.path.join(output_dir, output_filename)  
    image.save(output_path)  

    print(f"图片已保存到: {output_path}")  
    return output_path  


# 示例调用  
if __name__ == "__main__":  
    # 输入参数  
    uid = "12345"  
    base_image = "/mnt/afs/xueyingyi/image_vague/inpainting_demo/image_ (896).jpg" 
    response = "Writable text area[[0000, 0000, 0998, 0208]]:expect least from others enjoy your own company,\nWritable text area[[0000, 0798, 0998, 1000]]:for teaching us these important life lessons... thanks a lot mr. bean!!!,\nWritable text area[[0024, 0318, 0200, 0410]]:thank you mr. bean"  


    # 初始化 detections 和 texts 列表  
    detections = []  
    texts = []  

    # 按换行符分割 response  
    lines = response.split('\n')  

    # 遍历每一行  
    for line in lines:  
        try:  
            # 提取 detections  
            if '[' in line and ']' in line:  
                # 获取方括号内的内容  
                detection_part = line.split('[')[2].split(']')[0]  # 注意这里使用 split('[')[2]  
                if detection_part:  # 确保提取到了内容  
                    coords = tuple(map(int, detection_part.split(',')))  # 将字符串转换为整数的元组  
                    detections.append(coords)  

            # 提取 text  
            if ':' in line:  
                text_part = line.split(':')[1]  # 获取冒号后的内容  
                text = text_part.strip().rstrip(',')  # 去除前后空白字符和末尾的逗号  
                texts.append(text)  

        except (IndexError, ValueError) as e:  
            print(f"Error processing line: '{line}'. Error: {e}")  

    # 输出结果  
    print("detections =", detections)  
    print("texts =", texts)

    

    font_type = "DejaVuSans.ttf" 
    # 自定义参数（可选）  
    font_sizes = None  # 字体大小会自动计算  
    font_colors = None  # 每个文本框的字体颜色，会计算反色
    outline_colors = None #描边颜色
    outline_width = 4 #描边宽度
    alignments = ["left", "center", "right"]  # 可以分开控制每个文本框的对齐方式  
    bold = True  # 是否加粗  
    italic = False  # 是否斜体  

    # 调用函数生成图片  
    try:  
        output_path = generate_image_with_text(  
            uid, base_image, font_type, detections, texts,  
            font_sizes=font_sizes, font_colors=font_colors, outline_colors=outline_colors, outline_width=outline_width, alignments=alignments,  
            bold=bold, italic=italic  
        )  
        print(f"生成的图片路径: {output_path}")  
    except Exception as e:  
        print(f"生成图片时出错: {e}")
