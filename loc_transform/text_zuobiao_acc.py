from PIL import Image, ImageDraw

# 图片路径
image_path = "/mnt/afs/xueyingyi/image_vague/inpainting_demo/image_ (896).jpg"

# 检测到的文字位置坐标
text_boxes = {
    "1": [0, 0, 998, 208], 
    "2": [0, 798, 998, 1000],
    "3": [24, 318, 200, 410]
}

# 打开图片
image = Image.open(image_path)

# 打印图片信息
print(f"图片模式: {image.mode}")
print(f"图片尺寸: {image.size}")
print(image.height)
print(image.width)

# 如果图像是 P 模式（调色板模式），转换为 RGB 模式
if image.mode == 'P':
    image = image.convert('RGB')

# 创建一个可以在图像上绘制的对象
draw = ImageDraw.Draw(image)

# 遍历每个文本框的坐标
for label, box in text_boxes.items():
    # 提取左上角和右下角坐标
    x1, y1, x2, y2 = box
    # x2=x1+x2
    # y2=y1+y2
    x1=(x1/1000)*image.width
    x2=(x2/1000)*image.width
    y1=(y1/1000)*image.height
    y2=(y2/1000)*image.height
    print(f"绘制矩形框 ({label}): ({x1}, {y1}) -> ({x2}, {y2})")
    # 绘制矩形框
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    # 在矩形框上方绘制标签
    draw.text((x1, y1 - 20), label, fill="red")

# 保存绘制后的图像
output_path = "/mnt/afs/xueyingyi/COCO/test_output/000000000139.jpg"
print(f"保存路径: {output_path}")
image.save(output_path)

# 显示图像（可选）
image.show()