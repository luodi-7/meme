import xml.etree.ElementTree as ET

# 解析 XML 文件
tree = ET.parse('/mnt/afs/xueyingyi/annotations/annotations.xml')
root = tree.getroot()

# 初始化字典存储图片信息
successful_images = []
fail_images = []

# 遍历所有图片
for image in root.findall('image'):
    image_name = image.get('name')
    tags = image.findall('tag')
    
    # 检查标签
    for tag in tags:
        label = tag.get('label')
        if label == 'successful':
            successful_images.append(image_name)
        elif label == 'fail':
            fail_images.append(image_name)

# 将结果写入文件
with open('successful_images.txt', 'w') as f_success, open('fail_images.txt', 'w') as f_fail:
    for img in successful_images:
        f_success.write(img + '\n')
    for img in fail_images:
        f_fail.write(img + '\n')

# 输出统计信息
print(f"Successful images: {len(successful_images)}")
print(f"Fail images: {len(fail_images)}")