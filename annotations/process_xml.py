import xml.etree.ElementTree as ET

# 加载 XML 文件
tree = ET.parse('annotations.xml')
root = tree.getroot()

# 打印根标签
print("Root tag:", root.tag)

# 遍历 meta 信息
meta = root.find('meta')
job = meta.find('job')
print("Job ID:", job.find('id').text)
print("Created:", job.find('created').text)
print("Owner:", job.find('owner/username').text)

# 遍历 labels
labels = job.find('labels')
for label in labels.findall('label'):
    print("Label Name:", label.find('name').text)
    print("Label Color:", label.find('color').text)

# 遍历图像标注
for image in root.findall('image'):
    print("Image ID:", image.get('id'))
    print("Image Name:", image.get('name'))
    print("Image Width:", image.get('width'))
    print("Image Height:", image.get('height'))
    for tag in image.findall('tag'):
        print("Tag Label:", tag.get('label'))
        print("Tag Source:", tag.get('source'))