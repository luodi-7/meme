import json
import os

# 加载COCO的标注文件
annotations_file = '/mnt/afs/xueyingyi/COCO/annotations/annotations/instances_train2017.json'

# 读取COCO标注文件
with open(annotations_file, 'r') as f:
    coco_data = json.load(f)

# 提取图像信息
images_info = coco_data['images']
annotations_info = coco_data['annotations']

# 创建一个字典用于按图片ID查找标注信息
annotations_by_image_id = {}
for annotation in annotations_info:
    image_id = annotation['image_id']
    if image_id not in annotations_by_image_id:
        annotations_by_image_id[image_id] = []
    annotations_by_image_id[image_id].append({
        'bbox': annotation['bbox'],
        'category_id': annotation['category_id'],
        'id': annotation['id']
    })

# 打开jsonl文件准备写入
output_file = '/mnt/afs/xueyingyi/COCO/train_process/coco_annotations.jsonl'
with open(output_file, 'w') as out_file:
    for image in images_info:
        image_id = image['id']
        image_data = {
            'image_id': image_id,
            'file_name': image['file_name'],
            'height': image['height'],
            'width': image['width'],
            'annotations': annotations_by_image_id.get(image_id, [])
        }
        # 每行输出一个图像的json对象
        out_file.write(json.dumps(image_data) + '\n')

print(f"所有数据已经提取并保存到 {output_file}")
