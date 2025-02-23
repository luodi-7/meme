import json

# 输入文件路径
categories_file = '/mnt/afs/xueyingyi/COCO/annotations/annotations/instances_train2017.json'

# 读取COCO标注文件
with open(categories_file, 'r') as f:
    coco_data = json.load(f)

# 提取类别信息，并构建类别ID到类别名称的映射
categories = {category['id']: category['name'] for category in coco_data['categories']}

# 输出类别映射
for category_id, category_name in categories.items():
    print(f"Category ID: {category_id}, Category Name: {category_name}")

# 输入文件路径
input_file = '/mnt/afs/xueyingyi/COCO/train_process/coco_annotations.jsonl'
output_file = '/mnt/afs/xueyingyi/COCO/train_process/processed_coco_annotations.jsonl'

# 函数：将 [x, y, width, height] 转换为 [ymin, xmin, ymax, xmax]
def convert_bbox_format(bbox):
    x, y, width, height = bbox
    ymin = y
    xmin = x
    ymax = y + height
    xmax = x + width
    return [ymin, xmin, ymax, xmax]

# 函数：将 [ymin, xmin, ymax, xmax] 转换为 <loc> 标签
def bbox_to_loc(bbox, image_width, image_height):
    ymin, xmin, ymax, xmax = bbox
    
    # 将坐标归一化到 [0, 1024] 范围内
    ymin_norm = int((ymin / image_height) * 1024)
    xmin_norm = int((xmin / image_width) * 1024)
    ymax_norm = int((ymax / image_height) * 1024)
    xmax_norm = int((xmax / image_width) * 1024)
    
    # 格式化为 <loc> 标签
    loc_str = f"<loc{ymin_norm:04}><loc{xmin_norm:04}><loc{ymax_norm:04}><loc{xmax_norm:04}>"
    return loc_str

# 读取数据并处理
with open(input_file, 'r') as f:
    lines = f.readlines()

processed_data = []

# 处理每一行数据
for line in lines:
    data = json.loads(line.strip())  # 解析每一行
    
    # 获取图像的基本信息
    image_id = data['image_id']
    file_name = data['file_name']
    height = data['height']
    width = data['width']
    
    # 存储该图像的处理结果
    image_data = {
        'image_id': image_id,
        'file_name': file_name,
        'height': height,
        'width': width,
        'annotations': []
    }
    
    # 处理每个标注
    for annotation in data['annotations']:
        # 获取原始bbox并转换格式
        bbox = annotation['bbox']
        converted_bbox = convert_bbox_format(bbox)
        
        # 转换为 <loc> 标签
        loc_label = bbox_to_loc(converted_bbox, width, height)
        
        # 获取类别名称
        category_id = annotation['category_id']
        category_name = categories.get(category_id, "unknown")
        
        # 创建新的annotation数据
        new_annotation = {
            'bbox': loc_label,
            'category_name': category_name  # 使用类别名称替代 category_id
        }
        
        # 添加到图像的标注列表
        image_data['annotations'].append(new_annotation)
    
    # 将处理后的数据添加到最终列表
    processed_data.append(image_data)

# 将处理后的数据写入输出文件
with open(output_file, 'w') as out_file:
    for image_data in processed_data:
        out_file.write(json.dumps(image_data) + '\n')

print(f"处理后的数据已保存到 {output_file}")
