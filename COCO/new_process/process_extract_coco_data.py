import json

# 输入文件路径
categories_file = '/mnt/afs/xueyingyi/COCO/annotations/annotations/instances_val2017.json'

# 读取COCO标注文件
with open(categories_file, 'r') as f:
    coco_data = json.load(f)

# 提取类别信息，并构建类别ID到类别名称的映射
categories = {category['id']: category['name'] for category in coco_data['categories']}

# 输出类别映射
for category_id, category_name in categories.items():
    print(f"Category ID: {category_id}, Category Name: {category_name}")

# 输入文件路径
input_file = '/mnt/afs/xueyingyi/COCO/new_process/coco_annotations.jsonl'
output_file = '/mnt/afs/xueyingyi/COCO/new_process/processed_coco_annotations.jsonl'

# 函数：将 [x, y, width, height] 转换为 [xmin, ymin, xmax, ymax]
def convert_bbox_format(bbox,image_width,image_height):  
    x, y, width, height = bbox  
    ymin = y  
    xmin = x  
    ymax = y + height  
    xmax = x + width  
    xmin = int((xmin / image_width)*1000)
    xmax = int((xmax / image_width)*1000)
    ymin = int((ymin / image_height)*1000)
    ymax = int((ymax / image_height)*1000)
    
    # # Get the number of decimal places in xmin  
    # decimal_places = len(str(xmin).split('.')[-1]) if '.' in str(xmin) else 0  
    
    # # Format all coordinates to have the same number of decimal places as xmin  
    # xmin = round(xmin, decimal_places)  
    # ymin = round(ymin, decimal_places)  
    # xmax = round(xmax, decimal_places)  
    # ymax = round(ymax, decimal_places)  

    return [xmin, ymin, xmax, ymax]

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
        converted_bbox = convert_bbox_format(bbox,width, height)
        

        
        # 获取类别名称
        category_id = annotation['category_id']
        category_name = categories.get(category_id, "unknown")
        
        # 创建新的annotation数据
        new_annotation = {
            'bbox': converted_bbox,
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
