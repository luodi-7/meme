import json

# 输入文件路径
input_file = '/mnt/afs/xueyingyi/COCO/new_process/processed_coco_annotations.jsonl'
output_file = '/mnt/afs/xueyingyi/COCO/new_process/new_data.jsonl'

# 读取数据并进行处理
with open(input_file, 'r') as f:
    lines = f.readlines()

processed_data = []

# 设置数字ID的起始值
id_counter = 1

# 处理每一行数据
for line in lines:
    data = json.loads(line.strip())  # 解析每一行
    
    # 获取图像的基本信息
    image_id = data['image_id']
    file_name = data['file_name']
    image_path = f"/mnt/afs/xueyingyi/COCO/val2017/{file_name}"
    
    # 按类别名称将annotations分组
    category_annotations = {}
    for annotation in data['annotations']:
        category_name = annotation['category_name']
        bbox = annotation['bbox']
        
        # 如果该类别已存在，追加bbox，否则创建新的类别
        if category_name not in category_annotations:
            category_annotations[category_name] = []
        category_annotations[category_name].append(bbox)
    
    # 对每个类别生成一个对话
    for category_name, bboxes in category_annotations.items():
        # 生成纯数字ID，递增
        conversation_id = id_counter
        id_counter += 1  # 增加ID
        
        # 构建"human"提问
        human_value = f"<image>\nPlease provide the bounding box coordinates of the region this sentence describes: <ref>{category_name}</ref>"
        

        gpt_value = f"<ref>{category_name}</ref><box>{json.dumps(bboxes)}</box>"  # 使用分号分隔不同的bbox
        
        # 构建对话对象
        conversation_data = {
            "id": conversation_id,
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": human_value
                },
                {
                    "from": "gpt",
                    "value": gpt_value
                }
            ]
        }
        
        # 将生成的对话对象添加到最终列表
        processed_data.append(conversation_data)

# 将处理后的数据写入输出文件
with open(output_file, 'w') as out_file:
    for image_data in processed_data:
        out_file.write(json.dumps(image_data) + '\n')

print(f"处理后的数据已保存到 {output_file}")