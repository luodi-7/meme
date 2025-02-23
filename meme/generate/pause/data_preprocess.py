import csv
import json
import random

# 文件路径
csv_file_path = '/mnt/afs/xueyingyi/meme/generate/E_text_1.csv'  # CSV文件路径
user_input_jsonl_path = '/mnt/afs/xueyingyi/meme/pause_data/all_item/C_generate_pause.jsonl'  # user_input.jsonl文件路径
output_jsonl_path = '/mnt/afs/xueyingyi/meme/pause_data/all_item/Cjson/C_generate_pause.jsonl'  # 输出JSONL文件路径
train_jsonl_path = '/mnt/afs/xueyingyi/meme/pause_data/all_item/Cjson/C_generate_train_pause.jsonl'  # 训练集路径
eval_jsonl_path = '/mnt/afs/xueyingyi/meme/pause_data/all_item/Cjson/C_generate_eval_pause.jsonl'  # 测试集路径
train_config_path = '/mnt/afs/xueyingyi/meme/pause_data/all_item/C_generate_train_pause.jsonl'  # 训练集配置文件路径
eval_config_path = '/mnt/afs/xueyingyi/meme/pause_data/all_item/C_generate_eval_pause.jsonl'  # 测试集配置文件路径

# 读取CSV文件
csv_data = {}
with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        file_name = row['file_name']
        text = row['text'].strip()
        csv_data[file_name] = text  # 存储file_name和text的映射关系

# 读取user_input.jsonl文件
user_input_data = []
with open(user_input_jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        user_input_data.append(json.loads(line.strip()))

# 构建JSONL数据
jsonl_data = []
for idx, item in enumerate(user_input_data):
    file_name = item['file_name']
    user_input = item['user_input']
    
    # 检查file_name是否在CSV数据中
    if file_name not in csv_data:
        print(f"警告: {file_name} 在CSV文件中未找到，跳过此条数据")
        continue
    
    # 获取对应的text
    text = csv_data[file_name]
    
    # 构建提示词
    with open('/mnt/afs/xueyingyi/vl2.5/InternVL/inference/text_new.txt', 'r') as prompt_file:
        PROMPT = prompt_file.read()

    with open('/mnt/afs/xueyingyi/vl2.5/InternVL/inference/text_example.txt', 'r') as prompt_file:
        PROMPT_example = prompt_file.read()
    
    # 构建conversations
    conversations = [
        {
            "from": "human",
            "value": f"{PROMPT}<image>\n{PROMPT_example}\n<image>\n{user_input}"
        },
        {
            "from": "gpt",
            "value": text  # 使用CSV中的文字
        }
    ]
    
    # 构建JSON对象(单图像)
    # json_obj = {
    #     "id": idx,
    #     "image": f"/mnt/afs/xueyingyi/image_vague/inpainting_demo/{file_name}",
    #     "conversations": conversations
    # }
    # 构建JSON对象(多图像)
    json_obj = {
        "id": idx,
        "image": [
        f"/mnt/afs/xueyingyi/vl2.5/InternVL/inference/example.jpg",
        f"/mnt/afs/xueyingyi/image_vague/inpainting_demo/{file_name}"
        ], 
        "conversations": conversations
    }
    
    jsonl_data.append(json_obj)

# 保存为JSONL文件
with open(output_jsonl_path, 'w', encoding='utf-8') as f:
    for item in jsonl_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 划分训练集和测试集
random.seed(42)  # 设置随机种子以确保可重复性
random.shuffle(jsonl_data)  # 打乱数据
train_size = int(len(jsonl_data) * 0.9)  # 90%训练集
train_data = jsonl_data[:train_size]
eval_data = jsonl_data[train_size:]

# 保存训练集
with open(train_jsonl_path, 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 保存测试集
with open(eval_jsonl_path, 'w', encoding='utf-8') as f:
    for item in eval_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 生成训练集配置文件
train_config = {
    "classification_C": {
        "root": "/mnt/afs/xueyingyi/image_vague/inpainting_demo",
        "annotation": train_jsonl_path,
        "data_augment": False,
        "repeat_time": 1,
        "length": len(train_data)
    }
}
with open(train_config_path, 'w', encoding='utf-8') as f:
    json.dump(train_config, f, ensure_ascii=False, indent=4)

# 生成测试集配置文件
eval_config = {
    "classification_C": {
        "root": "/mnt/afs/xueyingyi/image_vague/inpainting_demo",
        "annotation": eval_jsonl_path,
        "data_augment": False,
        "repeat_time": 1,
        "length": len(eval_data)
    }
}
with open(eval_config_path, 'w', encoding='utf-8') as f:
    json.dump(eval_config, f, ensure_ascii=False, indent=4)

print("数据处理完成！")
print(f"训练集大小: {len(train_data)}")
print(f"测试集大小: {len(eval_data)}")