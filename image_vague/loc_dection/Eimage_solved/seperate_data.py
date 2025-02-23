import json  
import os  
import random  # 导入 random 模块  

# 文件路径设置  
input_file_path = '/mnt/afs/xueyingyi/image_vague/loc_dection/Eimage_solved/processed_data_match.jsonl'  
train_file_path = '/mnt/afs/xueyingyi/image_vague/loc_dection/Eimage_solved/matched/train_data.jsonl'  
eval_file_path = '/mnt/afs/xueyingyi/image_vague/loc_dection/Eimage_solved/matched/eval_data.jsonl'  

# 读取数据  
data = []  
with open(input_file_path, 'r') as file:  
    for line in file:  
        data.append(json.loads(line.strip()))  

# 打乱数据顺序  
random.shuffle(data)  # 使用 random.shuffle 打乱数据  

# 计算切分索引  
total_count = len(data)  
train_count = int(total_count * 0.9)  
eval_count = total_count - train_count  

# 分割数据  
train_data = data[:train_count]  
eval_data = data[train_count:]  

# 保存训练数据  
with open(train_file_path, 'w') as train_file:  
    for record in train_data:  
        train_file.write(json.dumps(record) + '\n')  

# 保存评估数据  
with open(eval_file_path, 'w') as eval_file:  
    for record in eval_data:  
        eval_file.write(json.dumps(record) + '\n')  

print(f'Train data saved to {train_file_path}')  
print(f'Eval data saved to {eval_file_path}')