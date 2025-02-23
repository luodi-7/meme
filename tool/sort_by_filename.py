import csv
import json

# 读取 merged_output.csv 文件
csv_file_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/merged_output.csv'
file_names = []

with open(csv_file_path, 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        file_names.append(row['file_name'])

# 读取 user_input.jsonl 文件
jsonl_file_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/user_input.jsonl'
jsonl_data = {}

with open(jsonl_file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        data = json.loads(line)
        jsonl_data[data['file_name']] = data

# 按照 file_names 的顺序重新排列 jsonl_data
sorted_jsonl_data = [jsonl_data[file_name] for file_name in file_names if file_name in jsonl_data]

# 将排序后的内容写回新的 user_input.jsonl 文件
sorted_jsonl_file_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/user_input_sorted.jsonl'

with open(sorted_jsonl_file_path, 'w') as sorted_jsonl_file:
    for data in sorted_jsonl_data:
        sorted_jsonl_file.write(json.dumps(data) + '\n')