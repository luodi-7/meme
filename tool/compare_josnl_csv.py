import json
import csv

# 文件路径
jsonl_file_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/user_input.jsonl'
csv_file_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/merged_output.csv'
output_jsonl_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/missing_in_csv.jsonl'

# 读取CSV文件中的所有file_name
csv_file_names = set()
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        csv_file_names.add(row['file_name'])

# 读取JSONL文件并找出在CSV中不存在的行
missing_in_csv = []
with open(jsonl_file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        data = json.loads(line)
        if data['file_name'] not in csv_file_names:
            missing_in_csv.append(data)

# 将缺失的行写入新的JSONL文件
if missing_in_csv:
    with open(output_jsonl_path, 'w') as output_jsonl_file:
        for item in missing_in_csv:
            output_jsonl_file.write(json.dumps(item) + '\n')
    print(f"Data missing in CSV has been written to {output_jsonl_path}")
else:
    print("No data missing in CSV found.")