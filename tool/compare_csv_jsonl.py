import json
import csv

# 文件路径
jsonl_file_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/user_input.jsonl'
csv_file_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/merged_output.csv'
output_csv_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/missing_data.csv'

# 读取JSONL文件中的所有file_name
jsonl_file_names = set()
jsonl_duplicates = set()  # 用于记录重复的file_name
with open(jsonl_file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        try:
            data = json.loads(line)
            file_name = data['file_name']
            if file_name in jsonl_file_names:
                jsonl_duplicates.add(file_name)  # 记录重复的file_name
            jsonl_file_names.add(file_name)
        except json.JSONDecodeError:
            print(f"JSONL文件解析错误: {line}")

# 输出JSONL文件中的重复file_name
if jsonl_duplicates:
    print(f"JSONL文件中存在重复的file_name: {jsonl_duplicates}")
else:
    print("JSONL文件中没有重复的file_name。")

# 读取CSV文件并找出缺失的行
missing_rows = []
csv_file_names = set()
csv_duplicates = set()  # 用于记录CSV文件中的重复file_name
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        file_name = row['file_name']
        if file_name in csv_file_names:
            csv_duplicates.add(file_name)  # 记录重复的file_name
        csv_file_names.add(file_name)
        if file_name not in jsonl_file_names:
            missing_rows.append(row)

# 输出CSV文件中的重复file_name
if csv_duplicates:
    print(f"CSV文件中存在重复的file_name: {csv_duplicates}")
else:
    print("CSV文件中没有重复的file_name。")

# 将缺失的行写入新的CSV文件
if missing_rows:
    with open(output_csv_path, 'w', newline='') as output_csv_file:
        csv_writer = csv.DictWriter(output_csv_file, fieldnames=csv_reader.fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(missing_rows)
    print(f"Missing data has been written to {output_csv_path}")
    print(f"缺失的数据条数: {len(missing_rows)}")
else:
    print("No missing data found.")