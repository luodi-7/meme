import csv

# 文件路径
missing_data_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/merged_output.csv'
unsolved_output_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/unsolved_output.csv'
output_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/missing_in_unsolved.csv'

# 读取 unsolved_output.csv 文件中的所有 file_name
unsolved_file_names = set()
with open(unsolved_output_path, 'r') as unsolved_file:
    csv_reader = csv.DictReader(unsolved_file)
    for row in csv_reader:
        unsolved_file_names.add(row['file_name'])

# 读取 missing_data.csv 文件并找出不在 unsolved_output.csv 中的行
missing_in_unsolved = []
with open(missing_data_path, 'r') as missing_file:
    csv_reader = csv.DictReader(missing_file)
    for row in csv_reader:
        if row['file_name'] not in unsolved_file_names:
            missing_in_unsolved.append(row)

# 将结果写入新的 CSV 文件
if missing_in_unsolved:
    with open(output_path, 'w', newline='') as output_file:
        csv_writer = csv.DictWriter(output_file, fieldnames=csv_reader.fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(missing_in_unsolved)
    print(f"在 missing_data.csv 但不在 unsolved_output.csv 中的数据已写入: {output_path}")
    print(f"找到 {len(missing_in_unsolved)} 条数据。")
else:
    print("没有找到在 missing_data.csv 但不在 unsolved_output.csv 中的数据。")