import csv

# 文件路径
csv_file_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/unsolved_output.csv'

# 初始化变量
total_lines = 0  # 文件总行数
header_line = 0  # 标题行
empty_lines = 0  # 空行数量
data_lines = 0  # 数据行数量

# 打开文件并逐行检查
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        total_lines += 1
        if total_lines == 1:
            header_line = 1  # 第一行是标题行
        elif not row:  # 检查是否为空行
            empty_lines += 1
        else:
            data_lines += 1

# 输出结果
print(f"文件总行数: {total_lines}")
print(f"标题行数量: {header_line}")
print(f"空行数量: {empty_lines}")
print(f"数据行数量: {data_lines}")

# 检查数据行数量是否与预期一致
if data_lines == 76229:
    print("数据行数量与预期一致。")
else:
    print("数据行数量与预期不一致，请检查文件内容。")