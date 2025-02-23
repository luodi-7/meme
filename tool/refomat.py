# 用 ISO-8859-1 编码读取文件，然后保存为 UTF-8 编码
import pandas as pd

# 原始文件路径
input_file_path = '/mnt/afs/xueyingyi/meme/data/label_C_evaluate.csv'

# 读取 CSV 文件时使用 ISO-8859-1 编码
df = pd.read_csv(input_file_path, encoding='ISO-8859-1')

# 输出检查读取的数据，确保没有乱码
print("数据读取成功，预览部分数据：")
print(df.head())

# 保存为 UTF-8 编码的 CSV 文件
output_file_path = '/mnt/afs/xueyingyi/meme/data/label_C_evaluate_utf8.csv'
df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"数据已成功转换并保存为 UTF-8 编码: {output_file_path}")
