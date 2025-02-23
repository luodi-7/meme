# import csv

# # 文件路径
# input_csv_path = '/mnt/afs/xueyingyi/meme/generate/E_text.csv'  # 原始文件路径
# output_csv_path = '/mnt/afs/xueyingyi/meme/generate/E_text_1.csv'  # 新文件路径

# # 用 ISO-8859-1 读取文件，并用 UTF-8 写入新文件
# with open(input_csv_path, 'r', encoding='gbk') as input_file, open(output_csv_path, 'w', encoding='utf-8', newline='') as output_file:
#     csv_reader = csv.reader(input_file)
#     csv_writer = csv.writer(output_file)
    
#     # 写入标题行
#     header = next(csv_reader)
#     csv_writer.writerow(header)
    
#     # 写入数据行
#     for row in csv_reader:
#         csv_writer.writerow(row)

# print(f"文件已重新编码并保存到: {output_csv_path}")


import csv  

# 文件路径  
input_file_path = '/mnt/afs/xueyingyi/location_bbox_align_ep10/output.txt'  # 原始文件路径  
output_file_path = '/mnt/afs/xueyingyi/location_bbox_align_ep10/output_utf8.txt'  # 新文件路径  

# 使用 gbk 读取文件，并用 utf-8 写入新文件  
with open(input_file_path, 'r', encoding='utf-8') as input_file, open(output_file_path, 'w', encoding='utf-8', newline='') as output_file:  
    # 在这里假设需要处理的文件是以逗号分隔的 CSV 格式  
    # 如果不是，请根据实际格式调整代码  
    csv_reader = csv.reader(input_file)  
    csv_writer = csv.writer(output_file)  
    
    # 将所有行写入新的文件  
    for row in csv_reader:  
        csv_writer.writerow(row)  

print(f"文件已重新编码并保存到: {output_file_path}")