import json

# 定义文件路径
file1_path = '/mnt/afs/xueyingyi/meme/data/Cjson/C_generate_eval_multi_100.jsonl'
file2_path = '/mnt/afs/xueyingyi/meme/data/Cjson/C_generate_eval_multi.jsonl'
output_path = '/mnt/afs/xueyingyi/meme/compare/eval.jsonl'

# 读取第一个文件并存储第二个image的值
image_set = set()
with open(file1_path, 'r') as file1:
    for line in file1:
        data = json.loads(line)
        if len(data['image']) > 1:
            image_set.add(data['image'][1])

# 读取第二个文件并比较第二个image的值
matching_data = []
with open(file2_path, 'r') as file2:
    for line in file2:
        data = json.loads(line)
        if len(data['image']) > 1 and data['image'][1] in image_set:
            matching_data.append(data)

# 将匹配的数据写入新的JSONL文件
with open(output_path, 'w') as output_file:
    for data in matching_data:
        output_file.write(json.dumps(data) + '\n')

print(f"匹配的数据已保存到 {output_path}")