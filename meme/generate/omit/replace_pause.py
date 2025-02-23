import re
import json

# 输入文件路径
input_file = "/mnt/afs/xueyingyi/meme/generate/omit/C_generate_multi.jsonl"
# 输出文件路径
output_file = "/mnt/afs/xueyingyi/meme/generate/omit/C_generate_multi_omit.jsonl"

# 正则表达式匹配 <pause_i> 样式的占位符
pause_pattern = re.compile(r'<pause_\d+>')

def clean_pause_placeholders(data):
    """
    递归清理数据中的 <pause_i> 占位符
    """
    if isinstance(data, dict):
        # 如果是字典，遍历所有键值对
        for key, value in data.items():
            if isinstance(value, str):
                # 如果是字符串，清理占位符
                data[key] = pause_pattern.sub('', value)
            elif isinstance(value, (list, dict)):
                # 如果是列表或字典，递归清理
                clean_pause_placeholders(value)
    elif isinstance(data, list):
        # 如果是列表，遍历所有元素
        for i, item in enumerate(data):
            if isinstance(item, str):
                # 如果是字符串，清理占位符
                data[i] = pause_pattern.sub('', item)
            elif isinstance(item, (list, dict)):
                # 如果是列表或字典，递归清理
                clean_pause_placeholders(item)
    return data

# 打开输入文件和输出文件
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # 解析 JSON 行
        data = json.loads(line.strip())
        
        # 清理 user_input 字段中的所有 <pause_i> 占位符
        if 'user_input' in data:
            data['user_input'] = clean_pause_placeholders(data['user_input'])
        
        # 将处理后的数据写回 JSONL 文件
        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"处理完成！清理后的文件已保存到: {output_file}")