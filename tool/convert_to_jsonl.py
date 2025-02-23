import json

# 输入和输出文件路径
input_file_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/user_input_descriptions_simple_miss.jsonl'  # 输入文件路径
output_file_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/user_input_1.jsonl'  # 输出的JSONL文件路径

# 打开输入文件和输出文件
with open(input_file_path, 'r', encoding='ISO-8859-1') as input_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
    # 读取输入文件内容
    content = input_file.read()
    
    # 按```json和```分割内容
    blocks = content.split('```json')
    
    # 遍历每个块
    for block in blocks:
        block = block.strip()  # 去除空白字符
        if not block:  # 如果块为空，跳过
            continue
        
        # 提取JSON对象部分（去掉最后的```）
        json_part = block.split('```')[0].strip()
        
        try:
            # 解析JSON对象
            data = json.loads(json_part)
            
            # 提取需要的字段
            transformed_item = {
                "file_name": data["file_name"],
                "user_input": {
                    "Emotion Category": data["user_input"]["Emotion Category"],
                    "Emotion Intensity": data["user_input"]["Emotion Intensity"],
                    "Intention Category": data["user_input"]["Intention Category"],
                    "Scene or Theme": data["user_input"]["Scene or Theme"],
                    "Style Preference": data["user_input"]["Style Preference"],
                    "Text Content Keywords": data["user_input"]["Text Content Keywords"]
                }
            }
            
            # 将转换后的对象写入输出文件（JSONL格式）
            output_file.write(json.dumps(transformed_item) + '\n')
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}，跳过此块: {json_part}")
            continue

print(f"转换完成，结果已保存到: {output_file_path}")