import json  
import re  

# 输入和输出文件路径  
input_file = '/mnt/afs/xueyingyi/COCO/new_process/new_data.jsonl'  
output_file = '/mnt/afs/xueyingyi/COCO/new_process/filtered_data.jsonl'  

# 读取数据并进行处理  
with open(input_file, 'r') as f:  
    lines = f.readlines()  

filtered_data = []  

# 处理每一行数据  
for line in lines:  
    data = json.loads(line.strip())  # 解析每一行  
    conversations = data.get('conversations', [])  
    
    # 查找gpt的回答  
    for conversation in conversations:  
        if conversation['from'] == "gpt":  
            # 使用正则表达式匹配<box>中的内容  
            match = re.search(r'<box>(.*?)</box>', conversation['value'])  
            if match:  
                bbox_content = match.group(1)  # 提取边界框内容  
                # 判断是否只有一个边界框  
                if bbox_content.startswith('[[') and bbox_content.endswith(']]'):  
                    # 检查框的数量  
                    bbox_list = json.loads(bbox_content)  
                    if len(bbox_list) == 1:  # 只有一个边界框  
                        filtered_data.append(data)  # 将该数据添加到结果中  
                        break  # 只需找到一个匹配就可以了  

# 将筛选后的数据写入输出文件  
with open(output_file, 'w') as out_file:  
    for image_data in filtered_data:  
        out_file.write(json.dumps(image_data) + '\n')  

print(f"只包含单个边界框的数据已保存到 {output_file}")