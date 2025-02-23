import json
import csv

# 输入和输出文件路径
jsonl_file_path = '/mnt/afs/xueyingyi/meme/generate/user_input_descriptions_simple.jsonl'  # JSONL文件路径
csv_file_path = '/mnt/afs/xueyingyi/meme/generate/E_text.csv'  # 包含文本内容的CSV文件路径
output_csv_path = '/mnt/afs/xueyingyi/meme/generate/text_pipeline_label.csv'  # 生成的CSV文件路径

# 打开CSV文件并读取内容到字典
text_dict = {}
with open(csv_file_path, 'r', encoding='ISO-8859-1') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        file_name = row['file_name']
        text = row['text']
        text_dict[file_name] = text  # 存储file_name和text的映射关系

# 打开JSONL文件并生成新的CSV文件
with open(jsonl_file_path, 'r') as jsonl_file, open(output_csv_path, 'w', newline='', encoding='utf-8') as output_csv:
    csv_writer = csv.writer(output_csv)
    
    # 写入CSV文件的标题行
    csv_writer.writerow([
        'file_name', 'Emotion Category', 'Emotion Intensity', 'Text Content Keywords', 'text'
    ])
    
    # 处理JSONL文件中的每一行
    for line in jsonl_file:
        try:
            data = json.loads(line.strip())  # 解析JSONL文件中的一行
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}，跳过此行: {line}")
            continue
        
        file_name = data['file_name']  # 获取文件名
        user_input = data['user_input']  # 获取用户输入
        
        # 提取所需的信息
        emotion_category = user_input['Emotion Category']  # 情感类别
        emotion_intensity = user_input['Emotion Intensity']  # 情感强度
        text_content_keywords = ', '.join(user_input['Text Content Keywords'])  # 文本关键词
        text = text_dict.get(file_name, '')  # 从CSV文件中获取对应的文本内容
        
        # 写入CSV文件
        csv_writer.writerow([
            file_name, emotion_category, emotion_intensity, text_content_keywords, text
        ])

print(f"CSV文件已生成: {output_csv_path}")