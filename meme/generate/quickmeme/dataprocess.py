import pandas as pd
import json

# 读取CSV文件
csv_file_path = '/mnt/afs/niuyazhe/data/lister/meme/quickmeme/table.csv'
df_csv = pd.read_csv(csv_file_path)

# 读取JSON文件
json_file_path = '/mnt/afs/niuyazhe/data/lister/meme/updated_quickmeme_label.json'
with open(json_file_path, 'r') as f:
    json_data = json.load(f)

# 创建一个字典来存储JSON数据，方便后续查找
json_dict = {item['uid'].split('/')[-1]: item for item in json_data}

# 定义映射关系
sentiment_category_map = {
    'happiness': '1(happiness)',
    'love': '2(love)',
    'anger': '3(anger)',
    'sorrow': '4(sorrow)',
    'fear': '5(fear)',
    'hate': '6(hate)',
    'surprise': '7(surprise)',
    '1': '1(happiness)',
    '2': '2(love)',
    '3': '3(anger)',
    '4': '4(sorrow)',
    '5': '5(fear)',
    '6': '6(hate)',
    '7': '7(surprise)'
}

sentiment_degree_map = {
    'slightly': '1(slightly)',
    'moderately': '2(moderately)',
    'very': '3(very)'

}

intention_detection_map = {
    'interactive': '1(interactive)',
    'expressive': '2(expressive)',
    'entertaining': '3(entertaining)',
    'offensive': '4(offensive)',
    'other': '5(other)',
    '1': '1(interactive)',
    '2': '2(expressive)',
    '3': '3(entertaining)',
    '4': '4(offensive)',
    '5': '5(other)'
}

offensiveness_detection_map = {
    'non-offensive': '0(non-offensive)',
    'slightly': '1(slightly)',
    'moderately': '2(moderately)',
    'very': '3(very)'
}

# 准备输出的数据
output_data = []

# 遍历CSV文件的每一行
for index, row in df_csv.iterrows():
    file_name = f"{row['id']}.jpg"
    text = row['title']
    
    # 查找对应的JSON数据
    json_item = json_dict.get(file_name)
    
    if json_item:
        # 映射各个字段
        sentiment_category = sentiment_category_map.get(json_item['sentiment_category'], json_item['sentiment_category'])
        sentiment_degree = sentiment_degree_map.get(json_item['sentiment_degree'], json_item['sentiment_degree'])
        intention_detection = intention_detection_map.get(json_item['intention_detection'], json_item['intention_detection'])
        offensiveness_detection = offensiveness_detection_map.get(json_item['offensiveness_detection'], json_item['offensiveness_detection'])
        
        # 添加一行数据到输出列表
        output_data.append([
            file_name,
            sentiment_category,
            sentiment_degree,
            intention_detection,
            offensiveness_detection,
            json_item['metaphor_occurrence'],
            json_item['metaphor_category'],
            json_item['target_domain'],
            json_item['source_domain'],
            json_item['target_modality'],
            json_item['source_modality'],
            text
        ])

# 将输出数据转换为DataFrame
output_df = pd.DataFrame(output_data, columns=[
    'file_name',
    'sentiment category',
    'sentiment degree',
    'intention detection',
    'offensiveness detection',
    'metaphor occurrence',
    'metaphor category',
    'target domain',
    'source domain',
    'target modality',
    'source modality',
    'text'
])

# 保存为新的CSV文件
output_csv_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/merged_output.csv'
output_df.to_csv(output_csv_path, index=False)

print(f"处理完成，结果已保存到 {output_csv_path}")