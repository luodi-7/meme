import json  
import os  
from PIL import Image  

# 设置文件路径  
jsonl_file_path = '/mnt/afs/xueyingyi/image_vague/loc_dection/processed_dections_Eimage.jsonl'  
output_jsonl_file_path = '/mnt/afs/xueyingyi/image_vague/loc_dection/Eimage_solved/dections_Eimage_normalized.jsonl'  

# 准备写入数据  
normalized_data = []  

# 读取jsonl文件  
with open(jsonl_file_path, 'r') as file:  
    for line in file:  
        # 解析每行的json  
        record = json.loads(line.strip())  
        image_path = record['image_path']  
        
        # 打开图片以获取宽高  
        image = Image.open(image_path)  
        image_width, image_height = image.size  

        # 更新detections中的bbox  
        for detection in record['detections']:  
            # 获取原始bbox值  
            ymin, xmin, ymax, xmax = detection['bbox']  
            
            # 转换bbox格式  
            x_min = xmin  
            y_min = ymin  
            x_max = xmax  
            y_max = ymax  
            
            # 归一化bbox  
            x_min = int((x_min / image_width) * 1000)  
            x_max = int((x_max / image_width) * 1000)  
            y_min = int((y_min / image_height) * 1000)  
            y_max = int((y_max / image_height) * 1000)  
            
            # 更新bbox  
            detection['bbox'] = [x_min, y_min, x_max, y_max]  

        # 对detections按照bbox进行排序  
        record['detections'] = sorted(record['detections'], key=lambda d: (d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3]))  

        # 将处理好的记录添加到结果列表中  
        normalized_data.append(record)  

# 将结果写入新的jsonl文件  
with open(output_jsonl_file_path, 'w') as output_file:  
    for entry in normalized_data:  
        output_file.write(json.dumps(entry) + '\n')  

print(f'Data has been normalized and saved to {output_jsonl_file_path}')