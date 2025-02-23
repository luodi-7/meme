import json
import csv

# 读取JSONL文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# 读取CSV文件
def read_csv(file_path):
    data = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data[row['file_name']] = row['text']
    return data

# 将detections转换为指定格式
def convert_detections(detections):
    converted = []
    for detection in detections:
        bbox = detection['bbox']
        text = detection['text']
        # 将bbox和text转换为指定格式
        converted.append({
            "bbox": f"<box>[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]</box>",
            "text": text
        })
    return converted

def process_data(jsonl_data, csv_data):  
    output = []  
    # 读取备用JSONL文件  
    with open('/mnt/afs/xueyingyi/image_vague/loc_dection/dections_Eimage.jsonl', 'r') as backup_file:  
        backup_data = {json.loads(line)['image_path'].split('/')[-1]: json.loads(line) for line in backup_file}  

    for idx, item in enumerate(jsonl_data):  
        file_name = item['image_path'].split('/')[-1]  
        if file_name in csv_data:  
            # 转换detections  
            converted_detections = convert_detections(item['detections'])  
            
            # 如果converted_detections为空且csv_data[file_name]不为空  
            if not converted_detections and csv_data[file_name].strip():  
                # 从备用JSONL文件中获取对应的detections  
                if file_name in backup_data:  
                    backup_item = backup_data[file_name]  
                    if 'detections' in backup_item and backup_item['detections']:  
                        # 只保留第一个bbox，并替换text  
                        first_detection = backup_item['detections'][0]  
                        converted_detections = [{  
                            "bbox": f"<box>[{first_detection['bbox'][0]}, {first_detection['bbox'][1]}, {first_detection['bbox'][2]}, {first_detection['bbox'][3]}]</box>",  
                            "text": csv_data[file_name]  
                        }]  
            
            # 构建conversations  
            conversation = [  
                {"from": "human", "value": f"I'm going to give you a sentence and a picture. Please divide the whole sentence sensibly and place each in the right place in the picture to convey the meaning of the whole picture humor. Finally, please give me the text box coordinates location and the corresponding text.\n<image>\n Now, please deal with this sentence: {csv_data[file_name]}"},  
                {"from": "gpt", "value": converted_detections}  
            ]  
            # 构建最终输出  
            output.append({  
                "id": idx,  
                "image": item['image_path'].replace("image_vague/image", "image_vague/inpainting_demo"),  
                "conversations": conversation  
            })  
    return output

# 文件路径
jsonl_file_path = '/mnt/afs/xueyingyi/image_vague/loc_dection/Eimage_solved/dections_Eimage_normalized.jsonl'
csv_file_path = '/mnt/afs/xueyingyi/meme/generate/E_text_1.csv'

# 读取数据
jsonl_data = read_jsonl(jsonl_file_path)
csv_data = read_csv(csv_file_path)

# 处理数据
processed_data = process_data(jsonl_data, csv_data)

# 输出到JSONL文件
with open('/mnt/afs/xueyingyi/image_vague/loc_dection/Eimage_solved/processed_data_match.jsonl', 'w') as outfile:
    for item in processed_data:
        json.dump(item, outfile)
        outfile.write('\n')  # 每行一个JSON对象

print("数据处理完成，已保存到 '/mnt/afs/xueyingyi/image_vague/loc_dection/Eimage_solved/processed_data_match.jsonl'")