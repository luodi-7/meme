import json  
import re  
import os  
from collections import defaultdict  

# 读取text.jsonl文件，建立文件名到文本的映射  
image_text_map = {}  
with open('/mnt/afs/xueyingyi/image_vague/loc_dection/text.jsonl', 'r') as f:  
    for line in f:  
        data = json.loads(line.strip())  
        image_text_map[data['file_name']] = data['text']  

def process_text(text):  
    """处理文本：转换为小写并用正则提取单词（保留顺序）"""  
    return re.findall(r"\b[\w']+\b", text.lower())  

# 处理检测结果并写入新文件  
output_lines = []  
deleted_detections = []  # 用于存储完全删除的 detection  
partially_deleted_detections = [] # 用于存储部分删除的 detection  

with open('/mnt/afs/xueyingyi/image_vague/loc_dection/dections_Eimage.jsonl', 'r') as f:  
    for line in f:  
        entry = json.loads(line.strip())  
        image_path = entry['image_path']  
        file_name = os.path.basename(image_path)  
        original_text = image_text_map.get(file_name, '')  
        
        # 统计原始文本单词频次  
        original_words = process_text(original_text)  
        word_counts = defaultdict(int)  
        for word in original_words:  
            word_counts[word] += 1  
        
        new_detections = []  
        for det in entry['detections']:  
            det_words = process_text(det['text'])  
            kept_words = []  
            
            # 逐个检查单词是否可用  
            for word in det_words:  
                if word_counts.get(word, 0) > 0:  
                    kept_words.append(word)  
                    word_counts[word] -= 1  
            
            # 保留非空检测结果  
            if kept_words:  
                new_text = ' '.join(kept_words) # 新的文本内容  

                if new_text != det['text']:  
                    # 记录部分删除的 detection 信息，包含 image_path和原始的det信息  
                    partially_deleted_detections.append({  
                        "image_path": image_path,  
                        "original_detection": det,  
                        "new_text": new_text  
                    })  

                new_det = {  
                    "bbox": det["bbox"],  
                    "text": new_text  # 按原顺序拼接小写单词  
                }  
                new_detections.append(new_det)  
            else:  
                # 记录被完全删除的 detection 信息，包含 image_path  
                deleted_detections.append({  
                    "image_path": image_path,  
                    "original_detection": det  
                })  
        
        # 更新检测结果  
        entry['detections'] = new_detections  
        output_lines.append(json.dumps(entry))  

# 写入筛选后的结果  
with open('/mnt/afs/xueyingyi/image_vague/loc_dection/filtered_dections.jsonl', 'w') as f:  
    for line in output_lines:  
        f.write(line + '\n')  

# 写入被完全删除的 detection  
with open('/mnt/afs/xueyingyi/image_vague/loc_dection/deleted_detections.jsonl', 'w') as f:  
    for det in deleted_detections:  
        f.write(json.dumps(det) + '\n')  

# 写入被部分删除的 detection  
with open('/mnt/afs/xueyingyi/image_vague/loc_dection/partially_deleted_detections.jsonl', 'w') as f:  
    for det in partially_deleted_detections:  
        f.write(json.dumps(det) + '\n')