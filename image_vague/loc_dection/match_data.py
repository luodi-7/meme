import json  
import re  
import os  
from collections import defaultdict  

# 读取text.jsonl文件，建立文件名到文本、elements、word_indices的映射  
image_text_map = {}  
with open('/mnt/afs/xueyingyi/image_vague/loc_dection/text.jsonl', 'r') as f:  
    for line in f:  
        data = json.loads(line.strip())  
        original_text = data['text']  
        file_name = data['file_name']  
        
        # 分割原文本为elements（单词和非单词部分）  
        elements = re.findall(r"\w+|\W+", original_text)  
        
        # 记录每个单词的索引  
        word_indices = []  
        for idx, elem in enumerate(elements):  
            # 检查是否为单词（包含字母或撇号）  
            if re.match(r"^[\w']+$", elem, re.IGNORECASE):  
                word_indices.append(idx)  
        
        image_text_map[file_name] = {  
            'original_text': original_text,  
            'elements': elements,  
            'word_indices': word_indices  
        }  

def process_text(text):  
    """处理文本：转换为小写并用正则提取单词（保留顺序）"""  
    return re.findall(r"\b[\w']+\b", text.lower())  

# 处理检测结果并写入新文件  
output_lines = []  
change_log = []  # 用于记录更改  
with open('/mnt/afs/xueyingyi/image_vague/loc_dection/dections_Eimage.jsonl', 'r') as f:  
    for line in f:  
        entry = json.loads(line.strip())  
        image_path = entry['image_path']  
        file_name = os.path.basename(image_path)  
        
        # 获取对应的原文本信息  
        file_info = image_text_map.get(file_name, {})  
        if not file_info:  
            # 如果找不到对应的原文本，保留原始detections  
            output_lines.append(json.dumps(entry))  
            continue  
        original_text = file_info['original_text']  
        elements = file_info['elements']  
        word_indices = file_info['word_indices']  
        
        # 统计原始文本单词频次  
        original_words = process_text(original_text)  
        word_counts = defaultdict(int)  
        for word in original_words:  
            word_counts[word] += 1  
        
        original_detections = entry['detections']  
        new_detections = []  

        for i, det in enumerate(original_detections):  
            original_det_text = det['text']  
            det_words = process_text(original_det_text)  
            kept_words = []  
            
            # 逐个检查单词是否可用  
            temp_counts = word_counts.copy()  
            for word in det_words:  
                if temp_counts.get(word, 0) > 0:  
                    kept_words.append(word)  
                    temp_counts[word] -= 1  
            
            if not kept_words:  
                # 检测框被删除  
                change_log.append({  
                    'file_name': file_name,  
                    'detection_index': i,  
                    'original_text': original_det_text,  
                    'change_type': 'deleted',  
                    'reason': 'No words left after filtering.'  
                })  
                continue  
            
            # 在word_indices中查找匹配的连续序列  
            n = len(kept_words)  
            found = False  
            match_start = -1  
            for j in range(len(word_indices) - n + 1):  
                match = True  
                for k in range(n):  
                    word_idx = word_indices[j + k]  
                    elem = elements[word_idx]  
                    # 检查元素是否为单词，并且小写匹配  
                    if not re.match(r"^[\w']+$", elem, re.IGNORECASE) or elem.lower() != kept_words[k]:  
                        match = False  
                        break  
                if match:  
                    match_start = j  
                    found = True  
                    break  
            
            if not found:  
                # 未找到匹配，保留过滤后的单词  
                new_det = {  
                    "bbox": det["bbox"],  
                    "text": ' '.join(kept_words)  
                }  
                new_detections.append(new_det)  
                change_log.append({  
                    'file_name': file_name,  
                    'detection_index': i,  
                    'original_text': original_det_text,  
                    'change_type': 'modified',  
                    'new_text': ' '.join(kept_words),  
                    'reason': 'No matching sequence found, kept filtered words.'  
                })  
                continue  
            
            # 找到匹配，计算start_word_idx和end_word_idx  
            start_word_idx = word_indices[match_start]  
            end_word_idx = word_indices[match_start + n - 1]  
            
            # 扩展start_element和end_element以包括周围的非单词字符  
            start_element = start_word_idx - 1  
            while start_element >= 0 and not re.match(r"^[\w']+$", elements[start_element], re.IGNORECASE):  
                start_element -= 1  
            start_element += 1  
            
            end_element = end_word_idx + 1  
            while end_element < len(elements) and not re.match(r"^[\w']+$", elements[end_element], re.IGNORECASE):  
                end_element += 1  
            end_element -= 1  
            
            # 确保索引不越界  
            start_element = max(0, start_element)  
            end_element = min(len(elements) - 1, end_element)  
            
            # 提取短语元素并合并  
            phrase_elements = elements[start_element:end_element + 1]  
            original_phrase = ''.join(phrase_elements).strip()  
            
            new_det = {  
                "bbox": det["bbox"],  
                "text": original_phrase  
            }  
            new_detections.append(new_det)  
            change_log.append({  
                'file_name': file_name,  
                'detection_index': i,  
                'original_text': original_det_text,  
                'change_type': 'modified',  
                'new_text': original_phrase,  
                'reason': 'Found matching sequence, expanded to surrounding non-word characters.'  
            })  
        
        # 更新检测结果  
        entry['detections'] = new_detections  
        output_lines.append(json.dumps(entry))  

# 写入筛选后的结果  
with open('/mnt/afs/xueyingyi/image_vague/loc_dection/filtered_dections.jsonl', 'w') as f:  
    for line in output_lines:  
        f.write(line + '\n')  

# 写入更改日志  
with open('/mnt/afs/xueyingyi/image_vague/loc_dection/change_log.jsonl', 'w') as f:  
    for line in change_log:  
        f.write(json.dumps(line) + '\n')