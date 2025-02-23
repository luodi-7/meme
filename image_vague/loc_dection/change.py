import csv  
import json  

def csv_to_jsonl(csv_path, jsonl_path):  
    """  
    将CSV文件转换为JSONL文件。  

    Args:  
        csv_path (str): CSV文件的路径。  
        jsonl_path (str): JSONL文件的路径。  
    """  
    with open(csv_path, 'r', encoding='utf-8') as csvfile, \
            open(jsonl_path, 'w', encoding='utf-8') as jsonlfile:  
        reader = csv.DictReader(csvfile)  
        for row in reader:  
            json.dump(row, jsonlfile, ensure_ascii=False)  # 避免中文乱码  
            jsonlfile.write('\n') # JSONL 格式要求每行一个 JSON 对象  

# 示例用法  
csv_file = '/mnt/afs/xueyingyi/meme/generate/E_text_1.csv'  # 替换为你的CSV文件路径  
jsonl_file = '/mnt/afs/xueyingyi/image_vague/loc_dection/text.jsonl' # 替换为你想要保存的JSONL文件路径  
csv_to_jsonl(csv_file, jsonl_file)
