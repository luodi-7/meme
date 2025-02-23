import os  
from volcenginesdkarkruntime import Ark  
import json  

def filter_detections(origin_text, detections):  
    # 构造系统提示词和示例  
    system_prompt = """你是一个专业的文本匹配处理器，需要根据原始文本内容筛选和修正OCR检测结果。请严格遵循以下规则：  
1. 比较每个检测框中的text是否出现在原始文本中（不区分大小写和换行符）,注意也要考虑单词部分字母识别错误的模糊匹配比如原句是"or we play..."假如识别成了"on we play..."则这个理应是匹配的，只是有个别识别错误的字母。  
2. 如果存在匹配项，保留该检测框，并将text替换为原始文本中的准确样式以及正确单词（保留原始大小写和标点）  
3. 完全删除没有匹配项的检测框  
4. 始终返回目标格式，不要添加任何解释  

示例输入1：  
origin_text: "That moment after you throw up and your friend asks you \"YOU GOOD BRO?\" I'M FUCKIN LIT\n"  
detections: [{'bbox': [1, 0, 151, 496], 'text': 'that moment after you throw up and your friend asks you "you good bro?"'}, {'bbox': [417, 138, 470, 373], 'text': "i'm fuckin lit"}, {'bbox': [481, 407, 499, 499], 'text': 'irunny.co'}]  

示例输出1：  
[{'bbox': [1, 0, 151, 496], 'text': 'That moment after you throw up and your friend asks you \"YOU GOOD BRO?\"'}, {'bbox': [417, 138, 470, 373], 'text': "I\'M FUCKIN LIT"}]  

示例输入2：  
origin_text: "me\nfood at a potluck"  
detections: [{"bbox": [212, 103, 243, 164], "text": "good at a potluck"}, {"bbox": [131, 55, 146, 79], "text": "me"}]  

示例输出2：  
[{"bbox": [212, 103, 243, 164], "text": "food at a potluck"}, {"bbox": [131, 55, 146, 79], "text": "me"}]  

"""  

    # 以华北 2 (北京) 为例，<ARK_BASE_URL> 处应改为 https://ark.cn-beijing.volces.com/api/v3  
    client = Ark(  
        api_key="<api-key>",  
        base_url="https://ark.cn-beijing.volces.com/api/v3",  
        timeout=1800,  
    )  

    # 构造用户提示词  
    user_prompt = f"""origin_text: {json.dumps(origin_text)}  
detections: {json.dumps(detections)}  

请根据规则处理上述检测结果，直接返回处理后的JSON数组："""  

    response = client.chat.completions.create(  
        model="doubao-1-5-pro-32k-250115",  
        messages=[  
            {"role": "system", "content": system_prompt},  
            {"role": "user", "content": user_prompt},  
        ],  
        response_format={"type": "json_object"},  
        temperature=0.1,  
        max_tokens=2048  
    )  
    
    try:  
        print(response.choices[0].message.content)  
        return json.loads(response.choices[0].message.content)  
    except:  
        # 异常处理  
        print("error")  
        return []  

def process_files(image_detections_file, text_file, output_file):  
    # 读取text文件  
    with open(text_file, 'r') as f:  
        texts = [json.loads(line) for line in f]  

    # 创建一个字典，以file_name为键，text为值  
    text_dict = {item['file_name']: item['text'] for item in texts}  

    # 打开输出文件准备写入  
    with open(output_file, 'w') as output_f:  
        # 读取image_detections文件并逐行处理  
        with open(image_detections_file, 'r') as input_f:  
            for line in input_f:  
                item = json.loads(line)  
                file_name = item['image_path'].split('/')[-1]  
                if file_name in text_dict:  
                    origin_text = text_dict[file_name]  
                    detections = item['detections']  
                    filtered_detections = filter_detections(origin_text, detections)  
                    item['detections'] = filtered_detections  
                
                # 将处理后的结果写入到输出文件  
                output_f.write(json.dumps(item) + '\n')

# 文件路径  
image_detections_file = '/mnt/afs/xueyingyi/image_vague/loc_dection/select.jsonl'  
text_file = '/mnt/afs/xueyingyi/image_vague/loc_dection/text.jsonl'  
output_file = '/mnt/afs/xueyingyi/image_vague/loc_dection/filtered_dections.jsonl'  

# 处理文件  
process_files(image_detections_file, text_file, output_file)