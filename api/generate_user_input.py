import csv
import json
from openai import OpenAI

# DeepSeek API调用函数
def call_api(prompt, system_prompt, frequency_penalty=0, presence_penalty=0):
    client = OpenAI(
        api_key='sk-514a633560104439a4324dc30deab907',  # 替换为你的API Key
        base_url="https://api.deepseek.com"
    )
    if 'json' in system_prompt:
        type = 'json_object'
    else:
        type = 'text'
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        response_format={
            'type': type
        },
        max_tokens=4096,
        temperature=0.5,
        stream=False,  # 默认值false，表示一次性返回完整的响应
        frequency_penalty=frequency_penalty,  # 默认值'0'，数值越大，模型越倾向于避免生成高频词
        presence_penalty=presence_penalty,  # 默认值'0'，数值越大，模型越倾向于避免重复已经生成的词
        top_p=1,  # 默认值'1'，表示不使用核采样策略，考虑所有可能的词
    )
    content = response.choices[0].message.content
    return content

# 构造提示词，用于推断用户输入
def construct_prompt(row):
    # 从CSV行中提取信息
    file_name = row['file_name']
    sentiment_category = row['sentiment category']
    sentiment_degree = row['sentiment degree']
    intention_detection = row['intention detection']
    offensiveness_detection = row['offensiveness detection']
    metaphor_occurrence = row['metaphor occurrence']
    metaphor_category = row['metaphor category']
    target_domain = row['target domain']
    source_domain = row['source domain']
    target_modality = row['target modality']
    source_modality = row['source modality']
    text = row['text']

    # 构造提示词
    prompt = f"""
    You are an assistant that helps infer the user input for generating a meme based on the following information:
    - File Name: {file_name}
    - Sentiment Category: {sentiment_category}
    - Sentiment Degree: {sentiment_degree}
    - Intention Detection: {intention_detection}
    - Offensiveness Detection: {offensiveness_detection}
    - Metaphor Occurrence: {metaphor_occurrence}
    - Metaphor Category: {metaphor_category}
    - Target Domain: {target_domain}
    - Source Domain: {source_domain}
    - Target Modality: {target_modality}
    - Source Modality: {source_modality}
    - Text Content: {text}

    Based on the above information, infer the user input that could have been provided to generate this meme. The user input should include:
    - Emotion Category (情感类别)
    - Emotion Intensity (情感强度)
    - Intention Category (意图类别)
    - Scene or Theme (场景或主题)
    - Style Preference (风格偏好)
    - Style Intensity (风格强度)
    - Text Content Keywords (文字内容关键词)

    Return the user input in the following JSON format:
    {{
        "file_name": "{file_name}",
        "user_input": {{
            "Emotion Category": "...",
            "Emotion Intensity": "...",
            "Intention Category": "...",
            "Scene or Theme": "...",
            "Style Preference": "...",
            "Style Intensity": "...",
            "Text Content Keywords": ["...", "..."]
        }}
    }}
    """
    return prompt

# 处理CSV文件，生成JSONL文件
def process_csv(input_csv_path, output_jsonl_path):
    with open(input_csv_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        with open(output_jsonl_path, mode='w', encoding='utf-8') as jsonl_file:
            for row in csv_reader:
                # 构造提示词
                prompt = construct_prompt(row)
                # 系统提示词
                system_prompt = "You are an assistant that helps infer the user input for generating memes. Based on the provided meme information, infer the user input that could have been used to generate this meme."
                # 调用API生成用户输入
                user_input_description = call_api(prompt, system_prompt)
                # 将结果写入JSONL文件
                jsonl_file.write(user_input_description + '\n')

# 执行代码
input_csv_path = '/mnt/afs/xueyingyi/meme/generate/generate_label.csv'
output_jsonl_path = '/mnt/afs/xueyingyi/meme/generate/user_input_descriptions.jsonl'
process_csv(input_csv_path, output_jsonl_path)