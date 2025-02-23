import csv
import json
from openai import OpenAI
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# 加载图像描述生成模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 生成图像描述
def generate_image_description(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

# DeepSeek API call function
def call_api(prompt, system_prompt, frequency_penalty=0, presence_penalty=0):
    client = OpenAI(
        api_key='sk-514a633560104439a4324dc30deab907',  # Replace with your API Key
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
        stream=False,  # Default is False, meaning the complete response is returned at once
        frequency_penalty=frequency_penalty,  # Default is '0', higher values make the model avoid high-frequency words
        presence_penalty=presence_penalty,  # Default is '0', higher values make the model avoid repeating words
        top_p=1,  # Default is '1', meaning no nucleus sampling, all possible words are considered
    )
    content = response.choices[0].message.content
    return content

# User input options (aligned with your training labels)
EMOTION_CATEGORY_OPTIONS = ["happiness", "love", "anger", "sorrow", "fear", "hate", "surprise"]
EMOTION_INTENSITY_OPTIONS = ["slightly", "moderate", "very"]
INTENTION_CATEGORY_OPTIONS = ["humor", "sarcasm", "encouragement", "rant", "self-mockery", "expression of love", "expression of dissatisfaction"]
SCENE_OR_THEME_OPTIONS = [
    "daily life", "work", "study", "romance", "friendship", "family", "gaming", "sports", "travel", "food",
    "socializing", "festivals", "pets", "health", "technology", "entertainment", "culture", "social phenomena", "self-mockery", "rant"
]
STYLE_PREFERENCE_OPTIONS = ["funny", "wholesome", "dark", "sarcastic", "motivational", "romantic"]

# Construct prompt to infer user input
def construct_prompt(row, image_description):
    # Extract information from the CSV row
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

    # Construct the prompt
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
    - Image Description: {image_description}

    Based on the above information, infer the user input that could have been provided to generate this meme. The user input should include:
    - Emotion Category: Choose from {EMOTION_CATEGORY_OPTIONS}
    - Emotion Intensity: Choose from {EMOTION_INTENSITY_OPTIONS}
    - Intention Category: Choose from {INTENTION_CATEGORY_OPTIONS}
    - Scene or Theme: Choose 1-2 from {SCENE_OR_THEME_OPTIONS}
    - Style Preference: Choose from {STYLE_PREFERENCE_OPTIONS}
    - Text Content Keywords: Extract 2-3 keywords that reflect the core meaning of the text and other information. Do not directly copy words from the text.

    Return the user input in the following JSON format:
    {{
        "file_name": "{file_name}",
        "user_input": {{
            "Emotion Category": "...",
            "Emotion Intensity": "...",
            "Intention Category": "...",
            "Scene or Theme": ["...", "..."],
            "Style Preference": "...",
            "Text Content Keywords": ["...", "..."]
        }}
    }}
    """
    return prompt

# Process CSV file and generate JSONL file
def process_csv(input_csv_path, output_jsonl_path):
    with open(input_csv_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        with open(output_jsonl_path, mode='w', encoding='utf-8') as jsonl_file:
            for row in csv_reader:
                # 生成图像描述
                file_name = row['file_name']
                image_path = f"/mnt/afs/niuyazhe/data/lister/meme/quickmeme_images/{file_name}"
                image_description = generate_image_description(image_path)

                # 构造 prompt
                prompt = construct_prompt(row, image_description)

                # System prompt
                system_prompt = "You are an assistant that helps infer the user input for generating memes. Based on the provided meme information, infer the user input that could have been used to generate this meme."

                # Call the API to generate user input
                user_input_description = call_api(prompt, system_prompt)

                # Write the result to the JSONL file
                jsonl_file.write(user_input_description + '\n')

# Execute the code
input_csv_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/merged_output.csv'
output_jsonl_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/user_input_descriptions_with_image.jsonl'
process_csv(input_csv_path, output_jsonl_path)