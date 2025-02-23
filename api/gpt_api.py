import os
import csv
import json
import openai
import logging
import time
import tiktoken

class AzureCall():

    def __init__(self, api_key, model, api_endpoint):
        self.seed = 42
        self.model = model
        self.client = openai.AzureOpenAI(
            azure_endpoint=api_endpoint,
            api_key=api_key,
            api_version="2024-02-15-preview"
        )

    def call(self, history: list, answer: str) -> str:
        '''
        Call chatgpt api to generate answers according to prompt and question.
        Keep trying until get valid answer.
        Default one round QA.
        Input:
        - history: conversation history
        - answer: user input
        Ouput:
        - answer
        '''
        history.append({"role":"user","content": answer})
        encoding = tiktoken.encoding_for_model(self.model)

        once_token_count = 0
        for dial in history:
            tokens = encoding.encode(dial["content"])
            once_token_count += len(tokens)

        if 'JSON' in history[0]["content"]:
            response_format = "json_object"
        else:
            response_format = "text"

        total_token_count = 0
        try_count = 0
        while True:
            try:
                total_token_count += once_token_count
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=history,
                    response_format={"type": response_format},
                    temperature=0.7,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None
                )
                reply = response.choices[0].message.content.strip()
                break
            except Exception as e:
                logging.warning('ERROR: ' + repr(e))
                try_count += 1
                if try_count > 3:
                    return None, total_token_count
                time.sleep(3)
                continue

        return reply, total_token_count


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
def construct_prompt(row):
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
def process_csv(input_csv_path, output_jsonl_path, api_key, model, api_endpoint):
    handle = AzureCall(api_key, model, api_endpoint)
    
    with open(input_csv_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        with open(output_jsonl_path, mode='w', encoding='utf-8') as jsonl_file:
            for row in csv_reader:
                # Construct the prompt
                prompt = construct_prompt(row)
                # Call the API to generate user input
                history = [{"role": "system", "content": "You are an assistant that helps infer user inputs for generating memes."}]
                history.append({"role": "user", "content": prompt})
                user_input_description, _ = handle.call(history, prompt)
                # Check if the response is empty or invalid
                if not user_input_description:
                    logging.warning(f"Empty response for row {row['file_name']}")
                    continue  # Skip to next row if response is empty
                
                # Check if the response is in valid JSON format
                try:
                    result = json.loads(user_input_description)
                except json.JSONDecodeError as e:
                    logging.warning(f"Invalid JSON for row {row['file_name']}: {e}")
                    continue  # Skip to next row if response is invalid
                
                # Write the result to the JSONL file
                jsonl_file.write(json.dumps(result) + '\n')



# Execute the code
input_csv_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/missing_data_un.csv'
output_jsonl_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/user_input_descriptions_simple_miss.jsonl'
api_key = os.getenv("LLM_API_KEY")
model = 'gpt-4o-mini'
api_endpoint = 'https://codekidz-australia-east.openai.azure.com/'

process_csv(input_csv_path, output_jsonl_path, api_key, model, api_endpoint)
