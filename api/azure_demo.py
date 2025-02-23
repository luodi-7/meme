import os
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
        - prompt: the system prompt for gpt
        - question: the question to ask
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
                # print(f'Getting chatgpt answer...')
                response = self.client.chat.completions.create(
                    #model="gpt-4-1106-preview",
                    #model="gpt-35-turbo-1106",
                    model=self.model,
                    messages=history,
                    # seed=self.seed,
                    response_format={"type": response_format},
                    temperature=0.7,
                    # max_tokens=4096,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None
                )
                reply = response.choices[0].message.content.strip()
                break
            # except any error
            except Exception as e:
                logging.warning('ERROR: ' + repr(e))
                try_count += 1
                if try_count > 3:
                    return None, total_token_count
                time.sleep(3)
                continue

        return reply, total_token_count


if __name__ == "__main__":
    api_key = os.getenv("LLM_API_KEY")
    model = 'gpt-4o-mini'
    api_endpoint = 'https://codekidz-australia-east.openai.azure.com/'
    handle = AzureCall(api_key, model, api_endpoint)
    print(handle.call([], 'create a name'))
