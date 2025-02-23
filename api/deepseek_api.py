from openai import OpenAI
def call_api(prompt, system_prompt, frequency_penalty=0, presence_penalty=0):
    client = OpenAI(
        api_key='sk-514a633560104439a4324dc30deab907',
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
        frequency_penalty=frequency_penalty,  # 默认值'0'，数值越大，模型越倾向于避免生成高频词，从而增加生成文本的多样性。
        presence_penalty=presence_penalty,  # 默认值'0'，数值越大，模型越倾向于避免重复已经生成的词
        top_p=1,  # 默认值'1'，表示不使用核采样策略，考虑所有可能的词。
    )
    content = response.choices[0].message.content
    return content