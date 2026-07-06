import os
from openai import OpenAI

# 初始化客户端（新版必须这样做）
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    # 如果用官方 OpenAI，不需要 base_url
    # 如果用国内中转（如通义千问），取消下面注释：
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def get_completion(prompt, model="qwen3-max"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


def get_completion_from_messages(messages, model="qwen3-max", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    # print(response.choices[0].message)
    return response.choices[0].message.content


messages = [
    {'role': 'system', 'content': '你是一个像莎士比亚一样说话的助手。'},
    {'role': 'user', 'content': '给我讲个笑话'},
    {'role': 'assistant', 'content': '鸡为什么过马路'},
    {'role': 'user', 'content': '我不知道'}]

response = get_completion_from_messages(messages, temperature=1)
print(response)
