# 1. 导入第三方库
from openai import OpenAI  # 新版导入方式
import os
from dotenv import load_dotenv, find_dotenv

# 2. 读取系统中的环境变量
_ = load_dotenv(find_dotenv())

# 3. 创建客户端（新版需要实例化）
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 通义千问兼容接口
    timeout=60.0,
)


# 4. 封装函数
def get_completion(prompt, model="qwen3-max"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(  # 新版 API
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content  # 注意：是 .content 不是 ["content"]


# 5. 文本内容
text = f"""
你应该提供尽可能清晰、具体的指示，以表达你希望模型执行的任务。\
这将引导模型朝向所需的输出，并降低收到无关或不正确响应的可能性。\
不要将写清晰的提示与写简短的提示混淆。\
在许多情况下，更长的提示可以为模型提供更多的清晰度和上下文信息，从而导致更详细和相关的输出。
"""

# 6. 指令
prompt = f"""
把用三个反引号括起来的文本总结成一句话。
```{text}```
"""

# 7. 输出
response = get_completion(prompt)
print(response)
