import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ===== 初始化模型 =====
# 对比你 Day 3 手写的：
#   response = requests.post(url, headers=headers, json={...})
# LangChain 只要一行：
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 通义千问兼容接口
    temperature=0.7
)

# ===== 最简单的调用 =====
print("=" * 60)
print("  实验 1：最简单的调用")
print("=" * 60)

response = llm.invoke([HumanMessage(content="你好，用一句话介绍你自己")])
print(f"  回复: {response.content}")
print(f"  Token 使用: {response.usage_metadata}")

# ===== 带 system 角色的调用 =====
print("\n" + "=" * 60)
print("  实验 2：带 system 角色")
print("=" * 60)

messages = [
    SystemMessage(content="你是一个毒舌但专业的编程老师，回答不超过50字"),
    HumanMessage(content="什么是递归？")
]

response = llm.invoke(messages)
print(f"  回复: {response.content}")

# ===== 对比：你 Day 3 手写的等价代码 =====
print("\n" + "=" * 60)
print("  对比：你手写的等价代码")
print("=" * 60)
print("""
# 你 Day 3 手写的（约 15 行）：
import requests
url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
headers = {"Authorization": f"Bearer {api_key}"}
data = {
    "model": "qwen-turbo",
    "messages": [
        {"role": "system", "content": "你是一个..."},
        {"role": "user", "content": "你好"}
    ],
    "temperature": 0.7
}
response = requests.post(url, headers=headers, json=data)
result = response.json()
content = result["choices"][0]["message"]["content"]

# LangChain 写的（约 3 行）：
llm = ChatOpenAI(model="qwen-turbo", ...)
response = llm.invoke([HumanMessage(content="你好")])
content = response.content
""")

print("  LangChain 帮你省了什么？")
print("  1. 不用手动拼 URL、headers、JSON body")
print("  2. 不用手动解析 response.json()['choices'][0]['message']['content']")
print("  3. 自动处理 Token 计数、错误重试等")
