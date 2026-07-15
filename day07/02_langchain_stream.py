"""
Day 7 任务 2：LangChain 流式输出
对比你 Day 4 手写的 SSE 解析
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 通义千问兼容接口
    temperature=0.7
)

# ===== LangChain 流式输出 =====
print("=" * 60)
print("  LangChain 流式输出")
print("=" * 60)
print("  开始生成...\n")

# 只要 .stream() 替代 .invoke()，就是流式
for chunk in llm.stream([HumanMessage(content="用100字解释什么是人工智能")]):
    print(chunk.content, end="", flush=True)

print("\n\n" + "=" * 60)
print("  对比：你 Day 4 手写的流式")
print("=" * 60)
print("""
# 你 Day 4 手写的（约 20 行）：
response = requests.post(url, headers=headers, json={..., "stream": True}, stream=True)
for line in response.iter_lines():
    if line:
        chunk = line.decode("utf-8")
        if chunk.startswith("data: "):
            data = chunk[6:]
            if data == "[DONE]":
                break
            chunk_data = json.loads(data)
            delta = chunk_data["choices"][0]["delta"].get("content", "")
            print(delta, end="", flush=True)

# LangChain 写的（2 行）：
for chunk in llm.stream([HumanMessage(content="...")]):
    print(chunk.content, end="", flush=True)
""")

print("  LangChain 帮你省了什么？")
print("  1. 不用手动解析 data: 前缀")
print("  2. 不用手动判断 [DONE]")
print("  3. 不用手动 json.loads 每个 chunk")
print("  4. 不用手动从 delta 里取 content")
print("  5. 自动处理连接断开、重连")
