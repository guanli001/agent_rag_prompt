"""
Day 3 实验 2：手写多轮对话
目标：理解 messages 怎么管理，对话历史怎么累积
不使用任何框架，纯 requests
"""
import os
import requests

api_key = os.environ.get("DASHSCOPE_API_KEY")

# 初始化对话历史
messages = [
    {"role": "system", "content": "你是一个友好的Python编程助手，回答简洁明了，每次回答不超过100字"}
]
print("=" * 50)
print("  Python 编程助手（输入 quit 退出）")
print("=" * 50)

while True:
    user_input = input("\n你: ").strip()
    if user_input.lower() == "quit":
        print("byebye")
        break

    # 1. 把用户消息加入历史
    messages.append({"role": "user", "content": user_input})
    # 2. 带着完整历史调用 API
    response = requests.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "qwen3-max",
            "messages": messages,
            "temperature": 0.7
        }
    )
    result = response.json()
    reply = result["choices"][0]["message"]["content"]

    # 3. 把AI回复也加入历史（关键！不加的话AI就不记得了）
    messages.append({"role": "assistant", "content": reply})

    print(f"AI: {reply}")

    # 4. 打印当前状态，感受上下文增长
    total_tokens = result["usage"]["total_tokens"]
    print(f"  [消息数: {len(messages)} | 累计Token: {total_tokens}]")

    # 5. 如果Token快超了，提醒
    if total_tokens > 50000:
        print("  ⚠️ Token快超窗了！后面的对话会丢失前面的内容")
