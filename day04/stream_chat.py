"""
Day 4 实验 2：流式多轮对话
目标：把昨天的多轮对话改成流式输出
"""
import os
import requests
import json

api_key = os.environ.get("DASHSCOPE_API_KEY")

messages = [
    {"role": "system", "content": "你是一个友好的Python编程助手，回答简洁明了"}
]

print("=" * 50)
print("  流式多轮对话（输入 quit 退出）")
print("=" * 50)

while True:
    user_input = input("\n你: ").strip()
    if user_input.lower() == "quit":
        break

    messages.append({"role": "user", "content": user_input})

    # 流式调用
    response = requests.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "qwen3-max",
            "messages": messages,
            "temperature": 0.7,
            "stream": True
        },
        stream=True
    )

    print("AI: ", end="", flush=True)
    full_reply = ""

    for line in response.iter_lines():
        if not line:
            continue

        line_str = line.decode("utf-8")
        if line_str.startswith("data: "):
            data = line_str[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                print(delta, end="", flush=True)
                full_reply += delta

    print()  # 换行

    # 把完整回复加入历史
    messages.append({"role": "assistant", "content": full_reply})
    print(f"  [消息数: {len(messages)}]")