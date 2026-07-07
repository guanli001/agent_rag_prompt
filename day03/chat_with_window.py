"""
Day 3 实验 3：带滑动窗口的多轮对话
目标：当对话太长时自动截断早期消息，防止Token超窗
"""
import os
import requests

api_key = os.environ.get("DASHSCOPE_API_KEY")

MAX_MESSAGES = 20  # 最多保留20条消息（不含system）
OVERLAP = 3  # 重合量：保留开头 3 条对话消息作为锚点（不包括 system）


def chat_with_window(user_input, messages):
    """带滑动窗口的对话"""
    # 加入用户消息
    messages.append({"role": "user", "content": user_input})

    # 如果消息太多，截断早期对话（保留system + 最近的消息）
    system_msgs = [m for m in messages if m["role"] == "system"]
    chat_msgs = [m for m in messages if m["role"] != "system"]

    if len(chat_msgs) > MAX_MESSAGES:
        # 丢掉最早的消息，只保留最近 MAX_MESSAGES 条
        chat_msgs = chat_msgs[len(chat_msgs) - MAX_MESSAGES - 5:]
        print(f"  [⚠️ 已截断早期 {len(messages) - len(system_msgs) - len(chat_msgs)} 条消息]")

    # 重新组装：system在前 + 截断后的对话
    current_messages = system_msgs + chat_msgs

    # 调用API
    response = requests.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "qwen-turbo",
            "messages": current_messages,
            "temperature": 0.7
        }
    )

    result = response.json()
    reply = result["choices"][0]["message"]["content"]

    # 记住：往原始 messages 里加AI回复（不是current_messages）
    messages.append({"role": "assistant", "content": reply})

    return reply, result["usage"]["total_tokens"]


# 测试
messages = [
    {"role": "system", "content": "你是一个Python编程助手，回答简洁明了"}
]

print("带滑动窗口的对话（输入 quit 退出）")
print(f"窗口大小: {MAX_MESSAGES} 条消息\n")

while True:
    user_input = input("\n你: ").strip()
    if user_input.lower() == "quit":
        break

    reply, tokens = chat_with_window(user_input, messages)
    print(f"AI: {reply}")
    print(f"  [消息数: {len(messages)} | Token: {tokens}]")
