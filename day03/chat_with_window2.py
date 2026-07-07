"""
Day 3 实验 3：带滑动窗口的多轮对话
目标：当对话太长时自动截断早期消息，防止Token超窗
当消息超过 15 条时，让 AI 把之前的对话总结成一段摘要，用摘要替代原始消息。
原始: [system, user1, ai1, user2, ai2, ..., user15, ai15]
                                    ↓ 超过阈值
压缩后: [system, {role: system, content: "之前对话摘要：用户叫顾安，在学Python..."}, user16, ai16, ...]
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
        # 1. 找出要丢弃的早期消息（除了锚点和最近消息以外的部分）
        kept_early = chat_msgs[:OVERLAP]  # 锚点
        kept_recent = chat_msgs[-(MAX_MESSAGES - OVERLAP):]  # 最近
        to_summarize = chat_msgs[OVERLAP:-(MAX_MESSAGES - OVERLAP)]  # 中间要压缩的部分

        if to_summarize:
            # 2. 调用 AI 生成摘要
            summary_response = requests.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "qwen-turbo",
                    "messages": [
                        {"role": "system",
                         "content": "请用一段话（不超过100字）总结以下对话的核心内容，包括用户的主要问题和已给出的答复。"},
                        {"role": "user", "content": str(to_summarize)}
                    ],
                    "temperature": 0.3
                }
            )
            summary = summary_response.json()["choices"][0]["message"]["content"]

            # 3. 用摘要替换被丢掉的消息（作为一条 system 消息插入）
            summary_msg = {"role": "system", "content": f"【历史摘要】{summary}"}
            chat_msgs = kept_early + [summary_msg] + kept_recent
            print(f"  [📝 已将中间 {len(to_summarize)} 条消息压缩为摘要]")
        else:
            chat_msgs = kept_early + kept_recent
            print(f"  [⚠️ 截断，保留 {OVERLAP} 条锚点 + 最近 {MAX_MESSAGES - OVERLAP} 条]")
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
