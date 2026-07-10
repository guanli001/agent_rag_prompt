"""
Day 5 实验 4：用 LLMClient 实现多轮对话
体会封装的好处——复杂逻辑都藏在 LLMClient 里了
"""
import os
import asyncio
from llm_client import LLMClient


async def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    client = LLMClient(api_key=api_key, max_concurrent=3)

    messages = [
        {"role": "system", "content": "你是一个Python编程助手，回答简洁明了"}
    ]

    print("=" * 55)
    print("  LLMClient 流式多轮对话（输入 quit 退出）")
    print("=" * 55)

    try:
        while True:
            user_input = input("\n你: ").strip()
            if user_input.lower() == "quit":
                break

            messages.append({"role": "user", "content": user_input})

            print("AI: ", end="", flush=True)
            full_reply = ""
            async for delta in client.chat_stream(messages):
                print(delta, end="", flush=True)
                full_reply += delta
            print()

            messages.append({"role": "assistant", "content": full_reply})
            print(f"  [消息数: {len(messages)}]")
    finally:
        await client.close()


asyncio.run(main())
