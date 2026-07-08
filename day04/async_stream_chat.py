"""
Day 4 实验 4：异步流式多轮对话
目标：把流式 + 异步 + 多轮对话三个技术合在一起
这是后面封装 LLMClient 的基础
"""
import os
import asyncio
import httpx
import json

api_key = os.environ.get("DASHSCOPE_API_KEY")


class AsyncChatClient:
    """异步聊天客户端，支持流式和非流式"""

    def __init__(self, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def chat(self, messages, model="qwen-turbo", temperature=0.7):
        """非流式调用"""
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def chat_stream(self, messages, model="qwen-turbo", temperature=0.7):
        """流式调用，逐字 yield"""
        async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": True
                }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta

    async def close(self):
        """关闭连接"""
        await self.client.aclose()


async def main():
    client = AsyncChatClient(api_key)
    messages = [
        {"role": "system", "content": "你是一个Python编程助手，回答简洁明了"}
    ]
    print("=" * 50)
    print("  异步流式多轮对话（输入 quit 退出）")
    print("=" * 50)

    try:
        while True:
            user_input = input("  用户：")
            if user_input.lower() == "quit":
                break
            messages.append({"role": "user", "content": user_input})
            print("AI: ", end="", flush=True)
            full_reply = ""
            async for delta in client.chat_stream(messages):
                print(delta, end="", flush=True)
                full_reply += delta
            print()

            messages.append({"role": "assistant", "content": delta})
    finally:
        await client.close()


asyncio.run(main())
