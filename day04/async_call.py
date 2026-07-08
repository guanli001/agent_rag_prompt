"""
Day 4 实验 3：异步 API 调用
目标：用 asyncio + httpx 替代 requests，实现异步调用

- `async def` 定义异步函数
- `await` 等待异步操作完成
- `asyncio.gather(*tasks)` 同时执行多个异步任务
- `async with httpx.AsyncClient()` 异步 HTTP 客户端

**核心对比**：同步 3 个问题 = 3 倍时间，异步 3 个问题 ≈ 1 倍时间（并行）。

"""
import os
import asyncio
import httpx
import time

api_key = os.environ.get("DASHSCOPE_API_KEY")


async def chat_async(messages, model="qwen3-max", temperature=0.7):
    """异步调用"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "api_key": api_key
            },
            timeout=30.0
        )
        data = response.json()
        return {
            "content": data["choices"][0]["message"]["content"],
            "tokens": data["usage"]["total_tokens"]
        }


# ===== 实验 1：单个异步调用 =====
async def test_single():
    print("=" * 50)
    print("  实验 1：单个异步调用")
    print("=" * 50)

    result = await chat_async([{"role": "user", "content": "什么是递归？用一句话解释"}])
    print(f"回答: {result['content']}")
    print(f"Token: {result['tokens']}")


# ===== 实验 2：同步 vs 异步对比 =====
async def test_compare():
    print("\n" + "=" * 50)
    print("  实验 2：同步 vs 异步（3个问题）")
    print("=" * 50)

    questions = [
        "什么是Python装饰器？一句话解释",
        "什么是闭包？一句话解释",
        "什么是生成器？一句话解释"
    ]

    # --- 同步：一个一个问 ---
    print("\n--- 同步（一个一个问）---")
    start = time.time()
    sync_results = []
    for q in questions:
        result = await chat_async([{"role": "user", "content": q}])
        sync_results.append(result)
        print(f"  Q: {q}")
        print(f"  A: {result['content'][:50]}...\n")
    sync_time = time.time() - start
    print(f"同步总耗时: {sync_time:.2f}s")

    # --- 异步：同时问 ---
    print("\n--- 异步（同时问3个）---")
    start = time.time()
    tasks = [
        chat_async([{"role": "user", "content": q}])
        for q in questions
    ]
    async_results = await asyncio.gather(*tasks)
    for q, r in zip(questions, async_results):
        print(f"  Q: {q}")
        print(f"  A: {r['content'][:50]}...\n")
    async_time = time.time() - start
    print(f"异步总耗时: {async_time:.2f}s")

    # --- 对比 ---
    print(f"\n--- 对比 ---")
    print(f"同步: {sync_time:.2f}s")
    print(f"异步: {async_time:.2f}s")
    print(f"加速比: {sync_time / async_time:.1f}x")


# ===== 实验 3：异步流式输出 =====
async def test_async_stream():
    print("\n" + "=" * 50)
    print("  实验 3：异步流式输出")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        async with client.stream(
                "POST",
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen-turbo",
                    "messages": [{"role": "user", "content": "用3句话介绍Python"}],
                    "stream": True
                }
        ) as response:
            print("AI: ", end="", flush=True)
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = __import__("json").loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
    print()


# 运行所有实验
async def main():
    await test_single()
    await test_compare()
    await test_async_stream()


asyncio.run(main())
