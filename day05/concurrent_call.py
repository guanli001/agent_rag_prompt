"""
Day 5 实验 1：并发调用 + Semaphore 限流
目标：理解并发控制，对比不同并发数的效果
"""

import os
import asyncio
import httpx
import time

api_key = os.environ.get("DASHSCOPE_API_KEY")


async def chat(client, messages):
    """单个异步调用"""

    response = await client.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "qwen3-max",
            "messages": messages,
            "temperature": 0
        }
    )
    return response.json()["choices"][0]["message"]["content"]


async def chat_with_limit(sem, client, messages):
    """带并发限制的调用"""
    async with sem:  # 获取信号量，超过限制就排队等
        return await chat(client, messages)


async def main():
    # 准备 10 个问题
    questions = [
        "什么是变量？", "什么是函数？", "什么是列表？", "什么是字典？",
        "什么是循环？", "什么是条件判断？", "什么是类？", "什么是继承？",
        "什么是异常处理？", "什么是模块？"
    ]
    async with httpx.AsyncClient(timeout=60.0) as client:
        # ===== 实验 1：不限并发，10个同时发 =====
        print("=" * 55)
        print("  实验 1：不限并发（10个同时发）")
        print("=" * 55)
        start = time.time()
        tasks = [chat(client, [{"role": "user", "content": q}])
                 for q in questions]
        try:
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - start
            for q, r in zip(questions, results):
                print(f"{q}->{r[:30]}...")
            print(f"\n 耗时: {elapsed:.2f}s(10个请求并行)")
        except Exception as e:
            elapsed = time.time() - start
            print(f"\n  出错了: {e}")
            print(f"  耗时: {elapsed:.2f}s（可能被限流了）")
        # ===== 实验 2：限制并发=3，10个排队 =====
        print("\n" + "=" * 55)
        print("  实验 2：Semaphore(3)，最多同时3个")
        print("=" * 55)
        sem = asyncio.Semaphore(3)
        start = time.time()
        tasks = [chat_with_limit(sem, client, [{"role": "user", "content": q}]) for q in questions]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        for q, r in zip(questions, results):
            print(f"  {q} → {r[:30]}...")
        print(f"\n  耗时: {elapsed:.2f}s（10个请求，每次最多3个并行）")

        # ===== 实验 3：限制并发=1，完全串行 =====
        print("\n" + "=" * 55)
        print("  实验 3：Semaphore(1)，完全串行")
        print("=" * 55)
        sem = asyncio.Semaphore(1)
        start = time.time()
        tasks = [chat_with_limit(sem, client, [{"role": "user", "content": q}]) for q in questions]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        print(f"  耗时: {elapsed:.2f}s（10个请求，一个一个跑）")

        # ===== 对比 =====
        print("\n" + "=" * 55)
        print("  对比结论")
        print("=" * 55)
        print("  Semaphore(0) 不限：最快，但可能被API限流(429)")
        print("  Semaphore(3)：安全且较快，生产环境推荐")
        print("  Semaphore(1)：最慢，但绝对不会被限流")

asyncio.run(main())
