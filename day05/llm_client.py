"""
Day 5 实验 3：封装生产级 LLMClient
整合：异步调用 + 流式输出 + 并发控制 + 多模型Fallback + 重试 + 错误处理
这是第1层地基的最终产出，后面所有阶段都基于它
"""
import os
import asyncio
import httpx
import json
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional


@dataclass
class LLMResponse:
    """LLM 响应数据结构"""
    content: str  # 回复内容
    model: str  # 使用的模型
    prompt_tokens: int = 0  # 输入 Token
    completion_tokens: int = 0  # 输出 Token
    total_tokens: int = 0  # 总 Token
    elapsed: float = 0.0  # 耗时（秒）
    used_fallback: bool = False  # 是否用了备用模型


class LLMClient:
    """
    生产级 LLM 调用客户端
    特性：异步、流式、并发控制、多模型Fallback、自动重试、错误处理
    """

    def __init__(
            self,
            api_key: str,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            models: list = None,
            max_concurrent: int = 5,
            timeout: float = 30.0,
            max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.models = models or ["qwen-turbo", "qwen-plus", "qwen-max"]
        self.max_retries = max_retries
        self.client = httpx.AsyncClient(timeout=timeout)
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def chat(
            self,
            messages: list,
            model: str = None,
            temperature: float = 0.7,
            max_tokens: int = None,

    ) -> LLMResponse:
        """
        非流式调用（带 Fallback + 重试）
        """
        model_chain = [model] if model else self.models
        errors = []
        for model_name in model_chain:
            for attempt in range(self.max_retries):
                try:
                    async with self.semaphore:
                        start_time = time.time()
                        body = {
                            "model": model_name,
                            "messages": messages,
                            "temperature": temperature,
                        }
                        if max_tokens:
                            body["max_tokens"] = max_tokens

                        response = await self.client.post(
                            f"{self.base_url}/chat/completions",
                            headers={"Authorization": f"Bearer {self.api_key}"},
                            json=body,
                        )
                        data = response.json()

                        if "choices" not in data:
                            raise Exception(f"模型返回异常: {data.get('error', data)}")
                        elapsed = time.time() - start_time
                        return LLMResponse(
                            content=data["choices"][0]["message"]["content"],
                            model=model_name,
                            prompt_tokens=data["usage"]["prompt_tokens"],
                            completion_tokens=data["usage"]["completion_tokens"],
                            total_tokens=data["usage"]["total_tokens"],
                            elapsed=elapsed,
                            used_fallback=(model_name != model_chain[0]),
                        )
                except httpx.TimeoutException:
                    print(f"  ⚠️ {model_name} 超时（第{attempt + 1}次重试）")
                    await asyncio.sleep(2 ** attempt)  # 指数退避
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        print(f"  ⚠️ {model_name} 频率限制（第{attempt + 1}次重试）")
                        await asyncio.sleep(5)
                    else:
                        print(f"  ⚠️ {model_name} http错误:{e.response.status_code}")
                        break
                except Exception as e:
                    print(f"  ⚠️ {model_name} 错误:{e}")
                    break

            errors.append(f"{model_name}:重试{self.max_retries}次后失败")

        raise Exception(f"所有模型都失败了: {errors}")

    async def chat_stream(
            self,
            messages: list,
            model: str = None,
            temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        流式调用（逐字 yield）
        注意：流式模式不自动 Fallback（因为部分内容已输出）
        """
        model_name = model or self.models[0]
        async with self.semaphore:
            async with self.client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": model_name,
                        "messages": messages,
                        "temperature": temperature,
                        "stream": True,
                    },
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta

    async def chat_batch(
            self,
            messages_list: list,
            temperature: float = 0.7,
    ) -> list:
        """
        批量并发调用（自动控制并发数）
        传入多个 messages，同时调用，返回结果列表
        """
        task = [self.chat(msgs, temperature=temperature) for msgs in messages_list]
        return await asyncio.gather(*task)

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()


# ===== 测试 =====
async def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    client = LLMClient(
        api_key=api_key,
        models=["qwen-turbo", "qwen-plus"],  # Fallback 链
        max_concurrent=3,
        max_retries=2
    )

    # 测试 1：普通调用
    print("=" * 55)
    print("  测试 1：普通调用")
    print("=" * 55)
    result = await client.chat([
        {"role": "system", "content": "你是一个Python助手，回答简洁"},
        {"role": "user", "content": "什么是装饰器？一句话解释"}
    ])
    print(f"  回答: {result.content}")
    print(f"  模型: {result.model}")
    print(f"  Token: {result.total_tokens}")
    print(f"  耗时: {result.elapsed:.2f}s")
    print(f"  用了Fallback: {result.used_fallback}")

    # 测试 2：流式调用
    print("\n" + "=" * 55)
    print("  测试 2：流式调用")
    print("=" * 55)
    print("  AI: ", end="", flush=True)
    async for delta in client.chat_stream([
        {"role": "user", "content": "用3句话介绍Python"}
    ]):
        print(delta, end="", flush=True)
    print()

    # 测试 3：批量并发调用
    print("\n" + "=" * 55)
    print("  测试 3：批量并发调用（3个问题同时问）")
    print("=" * 55)
    start = time.time()
    results = await client.chat_batch([
        [{"role": "user", "content": "什么是变量？一句话"}],
        [{"role": "user", "content": "什么是函数？一句话"}],
        [{"role": "user", "content": "什么是循环？一句话"}],
    ])
    elapsed = time.time() - start
    for r in results:
        print(f"  [{r.model}] {r.content}")
    print(f"\n  3个请求总耗时: {elapsed:.2f}s")

    await client.close()
    print("\n  ✅ LLMClient 测试完毕，这个类后面所有阶段都会用到！")

if __name__ == "__main__":
    asyncio.run(main())
