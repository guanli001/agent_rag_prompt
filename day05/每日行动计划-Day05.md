# Day 5 行动计划 — 并发调用 + 多模型 Fallback + 封装 LLMClient

> 日期：2026 年 7 月 9 日（周四）
> 预计耗时：4-5 小时
> 前置条件：已完成 Day 4（SSE 流式 + 异步调用）
> 今日目标：封装一个生产级 LLM 调用层，这是第 1 层地基的收尾

---

## 你现在的位置

```
第1层  LLM 通信（API调用、流式、异步）  ← 今天收尾！
第2层  Prompt 工程                      ← 明天 Day 6
第3层  记忆/会话管理
第4层  知识库 RAG                        ← 下周开始
第5层  工具调用
第6层  Agent 编排
第7层  工程化部署
```

今天做完，第 1 层就结束了。你会得到一个自己的 LLM 调用工具类，后面所有代码都基于它。

---

## 任务 1 ★☆☆ 必做：并发调用 + Semaphore 限流（约 1 小时）

### 1.1 为什么要限流？

昨天你学过 `asyncio.gather` 可以同时发多个请求。但如果同时发 100 个请求，API 会直接返回 429（限流）拒绝你。

**Semaphore（信号量）** 就是控制"同时最多跑几个"：

```
Semaphore(3) = 最多同时3个请求在跑，其余排队等
```

### 1.2 跑实验

创建文件 `day05/concurrent_call.py`：

```python
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
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "qwen-turbo",
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
        tasks = [chat(client, [{"role": "user", "content": q}]) for q in questions]
        try:
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - start
            for q, r in zip(questions, results):
                print(f"  {q} → {r[:30]}...")
            print(f"\n  耗时: {elapsed:.2f}s（10个请求并行）")
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
```

### 1.3 思考

- 不限并发时有没有报错？为什么？
- Semaphore(3) 比 Semaphore(1) 快多少倍？为什么不是 3 倍？

---

## 任务 2 ★☆☆ 必做：多模型 Fallback（约 1 小时）

### 2.1 什么是 Fallback？

```
用户提问 → 先调千问 turbo（免费/便宜）
              ↓ 失败了（超时/限流/服务器挂了）
            自动切 → 千问 plus（备用）
              ↓ 也失败了
            自动切 → DeepSeek（兜底）
              ↓ 全挂了
            返回错误提示
```

生产环境中，LLM 服务不可能 100% 可用，必须有兜底方案。

### 2.2 实现 Fallback

创建文件 `day05/fallback.py`：

```python
"""
Day 5 实验 2：多模型 Fallback
目标：主模型失败时自动切换备用模型
"""
import os
import asyncio
import httpx
import time

api_key = os.environ.get("DASHSCOPE_API_KEY")

# 模型列表：按优先级排列
MODEL_CHAIN = [
    {"name": "qwen-turbo", "desc": "免费/快速（主模型）"},
    {"name": "qwen-plus", "desc": "更强/稍慢（备用1）"},
    {"name": "qwen-max", "desc": "最强/最贵（备用2）"},
]

async def chat_with_model(client, model_name, messages, timeout=10):
    """调用指定模型"""
    response = await client.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model_name,
            "messages": messages,
            "temperature": 0.7
        },
        timeout=timeout
    )
    data = response.json()
    if "choices" not in data:
        raise Exception(f"模型返回异常: {data.get('error', data)}")
    return {
        "content": data["choices"][0]["message"]["content"],
        "model": model_name,
        "tokens": data["usage"]["total_tokens"]
    }

async def chat_with_fallback(client, messages):
    """带 Fallback 的调用：按优先级逐个尝试"""
    errors = []
    for model in MODEL_CHAIN:
        try:
            print(f"  尝试 {model['name']}（{model['desc']}）...")
            result = await chat_with_model(client, model["name"], messages)
            print(f"  ✅ {model['name']} 成功！")
            return result
        except Exception as e:
            print(f"  ❌ {model['name']} 失败: {e}")
            errors.append(f"{model['name']}: {e}")

    # 所有模型都失败了
    raise Exception(f"所有模型都失败了: {errors}")

async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:

        # ===== 实验 1：正常调用 =====
        print("=" * 55)
        print("  实验 1：正常 Fallback（主模型应该成功）")
        print("=" * 55)
        result = await chat_with_fallback(
            client,
            [{"role": "user", "content": "什么是递归？一句话解释"}]
        )
        print(f"\n  回答: {result['content']}")
        print(f"  使用模型: {result['model']}")
        print(f"  Token: {result['tokens']}")

        # ===== 实验 2：模拟主模型失败 =====
        print("\n" + "=" * 55)
        print("  实验 2：模拟主模型超时（timeout=0.001）")
        print("=" * 55)

        # 故意设极短的 timeout 让 turbo 超时
        async def chat_with_short_timeout(client, model_name, messages):
            return await chat_with_model(client, model_name, messages, timeout=0.001)

        # 临时替换调用函数
        original_chat = chat_with_model
        call_count = 0
        async def mock_chat(client, model_name, messages, timeout=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 第一次（turbo）故意超时
                raise asyncio.TimeoutError("模拟超时")
            # 后面的正常调用
            return await original_chat(client, model_name, messages)

        # 直接测试：turbo 超时 → 切到 plus
        print("  模拟 qwen-turbo 超时...")
        try:
            result = await chat_with_model(client, "qwen-turbo",
                [{"role": "user", "content": "测试"}], timeout=0.001)
        except Exception as e:
            print(f"  ❌ qwen-turbo 超时: {type(e).__name__}")

        # 切到 plus
        print("  尝试 qwen-plus...")
        result = await chat_with_model(client, "qwen-plus",
            [{"role": "user", "content": "什么是递归？一句话解释"}])
        print(f"  ✅ qwen-plus 成功: {result['content'][:50]}...")

        print("\n  结论：主模型超时后自动切到备用模型，用户无感知")

asyncio.run(main())
```

---

## 任务 3 ★☆☆ 必做：封装 LLMClient（约 2 小时）

### 3.1 把前 4 天学的全部整合

这是今天最重要的任务。把你学的**流式 + 异步 + 并发控制 + Fallback + 错误处理**全部封装到一个类里。

创建文件 `day05/llm_client.py`：

```python
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
    content: str                    # 回复内容
    model: str                      # 使用的模型
    prompt_tokens: int = 0          # 输入 Token
    completion_tokens: int = 0      # 输出 Token
    total_tokens: int = 0           # 总 Token
    elapsed: float = 0.0            # 耗时（秒）
    used_fallback: bool = False     # 是否用了备用模型


class LLMClient:
    """
    生产级 LLM 调用客户端
    特性：异步、流式、并发控制、多模型Fallback、自动重试、错误处理
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        models: list = None,
        max_concurrent: int = 5,
        timeout: float = 30.0,
        max_retries: int = 3
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
        max_tokens: int = None
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
                        start = time.time()
                        body = {
                            "model": model_name,
                            "messages": messages,
                            "temperature": temperature
                        }
                        if max_tokens:
                            body["max_tokens"] = max_tokens

                        response = await self.client.post(
                            f"{self.base_url}/chat/completions",
                            headers={"Authorization": f"Bearer {self.api_key}"},
                            json=body
                        )
                        data = response.json()

                        if "choices" not in data:
                            raise Exception(f"API错误: {data.get('error', data)}")

                        elapsed = time.time() - start
                        return LLMResponse(
                            content=data["choices"][0]["message"]["content"],
                            model=model_name,
                            prompt_tokens=data["usage"]["prompt_tokens"],
                            completion_tokens=data["usage"]["completion_tokens"],
                            total_tokens=data["usage"]["total_tokens"],
                            elapsed=elapsed,
                            used_fallback=(model_name != model_chain[0])
                        )

                except httpx.TimeoutException:
                    print(f"  ⚠️ {model_name} 超时（第{attempt+1}次重试）")
                    await asyncio.sleep(2 ** attempt)  # 指数退避
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        print(f"  ⚠️ {model_name} 被限流，等待后重试...")
                        await asyncio.sleep(5)
                    else:
                        print(f"  ⚠️ {model_name} HTTP错误: {e.response.status_code}")
                        break  # 非429错误不重试，直接换模型
                except Exception as e:
                    print(f"  ⚠️ {model_name} 异常: {e}")
                    break

            errors.append(f"{model_name}: 重试{self.max_retries}次后失败")

        raise Exception(f"所有模型都失败了: {errors}")

    async def chat_stream(
        self,
        messages: list,
        model: str = None,
        temperature: float = 0.7
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

    async def chat_batch(
        self,
        messages_list: list,
        temperature: float = 0.7
    ) -> list:
        """
        批量并发调用（自动控制并发数）
        传入多个 messages，同时调用，返回结果列表
        """
        tasks = [self.chat(msgs, temperature=temperature) for msgs in messages_list]
        return await asyncio.gather(*tasks)

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

asyncio.run(main())
```

---

## 任务 4 ★★☆ 进阶：用 LLMClient 实现多轮对话（约 30 分钟）

用你刚封装的 LLMClient，实现一个流式多轮对话。这次代码量很少，因为复杂逻辑都封装好了：

创建文件 `day05/chat_demo.py`：

```python
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
```

对比 Day 4 的 `async_stream_chat.py`，你会发现：**代码少了一半**，因为 LLMClient 帮你封装了 HTTP 请求、流式解析、连接管理。这就是封装的意义。

---

## 今日笔记要求

创建 `day05/note.md`，用自己的话回答：

1. Semaphore 的作用是什么？不限制并发会有什么问题？
2. 多模型 Fallback 的流程是什么？为什么流式模式不自动 Fallback？
3. 你的 LLMClient 有哪些特性？每个特性解决什么问题？
4. 对比 Day 4 的代码，封装后写多轮对话简单了多少？

---

## 10 道面试题

### Q1：什么是 Semaphore？在 LLM 应用中为什么需要它？

**参考答案**：
Semaphore（信号量）是并发控制工具，限制同时执行的任务数量。

LLM 应用中需要它的原因：
1. **API 限流**：千问/OpenAI 都有 QPS 限制，同时发太多请求会返回 429
2. **资源控制**：每个请求占用内存和网络连接，不限制会内存溢出
3. **公平性**：防止少数用户占用全部资源

```python
sem = asyncio.Semaphore(5)  # 最多同时5个
async with sem:
    await api_call()
```

---

### Q2：多模型 Fallback 的设计要点是什么？

**参考答案**：
1. **模型优先级**：按成本/速度排列，便宜的优先用
2. **失败判定**：超时、HTTP 429/500、网络异常都算失败
3. **流式不 Fallback**：流式输出部分内容已返回用户，切换模型会导致内容不连贯
4. **日志记录**：记录哪个模型失败、失败原因，用于监控
5. **熔断机制**：如果某个模型连续失败 N 次，短时间内不再尝试它

---

### Q3：指数退避（Exponential Backoff）是什么？为什么用指数而不是固定等待？

**参考答案**：
指数退避：每次重试等待时间翻倍。`sleep(2 ** attempt)` → 1s, 2s, 4s, 8s...

用指数而不是固定的原因：
1. **避免雪崩**：如果是服务器过载导致的失败，固定间隔重试会让所有客户端同时重试，加重服务器压力
2. **给服务恢复时间**：指数增长给服务器更多恢复时间
3. **快速放弃不可能成功的请求**：如果第一次就严重超时，后面等待越来越长，避免浪费资源

---

### Q4：LLMClient 中为什么要用 AsyncClient 而不是每次新建 client？

**参考答案**：
`httpx.AsyncClient` 内部维护连接池，复用 TCP 连接（HTTP Keep-Alive）。

- **复用 client**：多个请求共享连接池，避免重复 TCP 握手，性能更好
- **每次新建**：每次请求都要新建 TCP 连接，增加延迟和资源消耗

```python
# 正确：创建一次，复用
client = httpx.AsyncClient()
await client.post(...)  # 复用连接
await client.post(...)  # 复用连接
await client.close()

# 错误：每次新建
async def call():
    async with httpx.AsyncClient() as client:  # 每次新建连接
        await client.post(...)
```

---

### Q5：HTTP 429 错误是什么？应该怎么处理？

**参考答案**：
429 = Too Many Requests，表示请求频率超过 API 限制。

处理方式：
1. **等待后重试**：读取响应头中的 `Retry-After` 字段，等待指定秒数后重试
2. **降低并发**：减小 Semaphore 的值
3. **指数退避**：避免所有请求同时重试
4. **切换模型**：如果当前模型限流，切到备用模型

```python
if response.status_code == 429:
    retry_after = int(response.headers.get("Retry-After", 5))
    await asyncio.sleep(retry_after)
```

---

### Q6：@dataclass 的作用是什么？为什么 LLMResponse 用它？

**参考答案**：
`@dataclass` 是 Python 的装饰器，自动生成 `__init__`、`__repr__`、`__eq__` 等方法。

```python
@dataclass
class LLMResponse:
    content: str
    model: str
    tokens: int = 0
```

等价于手写：
```python
class LLMResponse:
    def __init__(self, content, model, tokens=0):
        self.content = content
        self.model = model
        self.tokens = tokens
```

用 dataclass 的原因：代码更简洁、可读性好、方便添加默认值。LLMResponse 只是数据容器，不需要复杂逻辑，适合用 dataclass。

---

### Q7：流式输出为什么不能自动 Fallback？

**参考答案**：
流式输出时，部分内容已经通过 yield 返回给用户了。如果此时切换模型：
1. **内容不连贯**：模型 A 输出了"Python 是一种"，切换到模型 B 继续输出"面向对象语言"，两个模型的风格可能不一致
2. **上下文丢失**：模型 B 不知道模型 A 之前输出了什么
3. **无法回退**：已经发给用户的内容无法撤回

所以流式模式的策略是：失败就报错，让用户重新提问。非流式模式才自动 Fallback。

---

### Q8：asyncio.gather 中一个任务失败会怎样？怎么处理？

**参考答案**：
默认情况下，`asyncio.gather` 中任意一个任务失败，会抛出异常，但其他任务仍在运行。

处理方式：
```python
# 方式1：return_exceptions=True，失败的返回异常对象而不是抛出
results = await asyncio.gather(*tasks, return_exceptions=True)
for r in results:
    if isinstance(r, Exception):
        print(f"某个任务失败: {r}")
    else:
        print(f"结果: {r}")

# 方式2：每个任务内部 try-except，返回 None 或默认值
async def safe_call(messages):
    try:
        return await client.chat(messages)
    except Exception:
        return None
```

---

### Q9：你的 LLMClient 有哪些可改进的地方？

**参考答案**（考察工程思维）：
1. **Token 预算控制**：调用前估算 Token，超预算时拒绝或压缩
2. **请求日志**：记录每次调用的模型、Token、耗时、成功/失败
3. **缓存**：相同输入直接返回缓存结果，省钱省时间
4. **熔断器**：某个模型连续失败 N 次，短时间内跳过它不浪费重试次数
5. **负载均衡**：多个 API Key 轮换使用，分散请求
6. **超时梯度**：第一次 10s 超时，重试时延长到 20s
7. **速率限制器**：不仅用 Semaphore 控制并发，还按 QPS 限流

---

### Q10：为什么说封装 LLMClient 是"第 1 层地基的收尾"？

**参考答案**：
因为 LLMClient 封装了与 LLM 通信的所有底层细节：
- **HTTP 通信**：不用再关心 URL、Header、请求体格式
- **异步处理**：不用再手写 async/await 逻辑
- **流式解析**：不用再手动解析 SSE 数据
- **错误处理**：不用再手写 try-except 和重试
- **并发控制**：不用再手动管理 Semaphore
- **多模型**：不用再手动写 Fallback 逻辑

后面的 Prompt 工程、RAG、Agent 层，只需要调用 `client.chat()` 或 `client.chat_stream()`，不需要关心底层通信细节。这就是分层架构的意义——每层只关注自己的职责。

---

## 今日完成检查清单

- [ ] 运行 concurrent_call.py，理解 Semaphore 并发控制
- [ ] 运行 fallback.py，理解多模型 Fallback
- [ ] 运行 llm_client.py，测试 LLMClient 的三个方法
- [ ] 运行 chat_demo.py，体验封装后的简洁代码
- [ ] 写完 note.md（4 个问题）
- [ ] 代码 push 到 GitHub（`git add . && git commit -m "day05: 并发控制+Fallback+封装LLMClient" && git push`）
- [ ] 看一遍 10 道面试题，能答出至少 6 道

> 🎉 今天做完，第 1 层（LLM 通信）地基就全部完成了！
> 明天 Day 6：Prompt 工程全套（结构化Prompt + Few-shot + CoT + Pydantic结构化输出 + Prompt管理）
