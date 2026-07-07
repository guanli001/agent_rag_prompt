# Day 4 行动计划 — SSE 流式输出 + 异步 API 调用

> 日期：2026 年 7 月 8 日（周三）
> 预计耗时：4-5 小时
> 前置条件：已完成 Day 3（采样参数 + 多轮对话）
> 今日目标：实现流式输出 + 用 asyncio 异步调用 LLM

---

## 压缩版整体时间线（供参考）

```
第1周（7/6~7/12）：认知校准 + API调用 + Prompt工程
  Day 1 ✅ LLM认知
  Day 2 ✅ Token + 首次API调用
  Day 3 ✅ 采样参数 + 多轮对话
  Day 4 ← 今天：SSE流式 + 异步调用
  Day 5    并发+Fallback+封装LLMClient
  Day 6    Prompt工程全套
  Day 7    对比LangChain + 阶段总结
第2周（7/13~7/19）：RAG 基础+进阶
第3周（7/20~7/26）：Agent 核心
第4周（7/27~8/2）：工程化
第5-6周（8/3~8/16）：项目实战
```

---

## 任务 1 ★☆☆ 必做：理解 SSE 流式输出（约 1.5 小时）

### 1.1 什么是 SSE？为什么要流式？

```
非流式（普通调用）：
用户提问 → 等5秒 → 一次性返回500字 → 用户看到

流式（SSE）：
用户提问 → 第0.1秒返回"你" → 第0.2秒返回"好" → ... → 逐字显示
```

**SSE（Server-Sent Events）** 是一种 HTTP 协议，服务器可以持续推送数据给客户端。

LLM API 的流式输出格式：
```
data: {"choices":[{"delta":{"content":"你"}}]}    ← 第一个字
data: {"choices":[{"delta":{"content":"好"}}]}    ← 第二个字
data: {"choices":[{"delta":{"content":""}}]}      ← 结束
data: [DONE]                                      ← 完毕标记
```

每个 `data:` 开头的行就是一个数据块，`delta.content` 是这次新增的内容。

### 1.2 跑流式输出实验

创建文件 `day04/sse_stream.py`：

```python
"""
Day 4 实验 1：SSE 流式输出
目标：实现逐字打印效果，理解 SSE 数据格式
不使用任何框架，纯 requests
"""
import os
import requests
import json
import time

api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    print("错误：请先设置环境变量 DASHSCOPE_API_KEY")
    exit(1)

print("=" * 50)
print("  实验 1：流式输出（逐字打印）")
print("=" * 50)

# ===== 流式调用 =====
print("\n--- 流式输出 ---")
start_time = time.time()

response = requests.post(
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        "model": "qwen-turbo",
        "messages": [{"role": "user", "content": "写一首关于编程的短诗，4句话"}],
        "stream": True,           # 开启流式
        "temperature": 0.7
    },
    stream=True                    # requests 也要开启 stream
)

full_text = ""
first_token_time = None
chunk_count = 0

for line in response.iter_lines():
    if not line:
        continue

    line_str = line.decode("utf-8")

    # SSE 格式：每行以 "data: " 开头
    if line_str.startswith("data: "):
        data = line_str[6:]        # 去掉 "data: " 前缀

        if data == "[DONE]":
            break

        chunk_data = json.loads(data)
        delta = chunk_data["choices"][0]["delta"].get("content", "")

        if delta:
            if first_token_time is None:
                first_token_time = time.time()
                print(f"  [首个字耗时: {first_token_time - start_time:.2f}s]")
            print(delta, end="", flush=True)   # 逐字打印
            full_text += delta
            chunk_count += 1

total_time = time.time() - start_time
print(f"\n\n--- 统计 ---")
print(f"总耗时: {total_time:.2f}s")
print(f"首字耗时: {first_token_time - start_time:.2f}s")
print(f"接收块数: {chunk_count}")
print(f"总字数: {len(full_text)}")

# ===== 对比：非流式调用 =====
print("\n" + "=" * 50)
print("  对比：非流式调用")
print("=" * 50)

print("\n--- 非流式 ---")
start_time = time.time()

response = requests.post(
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        "model": "qwen-turbo",
        "messages": [{"role": "user", "content": "写一首关于编程的短诗，4句话"}],
        "temperature": 0.7
    }
)

result = response.json()
total_time_no_stream = time.time() - start_time
print(result["choices"][0]["message"]["content"])
print(f"\n--- 统计 ---")
print(f"总耗时: {total_time_no_stream:.2f}s")

# ===== 对比结论 =====
print("\n" + "=" * 50)
print("  对比结论")
print("=" * 50)
print(f"流式首字耗时: {first_token_time - start_time:.2f}s（用户更快看到响应）")
print(f"非流式总耗时: {total_time_no_stream:.2f}s（用户要等全部生成完）")
print(f"流式总耗时: {total_time:.2f}s")
print(f"\n结论：流式输出让用户更快看到第一个字，体验更好")
```

### 1.3 思考

- 流式和非流式，**总耗时**差多少？为什么？
- 流式输出的**首字耗时**为什么比总耗时短很多？
- 什么场景必须用流式？什么场景用非流式就行？

---

## 任务 2 ★☆☆ 必做：流式多轮对话（约 1 小时）

### 2.1 把流式输出和多轮对话结合起来

创建文件 `day04/stream_chat.py`：

```python
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
            "model": "qwen-turbo",
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
```

---

## 任务 3 ★☆☆ 必做：异步 API 调用（约 1.5 小时）

### 3.1 为什么要异步？

```
同步：用户A提问 → 等3秒 → 返回 → 用户B提问 → 等3秒 → 返回
     用户B要等用户A完全结束才能得到响应

异步：用户A提问 → 同时用户B提问 → 两个请求并行等待 → 都返回
     两个用户同时得到响应，不用互相等
```

在生产环境中，多个用户同时使用你的 AI 应用，**必须用异步**，否则一个用户的请求会阻塞所有其他用户。

### 3.2 安装 httpx

```bash
pip install httpx
```

### 3.3 用 asyncio + httpx 异步调用

创建文件 `day04/async_call.py`：

```python
"""
Day 4 实验 3：异步 API 调用
目标：用 asyncio + httpx 替代 requests，实现异步调用
"""
import os
import asyncio
import httpx
import time

api_key = os.environ.get("DASHSCOPE_API_KEY")

async def chat_async(messages, model="qwen-turbo", temperature=0.7):
    """异步调用 LLM"""
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
                "temperature": temperature
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
```

### 3.4 重点理解

- `async def` 定义异步函数
- `await` 等待异步操作完成
- `asyncio.gather(*tasks)` 同时执行多个异步任务
- `async with httpx.AsyncClient()` 异步 HTTP 客户端

**核心对比**：同步 3 个问题 = 3 倍时间，异步 3 个问题 ≈ 1 倍时间（并行）。

---

## 任务 4 ★★☆ 进阶：异步流式多轮对话（约 1 小时）

把任务 2 的多轮对话改成异步版本，用 `httpx.AsyncClient` + `asyncio`：

创建文件 `day04/async_stream_chat.py`：

```python
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
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": model, "messages": messages, "temperature": temperature}
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def chat_stream(self, messages, model="qwen-turbo", temperature=0.7):
        """流式调用，逐字 yield"""
        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": model, "messages": messages, "temperature": temperature, "stream": True}
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
    finally:
        await client.close()

asyncio.run(main())
```

---

## 今日笔记要求

创建 `day04/note.md`，用自己的话回答：

1. SSE 流式输出的原理是什么？数据格式长什么样？
2. 流式和非流式，用户体验有什么区别？总耗时差多少？
3. 为什么要用异步？同步和异步在并发 3 个请求时耗时差多少？为什么？
4. `async def`、`await`、`asyncio.gather` 分别干什么？

---

## 10 道面试题

### Q1：什么是 SSE？它和 WebSocket 有什么区别？

**参考答案**：
SSE（Server-Sent Events）是基于 HTTP 的单向通信协议，服务器向客户端持续推送数据。

区别：
- **SSE**：单向（服务器→客户端），基于 HTTP，简单，自动重连，适合 LLM 流式输出
- **WebSocket**：双向，独立协议，更复杂，适合实时聊天室、游戏

LLM 流式输出用 SSE 而不用 WebSocket，因为：只需要服务器→客户端的单向推送，SSE 更轻量，兼容性更好。

---

### Q2：LLM 流式输出的数据格式是什么？如何解析？

**参考答案**：
格式为逐行 `data: {JSON}\n\n`，最后以 `data: [DONE]` 结束。

```python
for line in response.iter_lines():
    if line.startswith("data: "):
        data = line[6:]
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        delta = chunk["choices"][0]["delta"].get("content", "")
```

关键点：`delta.content` 是增量内容，不是完整内容，需要自己拼接。

---

### Q3：流式输出中 delta 和 message 有什么区别？

**参考答案**：
- `delta`：流式模式下的增量内容，每次只返回新增的字。如 `{"delta": {"content": "你"}}`
- `message`：非流式模式下的完整内容。如 `{"message": {"content": "你好世界"}}`

流式中第一个 chunk 的 delta 可能包含 `role` 字段（如 `{"delta": {"role": "assistant"}}`），后续 chunk 的 delta 只有 `content`。

---

### Q4：为什么 AI 应用必须用异步？同步有什么问题？

**参考答案**：
同步问题：一个 LLM 请求耗时 3-10 秒，同步模式下这期间整个程序卡住，其他用户必须排队等待。

异步优势：等待 LLM 响应期间，事件循环可以处理其他用户的请求，实现并发。一个进程可以同时处理几十上百个用户请求。

Python 中用 `asyncio` + `httpx.AsyncClient` 实现异步 HTTP 调用，用 `async def`/`await` 语法。

---

### Q5：asyncio.gather 和依次 await 有什么区别？

**参考答案**：
```python
# 依次 await：串行，总耗时 = 3+3+3 = 9秒
r1 = await task1()
r2 = await task2()
r3 = await task3()

# asyncio.gather：并行，总耗时 ≈ max(3,3,3) = 3秒
r1, r2, r3 = await asyncio.gather(task1(), task2(), task3())
```

`gather` 同时启动多个协程，并行等待它们完成。适合批量调用 LLM、并行处理多个独立任务。

---

### Q6：async def 函数和普通函数有什么区别？能直接调用吗？

**参考答案**：
`async def` 定义的是协程函数，调用它返回一个协程对象，**不会立即执行**。

```python
async def foo():
    return 1

# 不能直接调用
result = foo()       # 返回协程对象，不执行
result = await foo() # 正确，await 触发执行
```

协程必须通过 `await`、`asyncio.gather` 或 `asyncio.run()` 来执行。

---

### Q7：httpx.AsyncClient 为什么要在每次请求后保持打开？什么时候关闭？

**参考答案**：
`AsyncClient` 内部维护连接池，复用 TCP 连接（Keep-Alive）。如果每次请求都新建 client，每次都要建立新 TCP 连接，性能差。

正确用法：
```python
# 长期复用
client = httpx.AsyncClient()
# ... 多次请求 ...
await client.close()

# 或用上下文管理器自动关闭
async with httpx.AsyncClient() as client:
    response = await client.post(...)
```

在 LLM 应用中，通常创建一个全局 client 实例，应用生命周期内复用。

---

### Q8：流式输出中，如何处理网络中断？

**参考答案**：
网络中断会导致 `iter_lines()` 或 `aiter_lines()` 抛异常。处理方式：

```python
try:
    async for line in response.aiter_lines():
        # 处理数据
        pass
except (httpx.ConnectError, httpx.ReadTimeout) as e:
    # 1. 记录已接收的内容
    # 2. 决定是否重试（流式重试较复杂，因为部分内容已输出）
    # 3. 如果已输出部分内容，通常不重试，告知用户
    print(f"\n[网络中断: {e}]")
```

流式重试比非流式复杂，因为部分内容已经输出给用户了。生产中通常：
- 非流式：自动重试
- 流式：不重试，提示用户重新提问

---

### Q9：如果同时有 100 个用户调用 LLM，直接 asyncio.gather 会发生什么？

**参考答案**：
100 个请求同时发出，可能触发：
1. **API 限流（429）**：千问/OpenAI 都有 QPS 限制
2. **内存飙升**：100 个响应同时在内存中
3. **下游压力**：LLM 服务端也可能拒绝

解决方案：**用 Semaphore 控制并发数**
```python
sem = asyncio.Semaphore(5)  # 最多同时5个

async def limited_chat(messages):
    async with sem:
        return await chat_async(messages)

# 100个请求，但同时只有5个在执行
results = await asyncio.gather(*[limited_chat(m) for m in all_messages])
```

---

### Q10：如何在前端展示 LLM 的流式输出？

**参考答案**：
前端通过 `fetch` + `ReadableStream` 或 `EventSource` 接收 SSE：

```javascript
// 方式1：fetch + ReadableStream（推荐）
const response = await fetch("/api/chat", { method: "POST", body: ... });
const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    // 解析 SSE 数据，逐字追加到页面
}

// 方式2：EventSource（只支持 GET，不适合 POST 场景）
```

后端（FastAPI）用 `StreamingResponse` 返回 SSE 格式数据，前端逐字接收并渲染。

---

## 今日完成检查清单

- [ ] 运行 sse_stream.py，看到逐字打印效果
- [ ] 运行 stream_chat.py，体验流式多轮对话
- [ ] 运行 async_call.py，看到同步 vs 异步的耗时对比
- [ ] 运行 async_stream_chat.py（★★☆ 进阶）
- [ ] 写完 note.md（4 个问题）
- [ ] 代码 push 到 GitHub（`git add . && git commit -m "day04: SSE流式输出与异步调用" && git push`）
- [ ] 看一遍 10 道面试题，能答出至少 6 道

> 明天 Day 5：并发调用 + 多模型 Fallback + 封装 LLMClient（合并原 Day 10+11+12，节奏加快）
