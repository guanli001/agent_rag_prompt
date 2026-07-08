"""
Day 4 实验 1：SSE 流式输出
目标：实现逐字打印效果，理解 SSE 数据格式
不使用任何框架，纯 requests
"""
import os
import requests
import json
import time

import requests as requests

api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    print("错误：请先设置环境变量 DASHSCOPE_API_KEY")
    exit(1)

print("=" * 50)
print("  实验 1：流式输出（逐字打印）")
print("=" * 50)

# ===== 流式调用 =====
print("\n--- 流式输出 ---")
start_time = time.time()  # 计时开始
response = requests.post(
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        "model": "qwen3-max",
        "messages": [{"role": "user", "content": "写一首关于编程的短诗，4句话"}],
        "stream": True,  # 开启流式
        "temperature": 0.7
    }, stream=True,  # requests 也要开启 stream
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
        data = line_str[6:]  # 去掉 "data: " 前缀

        if data == "[DONE]":
            break

        chunk_data = json.loads(data)
        delta = chunk_data["choices"][0]["delta"].get("content", "")

        if delta:
            if first_token_time is None:
                first_token_time = time.time()
                print(f"  [首次Token时间: {first_token_time - start_time:.2f} 秒]")
                print(delta, end="", flush=True)
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