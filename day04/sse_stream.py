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
    }, stream=True, # requests 也要开启 stream
)
