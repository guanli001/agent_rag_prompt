"""
Day 3 实验 1：采样参数对比
目标：直观感受 temperature 对输出的影响
"""
import os
import requests

api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    print("错误：请先设置环境变量 DASHSCOPE_API_KEY")
    exit(1)


def ask(question, temperature, top_p=1.0, max_tokens=None):
    """调用千问API"""
    body = {
        "model": "qwen3-max",
        "messages": [{"role": "user", "content": question}],
        "temperature": temperature,
        "top_p": top_p
    }
    if max_tokens:
        body["max_tokens"] = max_tokens

    response = requests.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=body
    )
    data = response.json()
    return data["choices"][0]["message"]["content"]


# ===== 实验 1：temperature 对比 =====
print("=" * 60)
print("实验 1：同一问题，不同 temperature，各问 3 次")
print("=" * 60)

question = "用一句话描述什么是人工智能"

for temp in [0, 0.5, 1.0]:
    print(f"\n--- temperature={temp} ---")
    for i in range(3):
        answer = ask(question, temp)
        print(f"  第{i + 1}次: {answer}")

# ===== 实验 2：top_p 对比 =====
print("\n" + "=" * 60)
print("实验 2：同一问题，不同 top_p")
print("=" * 60)

question2 = "推荐一个Python学习路线"

for tp in [0.1, 0.5, 1.0]:
    print(f"\n--- top_p={tp} ---")
    answer = ask(question2, temperature=0.7, top_p=tp)
    print(f"  {answer}")

# ===== 实验 3：max_tokens 限制 =====
print("\n" + "=" * 60)
print("实验 3：max_tokens 限制输出长度")
print("=" * 60)

for mt in [10, 50, 200]:
    print(f"\n--- max_tokens={mt} ---")
    answer = ask("详细解释什么是装饰器", temperature=0, max_tokens=mt)
    print(f"  {answer}")
    print(f"  [输出字数: {len(answer)}]")
