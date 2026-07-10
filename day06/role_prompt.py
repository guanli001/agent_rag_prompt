"""
Day 6 实验 3：角色设定 + 结构化 Prompt
目标：学会用 system 设定角色，用结构化格式组织 Prompt
"""
import os
import asyncio
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "day05"))
from day05.llm_client import LLMClient


async def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    client = LLMClient(api_key=api_key, max_concurrent=3)

    question = "什么是递归？"

    # ===== 角色 1：小学老师 =====
    print("=" * 60)
    print("  角色 1：小学老师")
    print("=" * 60)
    result = await client.chat([
        {
            "role": "system",
            "content": """你是一位小学编程兴趣班的老师。
你的特点：
- 用小学生能听懂的语言
- 多用生活中的比喻
- 不超过 100 字
- 不用专业术语"""
        },
        {"role": "user", "content": question}
    ], temperature=0.7)
    print(f"  {result.content}\n")

    # ===== 角色 2：大学教授 =====
    print("=" * 60)
    print("  角色 2：大学教授")
    print("=" * 60)
    result = await client.chat([
        {
            "role": "system",
            "content": """你是一位计算机科学教授。
你的特点：
- 使用专业术语
- 讲解底层原理
- 会提到时间复杂度和空间复杂度
- 不超过 200 字"""
        },
        {"role": "user", "content": question}
    ], temperature=0.7)
    print(f"  {result.content}\n")

    # ===== 角色 3：代码审核员 =====
    print("=" * 60)
    print("  角色 3：代码审核员")
    print("=" * 60)
    result = await client.chat([
        {
            "role": "system",
            "content": """你是一位严格的代码审核员。
你的特点：
- 只关注代码实现
- 给出实际代码示例
- 指出常见错误
- 不解释概念，直接上代码"""
        },
        {"role": "user", "content": question}
    ], temperature=0.7)
    print(f"  {result.content}\n")

    # ===== 结构化 Prompt 示例 =====
    print("=" * 60)
    print("  结构化 Prompt（用标记分区）")
    print("=" * 60)

    structured_prompt = """请按照以下格式回答问题：

## 概念
用一句话解释

## 原理
用 2-3 句话说明工作原理

## 代码示例
给出一个最简单的 Python 代码

## 常见错误
列出 1-2 个初学者容易犯的错误

---
问题：什么是递归？"""

    result = await client.chat([
        {"role": "user", "content": structured_prompt}
    ], temperature=0.7)
    print(f"  {result.content}\n")

    # ===== 对比 =====
    print("=" * 60)
    print("  对比结论")
    print("=" * 60)
    print("  角色设定：控制回答风格、深度、用词")
    print("  结构化Prompt：用标题/分隔符分区，输出格式可控、可解析")
    print("  两者结合 = 生产级 Prompt 的基础")

    await client.close()

asyncio.run(main())