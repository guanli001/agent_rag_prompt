"""
Day 6 实验 2：Chain of Thought（思维链）
目标：对比"直接答"和"一步一步想"的准确率差异
"""
import os
import asyncio
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "day05"))
from day05.llm_client import LLMClient


async def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    client = LLMClient(api_key=api_key, max_concurrent=3)

    # 数学题（有明确答案，方便判断对错）
    questions = [
        "小明有 50 元，买了 3 本书每本 12 元，又买了一支笔 5 元，还剩多少钱？",
        "一个班 45 人，男生比女生多 7 人，女生有多少人？",
        "一桶油重 15 千克，用掉三分之一后又用掉 2 千克，还剩多少千克？",
        "火车每小时行 120 千米，3.5 小时能到目的地，已经行了 2 小时，还剩多远？",
        "一件衣服打八折后是 240 元，原价是多少？",
    ]

    # ===== 方式 1：直接答 =====
    print("=" * 60)
    print("  方式 1：直接答（不加 CoT）")
    print("=" * 60)

    direct_prompt = "请直接回答以下问题，只给最终答案：\n\n{question}"

    for q in questions:
        messages = [{"role": "user", "content": direct_prompt.format(question=q)}]
        result = await client.chat(messages, temperature=0)
        print(f"  问题: {q}")
        print(f"  回答: {result.content.strip()}")
        print()

    # ===== 方式 2：CoT 思维链 =====
    print("=" * 60)
    print("  方式 2：CoT（一步一步想）")
    print("=" * 60)

    cot_prompt = "请一步一步分析以下问题，写出计算过程，最后给出答案：\n\n{question}"

    for q in questions:
        messages = [{"role": "user", "content": cot_prompt.format(question=q)}]
        result = await client.chat(messages, temperature=0)
        print(f"  问题: {q}")
        print(f"  回答: {result.content.strip()}")
        print(f"  Token: {result.total_tokens}")
        print()

    # ===== 对比 =====
    print("=" * 60)
    print("  对比结论")
    print("=" * 60)
    print("  直接答：快、省 Token，但复杂题容易算错")
    print("  CoT   ：慢、费 Token，但有推理过程，准确率更高")
    print("  使用场景：简单问题用直接答，数学/逻辑/推理问题用 CoT")

    await client.close()


asyncio.run(main())
