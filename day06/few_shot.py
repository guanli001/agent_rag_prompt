"""
Day 6 实验 1：Zero-shot vs Few-shot 对比
目标：直观感受给不给例子，模型回答质量的差异
"""
import os
import asyncio
import sys

# 把 day05 目录加入路径，才能 import LLMClient
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "day05"))
from day05.llm_client import LLMClient


async def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    client = LLMClient(api_key=api_key, max_concurrent=3)

    # 待分类的句子
    test_sentences = [
        "这个手机电池续航太差了，半天就没电",
        "快递速度很快，包装也很仔细，好评！",
        "味道还行吧，不算特别惊艳，但也不难吃",
        "用了三天就坏了，什么垃圾质量",
        "性价比很高，推荐购买",
    ]

    # ===== 方式 1：Zero-shot（不给例子）=====
    print("=" * 60)
    print("  方式 1：Zero-shot（不给例子，直接问）")
    print("=" * 60)

    zero_shot_prompt = "请判断以下句子的情感倾向，只回答：正面/负面/中性\n\n句子：{text}\n情感："
    for sentence in test_sentences:
        messages = [
            {"role": "user", "content": zero_shot_prompt.format(text=sentence)}
        ]
        result = await client.chat(messages, temperature=0)
        print(f"  句子: {sentence}")
        print(f"  结果: {result.content.strip()}")
        print()

    # ===== 方式 2：Few-shot（给3个例子）=====
    print("=" * 60)
    print("  方式 2：Few-shot（给3个例子，让模型照着答）")
    print("=" * 60)

    few_shot_prompt = """请判断句子的情感倾向。

示例：
句子：这家餐厅菜品很新鲜，服务员态度也很好
情感：正面

句子：等了一个小时才送到，饭都凉了
情感：负面

句子：味道一般般，价格也还行，没什么特别的
情感：中性

现在请判断：
句子：{text}
情感："""

    for sentence in test_sentences:
        messages = [
            {"role": "user", "content": few_shot_prompt.format(text=sentence)}
        ]
        result = await client.chat(messages, temperature=0)
        print(f"  句子: {sentence}")
        print(f"  结果: {result.content.strip()}")
        print()

    # ===== 对比 =====
    print("=" * 60)
    print("  对比结论")
    print("=" * 60)
    print("  Zero-shot：模型可能输出多余内容（如'这个句子是负面的，因为...'）")
    print("  Few-shot ：模型照着例子格式回答，输出干净（'负面'）")
    print("  结论    ：给例子 = 给格式 + 给判断标准，准确率和格式一致性都更好")
    await client.client.aclose()


asyncio.run(main())
