"""
Day 6 实验 4：结构化输出（JSON）
目标：让模型输出 JSON，程序能直接解析
"""
import os
import asyncio
import json
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "day05"))
from day05.llm_client import LLMClient


async def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    client = LLMClient(api_key=api_key, max_concurrent=3)

    # ===== 实验 1：让模型输出 JSON =====
    print("=" * 60)
    print("  实验 1：让模型输出 JSON")
    print("=" * 60)

    reviews = [
        "这款手机拍照很清晰，电池续航也不错，但是价格有点贵。",
        "衣服质量很差，洗了一次就缩水了，客服态度也差。",
        "味道还行，环境一般，服务态度挺好的。",
    ]

    json_prompt = """请分析以下用户评价，并按 JSON 格式输出。

评价：{review}

请输出以下 JSON 格式（只输出 JSON，不要其他内容）：
```json
{{
  "情感": "正面/负面/中性",
  "优点": ["优点1", "优点2"],
  "缺点": ["缺点1"],
  "评分": 1-5,
  "关键词": ["关键词1", "关键词2"]
}}
```"""

    for review in reviews:
        messages = [{"role": "user", "content": json_prompt.format(review=review)}]
        result = await client.chat(messages, temperature=0)
        print(f"  评价: {review}")
        print(f"  原始输出: {result.content.strip()}")

        # 尝试解析 JSON
        try:
            # 去掉可能的 markdown 代码块标记
            content = result.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)
            print(f"  解析成功: {data}")
            print(f"  情感: {data['情感']}, 评分: {data['评分']}")
        except json.JSONDecodeError as e:
            print(f"  JSON 解析失败: {e}")
        print()

    # ===== 实验 2：用 Few-shot 提高 JSON 输出稳定性 =====
    print("=" * 60)
    print("  实验 2：Few-shot + JSON（给一个例子让格式更稳定）")
    print("=" * 60)

    few_shot_json_prompt = """请分析用户评价并输出 JSON。示例：
评价：这耳机音质不错，佩戴舒服，但线太短了。
```json
{{
  "情感": "正面",
  "优点": ["音质好", "佩戴舒服"],
  "缺点": ["线太短"],
  "评分": 4,
  "关键词": ["音质", "佩戴", "线短"]
}}现在请分析：
评价：{review}"""

    for review in reviews:
        messages = [{"role": "user", "content": few_shot_json_prompt.format(review=review)}]
        result = await client.chat(messages, temperature=0)
        print(f"  评价: {review}")
        print(f"  输出: {result.content.strip()[:100]}...")
        print()

    # ===== 对比 =====
    print("=" * 60)
    print("  对比结论")
    print("=" * 60)
    print("  不给例子：模型可能加多余文字、JSON 格式不标准")
    print("  给例子  ：格式更稳定，解析成功率高")
    print("  生产建议：重要场景用 Few-shot + JSON，并加 try-except 兜底")

    await client.close()


if __name__ == '__main__':
    asyncio.run(main())
