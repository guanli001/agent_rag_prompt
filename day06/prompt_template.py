"""
Day 6 实验 5：Prompt 模板管理
目标：把 Prompt 变量化、统一管理，为后续项目做准备
"""
import os
import asyncio
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "day05"))
from day05.llm_client import LLMClient


class PromptTemplate:
    """Prompt 模板：变量化 + 复用"""

    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        """填充变量"""
        return self.template.format(**kwargs)

    def to_messages(self, role="user", **kwargs) -> list:
        """直接转成 messages 格式"""
        return [{"role": role, "content": self.format(**kwargs)}]


class PromptManager:
    """Prompt 管理器：集中存储所有 Prompt 模板"""

    def __init__(self):
        self.templates = {}

    def add(self, name: str, template: str):
        """注册一个模板"""
        self.templates[name] = PromptTemplate(template)

    def get(self, name: str) -> PromptTemplate:
        """获取模板"""
        if name not in self.templates:
            raise KeyError(f"模板不存在: {name}，可用: {list(self.templates.keys())}")
        return self.templates[name]


async def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    client = LLMClient(api_key=api_key, max_concurrent=3)

    # ===== 注册模板 =====
    pm = PromptManager()

    pm.add("translate", "请将以下内容翻译为{lang}，只输出翻译结果：\n\n{text}")
    pm.add("summarize", "请用不超过{max_words}个字总结以下内容：\n\n{text}")
    pm.add("sentiment", """请判断以下文本的情感倾向。

示例：
文本：今天天气真好 → 正面
文本：心情很差 → 负面

文本：{text}
情感：""")
    pm.add("explain_code", """你是一位 Python 编程老师。
请解释以下代码的作用，用简洁明了的语言：

```python
{code}
```""")

    # ===== 使用模板 1：翻译 =====
    print("=" * 60)
    print("  模板 1：翻译")
    print("=" * 60)
    messages = pm.get("translate").to_messages(
        lang="英文",
        text="人工智能正在改变我们的生活方式"
    )
    result = await client.chat(messages, temperature=0)
    print(f"  {result.content}\n")

    # ===== 使用模板 2：总结 =====
    print("=" * 60)
    print("  模板 2：总结")
    print("=" * 60)
    long_text = """Python 是一种广泛使用的高级编程语言，由 Guido van Rossum 于 1991 年创建。
Python 的设计哲学强调代码的可读性和简洁性，使用缩进而非花括号来划分代码块。
Python 支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。
Python 拥有庞大的标准库和第三方包生态系统，广泛应用于 Web 开发、数据分析、人工智能、自动化运维等领域。"""
    messages = pm.get("summarize").to_messages(max_words=50, text=long_text)
    result = await client.chat(messages, temperature=0)
    print(f"  {result.content}\n")

    # ===== 使用模板 3：情感分析（Few-shot）=====
    print("=" * 60)
    print("  模板 3：情感分析（Few-shot）")
    print("=" * 60)
    test_texts = ["这个产品太好用了", "等了三天才发货，差评", "一般般吧"]
    for text in test_texts:
        messages = pm.get("sentiment").to_messages(text=text)
        result = await client.chat(messages, temperature=0)
        print(f"  {text} → {result.content.strip()}")

    # ===== 使用模板 4：代码解释（角色 + 结构化）=====
    print(f"\n{'=' * 60}")
    print("  模板 4：代码解释（角色 + 结构化）")
    print("=" * 60)
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
    """

    messages = pm.get("explain_code").to_messages(code=code)
    result = await client.chat(messages, temperature=0.7)
    print(f"  {result.content}\n")

    # ===== 总结 =====
    print("=" * 60)
    print("  总结")
    print("=" * 60)
    print("  PromptTemplate：变量化，同一个模板填不同参数复用")
    print("  PromptManager：集中管理，不用在代码里到处找 Prompt")
    print("  后面 RAG/Agent 阶段会大量用到模板管理")

    await client.close()


asyncio.run(main())