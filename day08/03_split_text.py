"""
Day 8 任务 5：文档切分
手写最简单的固定长度切分 + overlap
"""


def split_text(text, chunk_size=500, overlap=50):
    """
    将文本按固定长度切分，带 overlap

    参数：
        text: 原始文本
        chunk_size: 每块长度（字符数）
        overlap: 相邻块重叠的字符数

    返回：字符串列表，每个元素是一个文本块
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start = start + chunk_size - overlap

    return chunks


# ===== 测试 =====
print("=" * 60)
print("  文档切分实验")
print("=" * 60)

# 模拟一篇文档
sample_doc = """
人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。

机器学习是人工智能的核心技术之一，它使计算机系统能够从数据中自动学习和改进，而无需明确编程。机器学习算法可以分为监督学习、无监督学习和强化学习三大类。

深度学习是机器学习的一个子领域，它使用多层神经网络来模拟人脑的学习过程。深度学习在图像识别、自然语言处理、语音识别等领域取得了突破性进展。

自然语言处理（NLP）是人工智能的另一个重要分支，专注于让计算机理解和生成人类语言。大语言模型（LLM）是NLP领域的最新成果，如GPT系列、BERT、通义千问等。

RAG（检索增强生成）是一种结合检索和生成的技术，它通过从外部知识库检索相关信息来增强LLM的回答能力，有效减少幻觉问题。

Agent是AI应用的最高形态，它能够自主决策、调用工具、执行多步骤任务。Agent的核心设计模式包括ReAct、Plan-and-Execute等。
""".strip()

print(f"  原始文档长度: {len(sample_doc)} 字符")

# 用不同的参数切分
for chunk_size, overlap in [(200, 50), (300, 100), (100, 0)]:
    chunks = split_text(sample_doc, chunk_size=chunk_size, overlap=overlap)
    print(f"\n  参数: chunk_size={chunk_size}, overlap={overlap}")
    print(f"  切分成 {len(chunks)} 块")
    for i, chunk in enumerate(chunks):
        print(f"    块{i}: [{len(chunk)}字] {chunk[:30]}...")

# ===== 观察对比 =====
print("\n" + "=" * 60)
print("  观察")
print("=" * 60)
print("""
  1. chunk_size 越大，块越少，但每块语义可能越杂
  2. chunk_size 越小，块越多，但可能把一句话切断
  3. overlap=0 时，块之间没有重叠，可能丢失边界信息
  4. overlap 太大（接近 chunk_size），切分没意义，重复太多

  经验值：chunk_size=300-500字符，overlap=chunk_size的10%-20%
""")
