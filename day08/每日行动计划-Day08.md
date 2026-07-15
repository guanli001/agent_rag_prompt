# Day 8 行动计划 — RAG 概念入门 + Embedding 基础 + 手写最简检索

> 日期：2026 年 7 月 16 日（周四）
> 预计耗时：5-6 小时
> 前置条件：已完成 Day 1-7（第 1 周全部完成：LLM 认知 → API → 流式 → 并发 → Prompt → LangChain 对比）
> 今日目标：理解 RAG 解决什么问题，搞懂 Embedding 和向量相似度，手写文档切分 + 检索的下半部分

---

## 第 2 周概览：RAG 知识库（7/16~7/22）

```
第2周（7/16~7/22）：RAG 基础+进阶
  Day 8 ← 今天：RAG概念 + Embedding + 向量相似度 + 手写文档切分和检索
  Day 9：手写完整最简 RAG（切分→Embedding→存储→检索→LLM生成）
  Day 10：文档加载与切分策略（按段落、按 token、保留 overlap）
  Day 11：Advanced RAG 概念（混合检索、Rerank、Query 改写）
  Day 12：用 LangChain 实现 RAG（对比手写版）
  Day 13：RAG 评估体系（RAGAS 概念入门）
  Day 14：第 2 周总结 + 综合面试题
```

**第 2 周核心原则**：和第 1 周一样，**先手写再用框架**。今天纯手写，不用 LangChain 的 Retriever。

---

## 〇、快速复习：第 1 周你学了什么（10 分钟）

脑子里过一遍，不用写代码：

| # | 问题 | 你的答案 |
|---|------|---------|
| 1 | LangChain 的 `ChatOpenAI` 帮你封装了哪些你手写过的东西？ | ？ |
| 2 | `with_structured_output()` 替代了你手写的哪个函数？ | ？ |
| 3 | `prompt | llm` 管道符的底层是 Python 的什么魔术方法？ | ？ |

能答出来就继续。答不出来翻 Day 7 笔记。

---

## 任务 1 ★☆☆ 必做：理解 RAG — 为什么需要它？（约 30 分钟）

### 1.1 LLM 的三个硬伤

你第 1 周学了怎么调 LLM，但它有三个致命问题：

**硬伤 1：知识截止日期**
- qwen-turbo 的训练数据有截止日期，之后的新闻、事件它不知道
- 你问"2026年7月15日发生了什么？"它答不上来

**硬伤 2：不懂你的私有数据**
- 你公司的内部文档、产品手册、客户记录，LLM 从来没见过
- 你问"我们公司的退货政策是什么？"它会瞎编

**硬伤 3：会编造（幻觉 / Hallucination）**
- LLM 是概率性系统（Day 1 学过），它不知道答案时会"编"一个看起来合理的回答
- 你问"鲁迅和周树人是什么关系？"它可能说"他们是同时代的不同作家"（其实是同一个人）

### 1.2 RAG 怎么解决？

**RAG = Retrieval-Augmented Generation（检索增强生成）**

拆开看：

```
R（Retrieve 检索）：先从你的知识库里搜出和问题相关的内容
A（Augment 增强）：把搜到的内容塞进 Prompt 里，作为上下文
G（Generate 生成）：LLM 基于这些上下文来回答问题
```

**类比**：开卷考试
- 不用 RAG = 闭卷考试，LLM 只能靠记忆回答，记不住就编
- 用 RAG = 开卷考试，LLM 先翻书找到相关页面，再照着书上的内容回答

### 1.3 RAG 全流程

```
准备阶段（提前做一次）：
  文档 → 切分成小块 → 每块算 Embedding（向量）→ 存入向量库

查询阶段（每次用户提问）：
  用户问题 → 算 Embedding → 在向量库里找最相似的小块 → 拼入 Prompt → LLM 生成回答
```

用一张图理解：

```
┌─────────────────────────────────────────────────────┐
│  准备阶段                                            │
│                                                      │
│  原始文档 → 切分 → chunk1 → Embedding → 向量1 ─┐    │
│                    → chunk2 → Embedding → 向量2 ─┤    │
│                    → chunk3 → Embedding → 向量3 ─┘    │
│                                          ↓           │
│                                    存入向量库         │
│                                                      │
├─────────────────────────────────────────────────────┤
│  查询阶段                                            │
│                                                      │
│  用户提问 → Embedding → 查询向量                     │
│                              ↓                       │
│                    在向量库里找最相似的 Top-K 个块    │
│                              ↓                       │
│              "根据以下信息回答：\n{检索到的内容}\n     │
│               问题：{用户问题}"                       │
│                              ↓                       │
│                         LLM 生成回答                  │
└─────────────────────────────────────────────────────┘
```

### 1.4 思考

- 你之前做的智能客服分类器（Day 6），如果加 RAG 会变成什么样？
- RAG 和你 Day 6 学的 Few-shot 有什么区别？（提示：Few-shot 是给"示例"，RAG 是给"知识"）
- 为什么不直接把整个文档塞进 Prompt？（提示：Day 2 学的上下文窗口限制）

---

## 任务 2 ★☆☆ 必做：Embedding — 把文本变成向量（约 1 小时）

### 2.1 什么是 Embedding？

**Embedding 就是把一段文本变成一串数字（向量）**，让计算机能"理解"语义。

举个例子：
```
"我喜欢吃苹果"  →  [0.12, -0.34, 0.56, ..., 0.78]  （1024 个数字）
"我爱吃苹果"    →  [0.11, -0.33, 0.55, ..., 0.77]  （和上面很接近）
"今天天气不错"  →  [0.89, 0.12, -0.67, ..., 0.03]  （和上面差很远）
```

**核心原理**：语义相近的文本，向量距离也近；语义无关的文本，向量距离远。

这就像把文本映射到一个"语义空间"里——意思相近的词/句子在这个空间里靠得近。

### 2.2 为什么 Embedding 能表示语义？

Embedding 模型是通过海量文本训练出来的。它在训练过程中学会了：
- "苹果"和"水果"经常一起出现 → 向量距离近
- "苹果"和"电脑"有时一起出现（苹果电脑）→ 向量距离中等
- "苹果"和"火箭"很少一起出现 → 向量距离远

**你不需要理解训练过程的数学细节**，只需要知道：**Embedding 把语义变成了数学可以计算的距离**。

### 2.3 调用 DashScope Embedding API

你的千问 API 兼容 OpenAI 格式，Embedding 也一样：

创建文件 `day08/01_embedding_basic.py`：

```python
"""
Day 8 任务 2：Embedding 基础
目标：调用千问 Embedding API，把文本变成向量，观察语义相似度
"""
import os
import requests

# ===== 配置 =====
api_key = os.environ.get("DASHSCOPE_API_KEY")
url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# ===== 1. 获取单条文本的 Embedding =====
print("=" * 60)
print("  实验 1：获取一条文本的 Embedding")
print("=" * 60)

data = {
    "model": "text-embedding-v3",
    "input": "人工智能正在改变世界",
    "dimensions": 1024
}

response = requests.post(url, headers=headers, json=data)
result = response.json()

embedding = result["data"][0]["embedding"]
print(f"  文本: 人工智能正在改变世界")
print(f"  向量长度: {len(embedding)}")  # 应该是 1024
print(f"  前 5 个数字: {embedding[:5]}")
print(f"  Token 使用: {result.get('usage', {})}")

# ===== 2. 批量获取多条文本的 Embedding =====
print("\n" + "=" * 60)
print("  实验 2：批量获取多条文本的 Embedding")
print("=" * 60)

texts = [
    "我喜欢吃苹果",
    "我爱吃苹果",
    "今天天气不错",
    "机器学习很有趣",
    "深度学习是机器学习的分支"
]

data = {
    "model": "text-embedding-v3",
    "input": texts,  # 直接传列表，一次请求多条
    "dimensions": 1024
}

response = requests.post(url, headers=headers, json=data)
result = response.json()

# 提取所有向量
all_embeddings = [item["embedding"] for item in result["data"]]
print(f"  获取了 {len(all_embeddings)} 条向量")
print(f"  每条向量长度: {len(all_embeddings[0])}")

# ===== 3. 观察相似文本 vs 不相似文本的向量差异 =====
print("\n" + "=" * 60)
print("  实验 3：观察向量差异")
print("=" * 60)

# 看看前几个维度的数值对比
for i, text in enumerate(texts):
    emb = all_embeddings[i]
    print(f"  '{text}'")
    print(f"    前3维: [{emb[0]:.4f}, {emb[1]:.4f}, {emb[2]:.4f}]")

print("\n  观察：'我喜欢吃苹果' 和 '我爱吃苹果' 的前3维数值应该很接近")
print("  '我喜欢吃苹果' 和 '今天天气不错' 的前3维数值应该差很远")
print("  （前3维看不出来很正常，相似度要用全部1024维来算——下一步学）")
```

### 2.4 思考

- Embedding 向量有 1024 个数字，每个数字代表什么含义？（提示：单个数字没有明确含义，但组合在一起表示语义）
- 为什么不能只看前几个数字判断相似度？（提示：语义信息分散在所有维度里）
- 你觉得 "苹果手机" 和 "iPhone" 的 Embedding 会接近吗？为什么？

---

## 任务 3 ★☆☆ 必做：余弦相似度 — 怎么衡量向量像不像（约 45 分钟）

### 3.1 问题：怎么判断两个向量"像不像"？

你有两个 1024 维的向量，怎么判断它们相似不相似？

最常用的方法：**余弦相似度（Cosine Similarity）**

### 3.2 余弦相似度的直觉理解

不看公式，先理解直觉：

- 每个向量有"方向"
- 两个向量方向越一致（夹角越小），相似度越高
- 夹角 0° → 相似度 = 1（完全相同方向）
- 夹角 90° → 相似度 = 0（完全无关）
- 夹角 180° → 相似度 = -1（完全相反）

```
向量A ↗          向量A ↗          向量A ↗
向量B ↗          向量B →          向量B ↙
夹角≈0°          夹角≈90°         夹角≈180°
相似度≈1         相似度≈0         相似度≈-1
```

### 3.3 余弦相似度的计算公式

```
cos(A, B) = (A · B) / (|A| × |B|)
```

拆开看：
- `A · B` = 点积 = 把两个向量对应位置相乘再求和：`a1*b1 + a2*b2 + ... + a1024*b1024`
- `|A|` = 向量 A 的长度（模）= `sqrt(a1² + a2² + ... + a1024²)`
- `|B|` = 向量 B 的长度（模）= 同理

**不用怕数学**，就是三步：算点积、算两个长度、做除法。

### 3.4 纯 Python 实现

先不用 numpy，用纯 Python 写一遍，理解每一步：

```python
import math

def cosine_similarity(vec_a, vec_b):
    """
    计算两个向量的余弦相似度
    返回值范围：-1 到 1，越接近 1 越相似
    """
    # 1. 计算点积：对应位置相乘再求和
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    # 2. 计算向量 A 的长度（模）
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))

    # 3. 计算向量 B 的长度（模）
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    # 4. 如果任一向量为零向量，返回 0（避免除以零）
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    # 5. 余弦相似度 = 点积 / (长度A × 长度B)
    return dot_product / (magnitude_a * magnitude_b)
```

### 3.5 实际测试

创建文件 `day08/02_cosine_similarity.py`：

```python
"""
Day 8 任务 3：余弦相似度
用纯 Python 计算向量相似度，观察语义关系
"""
import os
import requests
import math

# ===== 配置 =====
api_key = os.environ.get("DASHSCOPE_API_KEY")
url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def get_embeddings(texts):
    """批量获取文本的 Embedding"""
    data = {
        "model": "text-embedding-v3",
        "input": texts,
        "dimensions": 1024
    }
    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    return [item["embedding"] for item in result["data"]]

def cosine_similarity(vec_a, vec_b):
    """计算两个向量的余弦相似度（纯 Python 实现）"""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)

# ===== 测试：对比不同文本对的相似度 =====
print("=" * 60)
print("  余弦相似度实验")
print("=" * 60)

texts = [
    "我喜欢吃苹果",           # 0
    "我爱吃苹果",             # 1
    "今天天气不错",           # 2
    "机器学习很有趣",         # 3
    "深度学习是机器学习的分支", # 4
    "苹果公司的手机卖得很好",  # 5
]

embeddings = get_embeddings(texts)

# 计算所有两两组合的相似度
pairs = [
    (0, 1, "我喜欢吃苹果 vs 我爱吃苹果"),
    (0, 2, "我喜欢吃苹果 vs 今天天气不错"),
    (0, 5, "我喜欢吃苹果 vs 苹果公司的手机卖得很好"),
    (3, 4, "机器学习很有趣 vs 深度学习是机器学习的分支"),
    (3, 2, "机器学习很有趣 vs 今天天气不错"),
    (0, 0, "我喜欢吃苹果 vs 我喜欢吃苹果（自己和自己）"),
]

for i, j, label in pairs:
    sim = cosine_similarity(embeddings[i], embeddings[j])
    print(f"  {label}")
    print(f"    相似度: {sim:.4f}")
    print()

print("=" * 60)
print("  观察结论")
print("=" * 60)
print("""
  1. 自己和自己的相似度应该 = 1.0（完全相同）
  2. "我喜欢吃苹果" vs "我爱吃苹果" → 相似度应该很高（0.8+）
  3. "我喜欢吃苹果" vs "今天天气不错" → 相似度应该较低（0.3-0.5）
  4. "我喜欢吃苹果" vs "苹果公司的手机" → 相似度中等（有"苹果"这个共同词）
  5. "机器学习" vs "深度学习是机器学习的分支" → 相似度应该很高
""")
```

### 3.6 对比：numpy 版本

同样的计算，用 numpy 只需要一行：

```python
# 安装 numpy（如果还没装）：pip install numpy
import numpy as np

def cosine_similarity_numpy(vec_a, vec_b):
    """numpy 版余弦相似度"""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

**对比**：
- 纯 Python：你能看懂每一步在做什么
- numpy：代码短、速度快（底层 C 实现），但黑箱程度增加
- 和你第 1 周 "先手写再框架" 的思路完全一致

### 3.7 思考

- 为什么用余弦相似度而不是欧氏距离（直线距离）？（提示：文本长度不同时，欧氏距离会受长度影响，余弦相似度只看方向）
- 如果两个向量完全相同，余弦相似度是多少？
- 余弦相似度的值域是 [-1, 1]，但 Embedding 向量的相似度通常在什么范围？（提示：文本 Embedding 的相似度通常在 0~1 之间，很少出现负数）

---

## 任务 4 ★★☆ 进阶：手写代码练习 — 自己实现 cosine_similarity（约 30 分钟）

### 练习说明

和之前 extract_json、count_words 的练习方式一样：**你自己写，不要复制上面的代码**。

### 题目

写一个函数 `cosine_similarity(vec_a, vec_b)`，要求：

1. 接收两个数字列表（长度相同）
2. 计算它们的余弦相似度
3. 返回一个浮点数
4. 处理零向量情况（避免除以零）

### 提示

```
步骤 1：算点积 = a[0]*b[0] + a[1]*b[1] + ... + a[n]*b[n]
         用 zip(vec_a, vec_b) 可以同时遍历两个列表
         用 sum() 可以求和

步骤 2：算向量 A 的长度 = sqrt(a[0]² + a[1]² + ... + a[n]²)
         用 math.sqrt() 开平方

步骤 3：算向量 B 的长度 = 同理

步骤 4：如果长度 A 或长度 B 等于 0，返回 0.0

步骤 5：返回 点积 / (长度A * 长度B)
```

### 验证

写完后用这个测试代码验证：

```python
import math

# 你的函数写在这里
def cosine_similarity(vec_a, vec_b):
    # ... 你来写 ...
    pass

# 测试
print(cosine_similarity([1, 0, 0], [1, 0, 0]))  # 应该 ≈ 1.0
print(cosine_similarity([1, 0, 0], [0, 1, 0]))  # 应该 ≈ 0.0
print(cosine_similarity([1, 1, 0], [1, 1, 0]))  # 应该 ≈ 1.0
print(cosine_similarity([0, 0, 0], [1, 2, 3]))  # 应该 = 0.0（零向量）
print(cosine_similarity([1, 2, 3], [4, 5, 6]))  # 应该 ≈ 0.9746
```

### 提交方式

写完贴给我，我来 review。第一轮能过最好，过不了我给你提示再改。

---

## 任务 5 ★★☆ 进阶：手写文档切分（约 45 分钟）

### 5.1 为什么要切分？

RAG 的第一步是把长文档切成小块。原因：

1. **上下文窗口限制**：文档可能几万字，不可能全塞进 Prompt
2. **检索精度**：如果整篇文档作为一个向量，检索时语义太模糊；切成小块后，每块语义集中，检索更精准
3. **Token 成本**：只检索相关的几块，不用把整篇文档都发给 LLM

### 5.2 最简单的切分方法：固定长度 + Overlap

```
原始文档（假设 1500 字）：
|--------------------------------------------------|
0                                                1500

切分（chunk_size=500, overlap=100）：
|--------chunk1--------|
0                    500
               |--------chunk2--------|
               400                  900
                          |--------chunk3--------|
                          800                  1300
                                     |--------chunk4--------|
                                     1200                 1600(截断)
```

**chunk_size**：每块多长
**overlap**：相邻块重叠多少字

**为什么要 overlap？** 如果一句关键信息正好被切断了（比如"退货政策适用于|购买后30天内"），overlap 能让这句话在两个块里都出现，不至于丢失。

### 5.3 手写 split_text

创建文件 `day08/03_split_text.py`：

```python
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
        # 取一块
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # 下一块的起点：当前起点 + chunk_size - overlap
        # 这样相邻两块会有 overlap 个字符的重叠
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
```

### 5.4 思考

- 固定长度切分有什么问题？（提示：可能把一句话从中间切断）
- 你能想到更好的切分方式吗？（提示：按句子、按段落切？这就是 Day 10 要学的"语义切分"）
- overlap 设为 0 会怎样？设为 chunk_size 会怎样？

---

## 任务 6 ★★☆ 进阶：迷你检索 — 把切分+Embedding+相似度组合起来（约 1 小时）

### 6.1 目标

把今天学的三个东西组合起来，做一个**迷你检索系统**：
1. 把文档切分成小块（任务 5）
2. 把每块算 Embedding（任务 2）
3. 用户提问时，算问题的 Embedding，找最相似的小块（任务 3）

**注意**：今天只做检索，不做 LLM 生成。明天 Day 9 再把 LLM 接上，完成完整 RAG。

### 6.2 代码

创建文件 `day08/04_mini_retrieval.py`：

```python
"""
Day 8 任务 6：迷你检索系统
组合：切分 → Embedding → 余弦相似度 → Top-K 检索
"""
import os
import math
import requests

# ===== 配置 =====
api_key = os.environ.get("DASHSCOPE_API_KEY")
embed_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# ===== 工具函数 =====

def get_embeddings(texts):
    """批量获取文本的 Embedding"""
    data = {
        "model": "text-embedding-v3",
        "input": texts,
        "dimensions": 1024
    }
    response = requests.post(embed_url, headers=headers, json=data)
    result = response.json()
    return [item["embedding"] for item in result["data"]]

def cosine_similarity(vec_a, vec_b):
    """余弦相似度（纯 Python）"""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)

def split_text(text, chunk_size=200, overlap=30):
    """固定长度切分 + overlap"""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        start = start + chunk_size - overlap
    return chunks

# ===== 迷你检索系统 =====

class MiniRetriever:
    """最简单的检索器：把文档存进去，用向量相似度检索"""

    def __init__(self):
        self.chunks = []       # 存文本块
        self.embeddings = []   # 存对应的向量

    def add_document(self, text, chunk_size=200, overlap=30):
        """添加文档：切分 → Embedding → 存储"""
        # 1. 切分
        new_chunks = split_text(text, chunk_size, overlap)
        print(f"  切分成 {len(new_chunks)} 块")

        # 2. 批量获取 Embedding
        new_embeddings = get_embeddings(new_chunks)
        print(f"  获取了 {len(new_embeddings)} 个向量")

        # 3. 存储
        self.chunks.extend(new_chunks)
        self.embeddings.extend(new_embeddings)
        print(f"  当前知识库共 {len(self.chunks)} 块")

    def search(self, query, top_k=3):
        """检索：返回和 query 最相似的 top_k 个文本块"""
        # 1. 获取查询的 Embedding
        query_emb = get_embeddings([query])[0]

        # 2. 计算和所有块的相似度
        similarities = []
        for i, doc_emb in enumerate(self.embeddings):
            sim = cosine_similarity(query_emb, doc_emb)
            similarities.append((i, sim))

        # 3. 按相似度排序，取前 top_k 个
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]

        # 4. 返回结果
        results = []
        for idx, sim in top_results:
            results.append({
                "chunk": self.chunks[idx],
                "similarity": sim,
                "index": idx
            })
        return results

# ===== 测试 =====
print("=" * 60)
print("  迷你检索系统")
print("=" * 60)

# 知识库文档
documents = [
    """机器学习是人工智能的核心技术之一，它使计算机系统能够从数据中自动学习和改进，而无需明确编程。机器学习算法可以分为监督学习、无监督学习和强化学习三大类。""",

    """深度学习是机器学习的一个子领域，它使用多层神经网络来模拟人脑的学习过程。深度学习在图像识别、自然语言处理、语音识别等领域取得了突破性进展。""",

    """RAG（检索增强生成）是一种结合检索和生成的技术，它通过从外部知识库检索相关信息来增强LLM的回答能力，有效减少幻觉问题。RAG的流程包括文档切分、向量化、检索和生成。""",

    """Agent是AI应用的最高形态，它能够自主决策、调用工具、执行多步骤任务。Agent的核心设计模式包括ReAct、Plan-and-Execute等。Agent需要循环执行思考和行动，直到完成任务。"""
]

# 构建检索器
retriever = MiniRetriever()
for i, doc in enumerate(documents):
    print(f"\n添加文档 {i+1}...")
    retriever.add_document(doc, chunk_size=200, overlap=30)

# 检索测试
print("\n" + "=" * 60)
print("  检索测试")
print("=" * 60)

queries = [
    "什么是深度学习？",
    "RAG 是什么？",
    "Agent 有哪些设计模式？",
    "机器学习分几类？"
]

for query in queries:
    print(f"\n  查询: {query}")
    results = retriever.search(query, top_k=2)
    for i, r in enumerate(results):
        print(f"    Top{i+1} [相似度: {r['similarity']:.4f}]")
        print(f"    内容: {r['chunk'][:60]}...")

# ===== 对比理解 =====
print("\n" + "=" * 60)
print("  你刚做了什么？")
print("=" * 60)
print("""
  1. 把 4 段文档切分成小块
  2. 每块算了 Embedding（1024 维向量）
  3. 用户提问时，也算 Embedding
  4. 用余弦相似度找到最相关的块

  这就是 RAG 的 "R"（Retrieve 检索）部分！

  明天 Day 9，你会把这个检索结果拼进 Prompt，
  交给 LLM 生成回答 —— 那就是完整的 RAG。
""")
```

### 6.3 思考

- 检索结果中，相似度最高的是不是就是你期望的答案所在的块？
- 如果相似度都很低（比如都不到 0.5），说明什么？（提示：知识库里可能没有相关内容）
- `top_k=2` 意味着取 2 个最相似的块。设太大或太小分别有什么问题？

---

## 10 道面试题

### Q1：什么是 RAG？它解决了 LLM 的什么问题？

**参考答案**：
RAG（Retrieval-Augmented Generation，检索增强生成）是一种通过从外部知识库检索相关信息来增强 LLM 回答能力的技术。

它解决 LLM 的三个问题：
1. **知识截止**：LLM 训练数据有截止日期，不知道之后的信息
2. **不懂私有数据**：LLM 没见过企业内部文档
3. **幻觉**：LLM 不知道答案时会编造

RAG 的流程：文档切分 → Embedding → 存入向量库 → 查询时检索相关内容 → 拼入 Prompt → LLM 基于检索内容生成回答。

---

### Q2：Embedding 是什么？它在 RAG 中的作用？

**参考答案**：
Embedding 是把文本映射成一组数值向量（如 1024 维浮点数数组）的技术。

核心特性：语义相近的文本，向量距离也近；语义无关的文本，向量距离远。

在 RAG 中的作用：
1. **建库阶段**：把每个文档块转成向量存入向量库
2. **检索阶段**：把用户问题也转成向量，和库里的向量比相似度，找到最相关的文档块

Embedding 是 RAG 检索的基础——没有 Embedding 就无法做语义搜索。

---

### Q3：为什么 RAG 要先切分文档？不切分行不行？

**参考答案**：
必须切分，原因：

1. **上下文窗口限制**：长文档可能超出模型的 Token 限制
2. **检索精度**：整篇文档作为一个向量，语义太杂，检索不精准；切分成小块后每块语义集中
3. **Token 成本**：只检索相关的几块发给 LLM，不用发整篇文档
4. **回答质量**：给 LLM 精准的小块上下文，比给一大堆无关内容效果更好

不切分的后果：要么放不进上下文窗口，要么检索精度差，要么 Token 成本爆炸。

---

### Q4：余弦相似度的原理是什么？为什么用它而不是欧氏距离？

**参考答案**：
余弦相似度通过计算两个向量夹角的余弦值来衡量相似度，公式：`cos(A,B) = A·B / (|A|×|B|)`，值域 [-1, 1]。

用余弦而不是欧氏距离的原因：
- **不受向量长度影响**：文本长短不同时，欧氏距离会受长度影响，而余弦只看方向
- **更适合语义比较**：Embedding 向量的"方向"代表语义，长度可能受文本长度影响
- **计算高效**：只需点积和模，不需要逐维算差值

---

### Q5：文档切分时为什么要加 overlap？

**参考答案**：
overlap（重叠）是指相邻两个文本块之间重叠的部分。

原因：如果不加 overlap，一句关键信息可能正好被切在两个块的边界上（如"退货政策适用于|购买后30天内"），导致两个块都不完整，检索时都匹配不上。

加了 overlap 后，边界附近的文本会在两个块中都出现，保证信息的完整性。

经验值：overlap 通常设为 chunk_size 的 10%-20%。

---

### Q6：RAG 和 Few-shot 有什么区别？

**参考答案**：
| 维度 | Few-shot | RAG |
|------|---------|-----|
| 给 LLM 什么 | 示例（输入→输出对） | 知识（文档内容） |
| 解决什么问题 | 教 LLM 按什么格式/模式回答 | 给 LLM 补充它不知道的知识 |
| 内容是否动态 | 固定写在 Prompt 里 | 根据问题动态检索 |
| 数据量 | 少量（1-5 个示例） | 大量（成百上千个文档块） |

简单说：Few-shot 教 LLM "怎么做"，RAG 告诉 LLM "用什么知识做"。

---

### Q7：top_k 参数设大设小各有什么问题？

**参考答案**：
top_k 控制检索返回多少个文档块。

- **太小（如 1）**：可能漏掉相关信息，如果检索不准就完全没有正确上下文
- **太大（如 20）**：
  - Token 成本增加（要把更多内容塞进 Prompt）
  - 噪声增加（不相关的内容干扰 LLM 判断）
  - 可能超出上下文窗口

经验值：通常设 3-5，配合 Rerank（Day 11 会学）先粗召回再精排。

---

### Q8：Embedding 向量的维度有什么影响？维度越高越好吗？

**参考答案**：
维度表示向量能表达的语义信息的丰富程度。

- **维度高**（如 1536、2048）：语义表达能力更强，精度更高，但存储和计算成本更大
- **维度低**（如 256、128）：资源消耗小，但可能丢失部分语义信息

不是越高越好。需要根据场景权衡：
- 原型开发/小规模数据：1024 维够用
- 追求精度/大规模数据：1536 或更高
- 资源受限/移动端：768 或更低

千问 text-embedding-v3 默认 1024 维，是性能与成本的最佳平衡点。

---

### Q9：你手写的 MiniRetriever 和真正的向量数据库有什么区别？

**参考答案**：
手写的 MiniRetriever：
- 向量存在 Python 列表里
- 检索时遍历所有向量算相似度（暴力搜索）
- 数据量大了会很慢（O(n) 复杂度）
- 重启后数据丢失

真正的向量数据库（如 Chroma、Qdrant、Milvus）：
- 使用 ANN（近似最近邻）算法，检索速度快（O(log n)）
- 支持持久化存储
- 支持过滤、分页、增删改
- 支持百万级甚至亿级向量

但核心原理一样：都是算向量相似度。手写版帮你理解原理，向量库帮你解决工程问题。

---

### Q10：如果用户问的问题在知识库里没有相关内容，RAG 会怎样？怎么处理？

**参考答案**：
如果知识库里没有相关内容：
1. 检索出来的块相似度都很低（比如 < 0.5）
2. LLM 还是会基于这些不相关的内容生成回答 → 可能产生幻觉

处理方法：
1. **设相似度阈值**：如果最高相似度低于某个阈值（如 0.6），直接回复"抱歉，我没有找到相关信息"
2. **Prompt 约束**：在 Prompt 里加"如果以下信息和问题无关，请说不知道"
3. **Fallback 策略**：知识库没有就转搜索引擎，或者转人工客服

这是 RAG 工程化的重要环节——不是检索到了就一定要用。

---

## 今日完成检查清单

- [ ] 理解 RAG 的概念（Retrieve + Augment + Generate）
- [ ] 运行 `01_embedding_basic.py`，看到文本变成 1024 维向量
- [ ] 运行 `02_cosine_similarity.py`，看到相似/不相似文本的相似度差异
- [ ] 手写 `cosine_similarity` 函数并提交 review
- [ ] 运行 `03_split_text.py`，理解切分和 overlap
- [ ] 运行 `04_mini_retrieval.py`，完成迷你检索系统
- [ ] 看一遍 10 道面试题，能答出至少 7 道
- [ ] 代码 push 到 GitHub：
  ```bash
  git add day08/
  git commit -m "day08: RAG概念-Embedding基础-余弦相似度-迷你检索"
  git push
  ```

---

> 📌 今天你完成了 RAG 的 "R"（Retrieve）部分。
> 你手写了：文档切分、Embedding 调用、余弦相似度、迷你检索器。
> 明天 Day 9：把这些拼上 LLM，完成完整 RAG —— 检索到的内容拼进 Prompt，让 LLM 基于知识库回答问题。
>
> **今天的核心收获**：Embedding 把"语义"变成了"数学"，余弦相似度把"像不像"变成了"距离远近"。这就是 RAG 检索的底层原理。
