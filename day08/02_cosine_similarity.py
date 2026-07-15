"""
Day 8 任务 3：余弦相似度
用纯 Python 计算向量相似度，观察语义关系
"""
import os
import requests
import math
import numpy as np

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
    "我喜欢吃苹果",  # 0
    "我爱吃苹果",  # 1
    "今天天气不错",  # 2
    "机器学习很有趣",  # 3
    "深度学习是机器学习的分支",  # 4
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


# 安装 numpy（如果还没装）：pip install numpy
# 同样的计算，用 numpy 只需要一行：
def cosine_similarity_numpy(vec_a, vec_b):
    """numpy 版余弦相似度"""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
