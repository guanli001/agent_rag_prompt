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
