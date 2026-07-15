"""
写一个函数 `cosine_similarity(vec_a, vec_b)`，要求：

1. 接收两个数字列表（长度相同）
2. 计算它们的余弦相似度
3. 返回一个浮点数
4. 处理零向量情况（避免除以零）
"""
import math


def cosine_similarity(vec_a, vec_b):
    """
    计算两个向量的余弦相似度
    :param a: 向量A（列表或元组）
    :param b: 向量B（列表或元组）
    :return: 余弦相似度，范围[-1, 1]
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("向量长度不一致")
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a ** 2 for a in vec_a))
    norm_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# 测试
print(cosine_similarity([1, 0, 0], [1, 0, 0]))  # 应该 ≈ 1.0
print(cosine_similarity([1, 0, 0], [0, 1, 0]))  # 应该 ≈ 0.0
print(cosine_similarity([1, 1, 0], [1, 1, 0]))  # 应该 ≈ 1.0
print(cosine_similarity([0, 0, 0], [1, 2, 3]))  # 应该 = 0.0（零向量）
print(cosine_similarity([1, 2, 3], [4, 5, 6]))  # 应该 ≈ 0.9746
