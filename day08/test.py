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
