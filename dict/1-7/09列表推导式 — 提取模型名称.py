"""
场景：你的 Agent 需要管理多个 AI 模型配置。现在有一批模型信息，要快速筛选出符合条件的模型名称。

📝 题目
写一个函数 filter_models(models: list, min_tokens: int) -> list：

models 是一个列表，每个元素是字典：{"name": "模型名", "max_tokens": 数字}
筛选出 max_tokens >= min_tokens 的模型
返回这些模型的名称列表（只要 name 字段）
用列表推导式一行写完核心逻辑
"""


def filter_models(models: list, min_tokens: int) -> list:
    return [model["name"] for model in models if model["max_tokens"] >= min_tokens]


"""
练习 9b：字典推导式 — 模型名 → 容量映射
列表推导式还有一种变体叫字典推导式。场景：把模型列表转成一个字典，方便按名字快速查容量。

📝 题目
写一个函数 build_model_map(models: list) -> dict：

输入和上一题一样：[{"name": "xxx", "max_tokens": N}, ...]
输出是一个字典：{"模型名": max_tokens值, ...}
用字典推导式一行写完
如果有重复的 name，后面的覆盖前面的（默认行为就好）
"""


def build_model_map(models: list) -> dict:
    return {model["name"]: model["max_tokens"] for model in models}
