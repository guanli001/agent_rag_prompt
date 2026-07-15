models = [
    {"name": "qwen-turbo", "pricing": {"input": 0.3, "output": 0.6}, "max_tokens": 8192},
    {"name": "qwen-plus", "pricing": {"input": 0.8, "output": 2.0}, "max_tokens": 32768},
    {"name": "qwen-max", "pricing": {"input": 2.0, "output": 6.0}, "max_tokens": 32768},
    {"name": "qwen-turbo-latest", "pricing": {"input": 0.25, "output": 0.5}, "max_tokens": 16384},
]

# 要求：提取每个模型的名字和总费用（input+output），按总费用从低到高排序
# 期望：[("qwen-turbo-latest", 0.75), ("qwen-turbo", 0.9), ("qwen-plus", 2.8), ("qwen-max", 8.0)]

# 创建一个字典
modellll = {model["name"]: model["pricing"]["input"] + model["pricing"]["output"]
            for model in models}

sorted_modellll = sorted(modellll.items(), key=lambda x: x[1])
