logs = [
    {"model": "qwen-turbo", "tokens": 150, "time_ms": 320},
    {"model": "qwen-max", "tokens": 420, "time_ms": 890},
    {"model": "qwen-turbo", "tokens": 280, "time_ms": 510},
    {"model": "qwen-plus", "tokens": 300, "time_ms": 600},
    {"model": "qwen-turbo", "tokens": 90, "time_ms": 180},
    {"model": "qwen-max", "tokens": 350, "time_ms": 750},
]

# 要求：按 model 分组，统计每个模型的「总 tokens」和「总耗时」
# 期望：{
#     "qwen-turbo": {"total_tokens": 520, "total_time_ms": 1010},
#     "qwen-max": {"total_tokens": 770, "total_time_ms": 1640},
#     "qwen-plus": {"total_tokens": 300, "total_time_ms": 600}
# }

result = {}
for log in logs:
    model_name = log["model"]         # 1. 取 model 名字
    if model_name not in result:   # 2. 第一次见？
        result[model_name] = {"total_tokens": 0, "total_time_ms": 0}  # 3. 初始化
    result[model_name]["total_tokens"] += log["tokens"]   # 4. 累加 tokens
    result[model_name]["total_time_ms"] += log["time_ms"]  # 5. 累加 time_ms
print(result)
