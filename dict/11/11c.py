api_response = {
    "model": "qwen-max",
    "usage": {"prompt_tokens": 120, "completion_tokens": 85, "total_tokens": 205},
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "你好！有什么可以帮助你的？",
                "tool_calls": [
                    {"name": "search", "arguments": {"query": "天气", "city": "北京"}},
                    {"name": "calculator", "arguments": {"expression": "2+3"}}
                ]
            }
        }
    ]
}

# 要求：提取所有 tool_calls 的 name，去重
# 期望：{"search", "calculator"}
result = set()
for choice in api_response["choices"]:
    for tool_call in choice["message"]["tool_calls"]:
        result.add(tool_call["name"])
print(result)
