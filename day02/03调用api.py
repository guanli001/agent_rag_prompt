import os, requests, json

api_key = os.environ.get("DEEPSEEK_API_KEY")

response = requests.post(
    "https://api.deepseek.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个Python编程助手"},
            {"role": "user", "content": "解释什么是装饰器"}
        ],
        "temperature": 0.7
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
print(f"输入: {result['usage']['prompt_tokens']} Token")
print(f"输出: {result['usage']['completion_tokens']} Token")
