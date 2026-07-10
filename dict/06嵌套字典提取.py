"""
AI Agent 调用 API 后，返回的 JSON 经常是嵌套结构
一层一层安全地取到最里面的 content。

📝 题目
写一个函数 get_ai_reply(api_response: dict) -> str：

从 api_response["choices"][0]["message"]["content"] 取出 AI 的回复文本
任何一层缺失或类型不对（比如没有 "choices" 这个 key、choices 是空列表、没有 "message" 等），
都要返回 {"error": "响应结构异常"}
"""


def get_ai_reply(api_response: dict) -> str:
    try:
        ai_messages = api_response["choices"][0]["message"]["content"]
        return ai_messages
    except (KeyError, TypeError, IndexError):
        return {"error": "响应结构异常"}

