"""
场景：调大模型 API 时，对话历史是一个列表，每条消息是
{"role": "user", "content": "..."} 的格式。
现在要把这个列表转成人类可读的文本，用于打印日志或调试。
📝 题目
写一个函数 format_chat_history(messages: list) -> str：

输入是一个消息列表，每条消息格式为 {"role": "xxx", "content": "yyy"}
输出是一个格式化的多行字符串，每行格式为 [角色]: 内容
如果列表为空，返回 "(空对话)"
如果某条消息缺少 content，跳过该条（不输出）
"""


def format_chat_history(messages: list) -> str:
    if not messages:
        return "(空对话)"
    result = []
    for message in messages:
        if "content" not in message:
            continue
        result.append(f"[{message['role']}]: {message['content']}")
    return "\n".join(result)