# import json, re
#
#
# def extract_json(text):
#     """加载数据"""
#     if not text:
#         return None
#     if not re.search(r"\{.*\}", text):
#         return None
#     return json.loads(re.search(r"\{.*\}", text).group())
#
#
# # extract_json("今天天气不错")  # 💥 AttributeError: 'NoneType' object has no attribute 'group'
#
# def count_words(text):
#     """统计单词数"""
#     if not text:
#         return {}
#     words = re.findall(r"\w+", text)
#     counts = {}
#     for word in words:
#         counts[word] = counts.get(word, 0) + 1
#     return counts
#
#
# class MessageBuilder:
"""
写一个 MessageBuilder 类，帮你拼对话消息。三个东西：
__init__：创建一个空列表
add(role, content)：往列表里加一条消息
get_messages()：返回整个列表
"""
import asyncio
import time

#
# class MessageBuilder:
#     def __init__(self):
#         self.messages = []
#
#     def add(self, role, content):
#         self.messages.append({"role": role, "content": content})
#
#     def get_messages(self):
#         return self.messages

"""
写两个函数：
fake_api_call(name, delay)：
用 await asyncio.sleep(delay) 模拟网络延迟，返回 f"{name} 完成，耗时 {delay}秒"
run_all()：
用 asyncio.gather 同时跑 3 个 fake_api_call，打印所有结果
"""


async def fake_api_call(name, delay):
    """模拟 API 调用"""
    await asyncio.sleep(delay)
    return f"{name} 完成，耗时 {delay}秒"


async def run_all():
    """运行所有任务"""
    tasks = [
        fake_api_call("任务1", 2),
        fake_api_call("任务2", 1),
        fake_api_call("任务3", 3),
    ]

    print(await asyncio.gather(*tasks))