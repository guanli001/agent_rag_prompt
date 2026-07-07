import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

messages = [{"role": "system", "content": "你是一个助手"}]

for i in range(1, 301):  # 循环50轮
    # 每轮添加一条用户消息
    messages.append({"role": "user", "content": f"这是第{i}轮对话..."})

    # 每轮添加一条AI回复（100个x模拟长回复）
    messages.append({"role": "assistant", "content": "模拟回复..." + "x" * 100})

    # 计算当前总Token数
    total = sum(len(enc.encode(m["content"])) for m in messages)

    # 每10轮打印一次
    if i % 10 == 0:
        print(f"第{i:2d}轮 → 总Token: {total:6d}")

print(f"\nDeepSeek 窗口: 65536 Token")

# 对话超窗了怎么办？有哪几种解决方案？
