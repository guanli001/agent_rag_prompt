import tiktoken

# cl100k_base 是 GPT-4 用的编码器，拿来理解 Token 概念
enc = tiktoken.get_encoding("cl100k_base")

# 中文
cn = "你好，我正在学习 AI Agent 开发"
print(f"中文：{len(cn)} 字 → {len(enc.encode(cn))} Token")

# 英文
en = "Hello, I am learning AI Agent development"
print(f"英文：{len(en.split())} 词 → {len(enc.encode(en))} Token")

# 看看每个 Token 长什么样
print(f"中文拆分：{[enc.decode([t]) for t in enc.encode(cn)]}")

# 为什么同样意思的中文比英文 Token 多？这对成本和上下文窗口有什么影响？
