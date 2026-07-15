words = ["hello", "world", "python", "ai", "code"]

# 要求：把所有长度 >= 4 的单词全转大写
# 期望：["HELLO", "WORLD", "PYTHON", "CODE"]
result = []
for word in words:
    if len(word) >= 4:
        result.append(word.upper())
print(result)

result = [word.upper() for word in words if len(word) >= 4]
