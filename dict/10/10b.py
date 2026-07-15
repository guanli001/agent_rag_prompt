users = [
    {"id": 1, "name": "张三", "role": "admin"},
    {"id": 2, "name": "李四", "role": "user"},
    {"id": 3, "name": "王五", "role": "admin"},
]

# 构建 {id: name}，只包含 role == "admin" 的用户
# 期望：{1: "张三", 3: "王五"}

# dict_user = {user["id"]: user["name"] for user in users if user["role"] == "admin"}

dict_user = {}
for user in users:
    if user["role"] == "admin":
        dict_user[user["id"]] = user["name"]
print(dict_user)