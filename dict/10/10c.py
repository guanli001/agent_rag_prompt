students = [
    {"name": "小明", "courses": ["语文", "数学", "英语"], "grade": 3},
    {"name": "小红", "courses": ["物理", "化学"], "grade": 5},
    {"name": "小刚", "courses": ["数学", "物理"], "grade": 4},
]

# 列出所有 grade >= 4 的学生选的课，去重
# 期望：{"物理", "化学", "数学"}
result = set()
for student in students:
    if student["grade"] >= 4:
        result.update(student["courses"])

result = {course for student in students if student["grade"] >= 4
          for course in student["courses"]
          }

print(result)
