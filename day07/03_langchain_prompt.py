"""
Day 7 任务 3：LangChain Prompt 模板
对比你 Day 6 手写的 PromptTemplate + PromptManager
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 通义千问兼容接口
    temperature=0.7
)

# ===== 你 Day 6 手写的 =====
print("=" * 60)
print("  对比 1：简单模板")
print("=" * 60)

# 你手写的：
# template = PromptTemplate("请翻译以下内容为{lang}：{text}")
# messages = template.to_messages(role="user", lang="英文", text="你好")

# LangChain 的：
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业翻译"),
    ("user", "请将以下内容翻译为{lang}，只输出翻译结果：\n\n{text}")
])

# 填充变量并调用
chain = prompt | llm
response = chain.invoke({"lang": "英文", "text": "人工智能正在改变世界"})
print(f"  翻译结果: {response.content}")

# ===== 对比 2：Few-shot 模板 =====
print("\n" + "=" * 60)
print("  对比 2：Few-shot")
print("=" * 60)

# 你 Day 6 手写的：把示例写死在模板字符串里
# LangChain 的：用 MessagesPlaceholder 灵活管理
from langchain_core.prompts import FewShotChatMessagePromptTemplate

# 定义示例
examples = [
    {"input": "这家餐厅菜品很新鲜", "output": "正面"},
    {"input": "等了一个小时才送到", "output": "负面"},
    {"input": "味道一般般", "output": "中性"},
]

# 示例模板
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

# Few-shot 模板
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 组合成完整 prompt
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "请判断文本的情感倾向，只回答：正面/负面/中性"),
    few_shot_prompt,
    ("human", "{input}")
])

# 调用
chain = final_prompt | llm
test_texts = ["这个产品太好用了", "等了三天才发货，差评", "一般般吧"]
for text in test_texts:
    response = chain.invoke({"input": text})
    print(f"  {text} → {response.content}")

# ===== 对比总结 =====
print("\n" + "=" * 60)
print("  对比总结")
print("=" * 60)
print("""
你手写的 PromptTemplate：
  - 用 {{var}} 自定义占位符避开 JSON 冲突
  - 用 PromptManager 集中管理
  - format() 方法手动替换
  - 优点：简单、可控
  - 缺点：功能少，复杂场景要自己写很多

LangChain 的 ChatPromptTemplate：
  - 用 {var} 占位符（它会自动处理转义）
  - 支持多角色消息（system/user/ai）
  - 支持 Few-shot 示例管理
  - 支持 | 管道符串联（prompt | llm）
  - 优点：功能强大、生态丰富
  - 缺点：学习成本高、黑箱程度增加

核心结论：你手写的模板思路和 LangChain 完全一致。
LangChain 只是在你的基础上加了更多功能和封装。
理解了你自己的版本，就能看懂 LangChain 在做什么。
""")
