"""
Day 7 挑战：LangChain 链式调用
把 Prompt + LLM + 输出解析串成一条链
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 通义千问兼容接口
    temperature=0.7
)

# ===== 链 1：翻译链 =====
print("=" * 60)
print("  链 1：翻译链")
print("=" * 60)

translate_prompt = ChatPromptTemplate.from_template(
    ("system", "你是一个专业翻译，只输出翻译结果"),
    ("human", "将以下内容翻译为{language}：\n{text}")
)

# 链：prompt → llm → 字符串解析
translate_chain = translate_prompt | llm | StrOutputParser()

result = translate_chain.invoke({
    "language": "日文",
    "text": "人工智能正在改变世界"
})
print(f"  翻译结果: {result}")

# ===== 链 2：分类 + 回复链（两步串联）=====
print("\n" + "=" * 60)
print("  链 2：分类 → 回复（两步串联）")
print("=" * 60)


# 第一步：分类
class ClassificationResult(BaseModel):
    """工单分类结果"""
    category: str = Field(description="类别：退货/换货/维修/咨询/投诉")
    urgency: str = Field(description="紧急程度：高/中/低")


classify_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是客服工单分类专家，请对用户工单进行分类"),
    ("human", "{ticket}")
])

classify_chain = (
        classify_prompt | llm.with_structured_output(ClassificationResult)
)

# 第二步：根据分类生成回复
reply_prompt = ChatPromptTemplate.from_messages(
    ("system", "你是专业客服，根据工单和分类结果生成回复，50字以内"),
    ("human", "工单：{ticket}\n分类：{category}\n紧急程度：{urgency}")
)

reply_chain = reply_prompt | llm | StrOutputParser()

# 手动串联两步
tickets = [
    "我买的手机用了两天屏幕就花了，要退货！",
    "请问这个产品支持七天无理由退货吗？",
]

for ticket in tickets:
    # 第一步：分类
    classification = classify_chain.invoke({"ticket": ticket})
    print(f"\n  工单: {ticket}")
    print(f"  分类: {classification.category}（紧急度: {classification.urgency}）")

    # 第二步：生成回复
    reply = reply_chain.invoke({
        "ticket": ticket,
        "category": classification.category,
        "urgency": classification.urgency,
    })
    print(f"  回复: {reply}")

# ===== 对比你 Day 6 的挑战 =====
print("\n" + "=" * 60)
print("  对比你 Day 6 的智能客服分类器")
print("=" * 60)
print("""
你 Day 6 手写的：
  - 自己写 PromptTemplate 类（{{var}} 占位符）
  - 自己写 extract_json 函数（3 层兜底）
  - 自己写 classify_ticket + generate_reply 函数
  - 手动调用、手动解析、手动传参
  - 约 150 行代码

LangChain 的：
  - ChatPromptTemplate 替代你的 PromptTemplate
  - with_structured_output 替代你的 extract_json
  - | 管道符替代手动调用和传参
  - 约 40 行代码

省了 100 行？是的。但那 100 行是你理解原理的代价。
现在你知道了每一行在做什么，再用框架就心里有数。
""")
