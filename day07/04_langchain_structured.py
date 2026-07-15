"""
Day 7 任务 4：LangChain 结构化输出
对比你 Day 6 手写的 extract_json + try/except 重试
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 通义千问兼容接口
    temperature=0.7
)

# ===== 你 Day 6 的方式 =====
print("=" * 60)
print("  对比：你 Day 6 手写的 JSON 提取")
print("=" * 60)
print("""
你手写的流程：
  1. 在 Prompt 里写"请输出 JSON 格式：{...}"
  2. 调用 LLM
  3. 手动提取 JSON（去 markdown 壳、find('{') 兜底）
  4. json.loads() 解析
  5. 失败了重试或返回 None

问题：
  - 模型不一定听话，可能输出多余文字
  - JSON 字段名可能和预期不一致
  - 需要手写大量兜底逻辑
""")

# ===== LangChain 的方式：用 Pydantic 定义输出 =====
print("=" * 60)
print("  LangChain 的方式：structured output")
print("=" * 60)


# 1. 用 Pydantic 定义你想要的输出结构
class SentimentResult(BaseModel):
    """情感分析结果"""
    sentiment: str = Field(description="情感倾向：正面/负面/中性")
    score: float = Field(description="情感分数，0到1之间")
    keywords: list[str] = Field(description="关键词列表")
    reason: str = Field(description="一句话说明判断依据")


# 2. 把模型绑定到这个结构
structured_llm = llm.with_structured_output(SentimentResult)

# 3. 定义Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个情感分析专家，请分析用户输入文本的情感倾向"),
    ("human", "{text}")
])

# 4. 组合成 chain 并调用
chain = prompt | structured_llm

reviews = [
    "这款手机拍照很清晰，电池续航也不错，但是价格有点贵。",
    "衣服质量很差，洗了一次就缩水了，客服态度也差。",
    "味道还行，环境一般，服务态度挺好的。",
]

for review in reviews:
    result = chain.invoke({"text": review})
    print(f"\n  评价: {review}")
    print(f"  情感: {result.sentiment}")
    print(f"  分数: {result.score}")
    print(f"  关键词: {result.keywords}")
    print(f"  理由: {result.reason}")
    print(f"  类型: {type(result)}")  # 直接是 SentimentResult 对象！

# ===== 对比总结 =====
print("\n" + "=" * 60)
print("  对比总结")
print("=" * 60)
print("""
你手写的：
  - Prompt 里描述 JSON 格式 → 模型可能不听
  - 手动提取 JSON（3 层兜底）
  - json.loads() → 返回 dict
  - 用 dict["类别"] 取值，字段名打错不报错
LangChain structured output：
  - 用 Pydantic 类定义结构 → 模型强制遵守
  - 自动解析 → 直接返回对象
  - 用 result.sentiment 取值，字段名打错直接报错
  - 类型安全、有自动补全
核心结论：你手写的 extract_json 是"事后补救"，
LangChain 的 structured output 是"事前约束"。
生产环境用后者更稳，但你必须理解前者才知道为什么会出错。
""")
