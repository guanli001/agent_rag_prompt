# Day 7 行动计划 — LangChain 入门 + 手写 vs 框架对比 + 第 1 周总结

> 日期：2026 年 7 月 15 日（周三）
> 预计耗时：4-5 小时
> 前置条件：已完成 Day 1-6（LLM 认知 → API 调用 → 流式 → 并发封装 → Prompt 工程）
> 今日目标：安装 LangChain，用它重写你手写过的代码，理解框架帮你做了什么；完成第 1 周总结

---

## 压缩版整体时间线（供参考）

```
第1周（7/6~7/15）：认知校准 + API调用 + Prompt工程
  Day 1 ✅ LLM认知
  Day 2 ✅ Token + 首次API调用
  Day 3 ✅ 采样参数 + 多轮对话
  Day 4 ✅ SSE流式 + 异步调用
  Day 5 ✅ 并发+Fallback+封装LLMClient
  Day 6 ✅ Prompt工程全套
  Day 7 ← 今天：LangChain对比 + 第1周总结
第2周（7/16~7/22）：RAG 基础+进阶
第3周（7/23~7/29）：Agent 核心
第4周（7/30~8/5）：工程化
第5-6周（8/6~8/19）：项目实战
```

---

## 〇、热身：4 天没学，先做个快速复习（约 20 分钟）

休息了 4 天（周六周日 + 周一周二大雨），先回答这些问题，不用写代码，脑子里过一遍就行。答不上来的回去翻对应 Day 的笔记。

### 快速复习 quiz

| # | 问题 | 对应 Day | 你的答案 |
|---|------|---------|---------|
| 1 | LLM 是确定性系统还是概率性系统？这意味着什么？ | Day 1 | ？ |
| 2 | "你好世界" 大约消耗多少 Token？中文和英文哪个 Token 多？ | Day 2 | ？ |
| 3 | temperature=0 和 temperature=0.7 有什么区别？什么场景用哪个？ | Day 3 | ？ |
| 4 | 多轮对话中，messages 数组里的 role 有哪几种？ | Day 3 | ？ |
| 5 | SSE 流式输出中，每个 chunk 的格式是什么？`[DONE]` 代表什么？ | Day 4 | ？ |
| 6 | `asyncio.gather` 和 `for` 循环串行调用有什么区别？总耗时怎么算？ | Day 5 | ？ |
| 7 | `asyncio.Semaphore(3)` 解决什么问题？不加会怎样？ | Day 5 | ？ |
| 8 | Few-shot 和 Zero-shot 的区别？什么场景用哪个？ | Day 6 | ？ |
| 9 | CoT（思维链）为什么能提高准确率？代价是什么？ | Day 6 | ？ |
| 10 | 你手写的 `extract_json` 函数处理了哪几种情况？ | Day 6 练习 | ？ |

**能答出 7 道以上就过关，继续往下学。答不出 5 道的，先花 30 分钟翻一遍前几天的笔记。**

---

## 任务 1 ★☆☆ 必做：安装 LangChain + 第一次调用（约 1 小时）

### 1.1 为什么今天才学 LangChain？

回想一下你的学习路线规划里的核心原则：

```
手写 API 调用 → 再用 LangChain ChatModel
手写 RAG 流程 → 再用 LangChain Retriever
手写 ReAct 循环 → 再用 LangGraph
```

**先手写再用框架**——这样你才知道框架帮你做了什么、隐藏了什么。你已经手写了：
- `requests.post()` 调 API（Day 3）
- SSE 流式解析（Day 4）
- `asyncio` 并发 + Semaphore 限流 + Fallback（Day 5）
- `PromptTemplate` + `PromptManager`（Day 6）

现在用 LangChain 重写这些，你会**直观感受到框架的价值和代价**。

### 1.2 安装 LangChain

打开终端，执行：

```bash
# 安装 LangChain 核心包 + OpenAI 兼容包
pip install langchain langchain-openai

# 验证安装
python -c "import langchain; print(langchain.__version__)"
```

如果安装慢，加国内镜像：
```bash
pip install langchain langchain-openai -i https://mirrors.aliyun.com/pypi/simple/
```

### 1.3 用 LangChain 调用千问模型

你的千问 API 兼容 OpenAI 格式，所以可以用 `ChatOpenAI` 类，只需要改 `base_url`：

创建文件 `day07/01_langchain_basic.py`：

```python
"""
Day 7 任务 1：LangChain 第一次调用
目标：用 LangChain 调用千问模型，对比你之前手写的 requests.post()
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ===== 初始化模型 =====
# 对比你 Day 3 手写的：
#   response = requests.post(url, headers=headers, json={...})
# LangChain 只要一行：
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7
)

# ===== 最简单的调用 =====
print("=" * 60)
print("  实验 1：最简单的调用")
print("=" * 60)

response = llm.invoke([HumanMessage(content="你好，用一句话介绍你自己")])
print(f"  回复: {response.content}")
print(f"  Token 使用: {response.usage_metadata}")

# ===== 带 system 角色的调用 =====
print("\n" + "=" * 60)
print("  实验 2：带 system 角色")
print("=" * 60)

messages = [
    SystemMessage(content="你是一个毒舌但专业的编程老师，回答不超过50字"),
    HumanMessage(content="什么是递归？")
]
response = llm.invoke(messages)
print(f"  回复: {response.content}")

# ===== 对比：你 Day 3 手写的等价代码 =====
print("\n" + "=" * 60)
print("  对比：你手写的等价代码")
print("=" * 60)
print("""
# 你 Day 3 手写的（约 15 行）：
import requests
url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
headers = {"Authorization": f"Bearer {api_key}"}
data = {
    "model": "qwen-turbo",
    "messages": [
        {"role": "system", "content": "你是一个..."},
        {"role": "user", "content": "你好"}
    ],
    "temperature": 0.7
}
response = requests.post(url, headers=headers, json=data)
result = response.json()
content = result["choices"][0]["message"]["content"]

# LangChain 写的（约 3 行）：
llm = ChatOpenAI(model="qwen-turbo", ...)
response = llm.invoke([HumanMessage(content="你好")])
content = response.content
""")

print("  LangChain 帮你省了什么？")
print("  1. 不用手动拼 URL、headers、JSON body")
print("  2. 不用手动解析 response.json()['choices'][0]['message']['content']")
print("  3. 自动处理 Token 计数、错误重试等")
```

### 1.4 思考

- LangChain 的 `ChatOpenAI` 帮你封装了哪些你 Day 3 手动做的事情？
- `response.content` 和你手写的 `result["choices"][0]["message"]["content"]` 是不是同一个东西？
- `base_url` 参数的作用是什么？为什么需要它？

---

## 任务 2 ★☆☆ 必做：对比——流式输出（约 1 小时）

### 2.1 你手写的 vs LangChain 的

你 Day 4 手写流式输出时，需要手动解析 SSE 协议：`data: {...}\n\n`、`[DONE]` 标记、逐 chunk 拼接。

LangChain 把这些全封装了：

创建文件 `day07/02_langchain_stream.py`：

```python
"""
Day 7 任务 2：LangChain 流式输出
对比你 Day 4 手写的 SSE 解析
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7
)

# ===== LangChain 流式输出 =====
print("=" * 60)
print("  LangChain 流式输出")
print("=" * 60)
print("  开始生成...\n")

# 只要 .stream() 替代 .invoke()，就是流式
for chunk in llm.stream([HumanMessage(content="用100字解释什么是人工智能")]):
    print(chunk.content, end="", flush=True)

print("\n\n" + "=" * 60)
print("  对比：你 Day 4 手写的流式")
print("=" * 60)
print("""
# 你 Day 4 手写的（约 20 行）：
response = requests.post(url, headers=headers, json={..., "stream": True}, stream=True)
for line in response.iter_lines():
    if line:
        chunk = line.decode("utf-8")
        if chunk.startswith("data: "):
            data = chunk[6:]
            if data == "[DONE]":
                break
            chunk_data = json.loads(data)
            delta = chunk_data["choices"][0]["delta"].get("content", "")
            print(delta, end="", flush=True)

# LangChain 写的（2 行）：
for chunk in llm.stream([HumanMessage(content="...")]):
    print(chunk.content, end="", flush=True)
""")

print("  LangChain 帮你省了什么？")
print("  1. 不用手动解析 data: 前缀")
print("  2. 不用手动判断 [DONE]")
print("  3. 不用手动 json.loads 每个 chunk")
print("  4. 不用手动从 delta 里取 content")
print("  5. 自动处理连接断开、重连")
```

### 2.2 思考

- 你 Day 4 写了大约 20 行来处理流式，LangChain 只用了 2 行。**那 18 行逻辑消失了吗？**
  - 没消失，LangChain 在内部帮你做了。你需要知道的是**原理**，不是每次都手写
- 如果 LangChain 的流式出了 bug，你能看懂它的源码吗？**这就是"先手写再框架"的意义**

---

## 任务 3 ★★☆ 进阶：对比——Prompt 模板（约 1 小时）

### 3.1 你手写的 vs LangChain 的

你 Day 6 自己写了 `PromptTemplate` 类和 `PromptManager`，用 `{{var}}` 占位符避开 JSON 花括号冲突。LangChain 也有 `ChatPromptTemplate`，看看它怎么做的：

创建文件 `day07/03_langchain_prompt.py`：

```python
"""
Day 7 任务 3：LangChain Prompt 模板
对比你 Day 6 手写的 PromptTemplate + PromptManager
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0
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
```

### 3.2 思考

- `prompt | llm` 这个管道符 `|` 是什么意思？（提示：Python 的 `__or__` 魔术方法）
- 你手写的 `PromptManager` 和 LangChain 的 `FewShotChatMessagePromptTemplate` 解决的是同一个问题吗？
- LangChain 的 `ChatPromptTemplate` 怎么处理 JSON 花括号冲突的？（提示：用 `{{}}` 转义，和你思路一样）

---

## 任务 4 ★★☆ 进阶：对比——结构化输出（约 1 小时）

### 4.1 你手写的 JSON 提取 vs LangChain 的结构化输出

你 Day 6 手写了 `extract_json` 函数，处理了 markdown 包裹、前后废话、无 JSON 三种情况。你还手写了 try/except + 重试逻辑。

LangChain 提供了更优雅的方案——**用 Pydantic 定义输出格式，让模型强制遵守**：

创建文件 `day07/04_langchain_structured.py`：

```python
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
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0
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

# 3. 定义 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个情感分析专家，请分析用户输入文本的情感倾向"),
    ("human", "{text}")
])

# 4. 组合成 chain 并调用
chain = prompt | structured_llm

reviews = [
    "这款手机拍照很清晰，电池续航也不错，但是价格有点贵",
    "衣服质量很差，洗了一次就缩水了，客服态度也差",
    "味道还行，环境一般，服务态度挺好的"
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
```

### 4.2 思考

- `with_structured_output()` 底层是怎么实现的？（提示：它就是在 Prompt 里自动加了格式约束 + 自动解析）
- Pydantic 的 `Field(description=...)` 有什么用？（提示：这个描述会被传给 LLM，帮助它理解每个字段应该填什么）
- 如果你不用 LangChain，怎么用纯 Python 实现类似的效果？（提示：Pydantic + 自定义 Prompt + extract_json）

---

## 任务 5 ★★★ 挑战（选做）：LangChain 链式调用（约 1 小时）

### 5.1 什么是 Chain（链）？

你已经见过 `prompt | llm` 这种写法了。这就是 LangChain 的 **LCEL（LangChain Expression Language）**——用管道符 `|` 把多个步骤串起来：

```
用户输入 → Prompt 模板 → LLM 调用 → 输出解析 → 结果
```

创建文件 `day07/05_langchain_chain.py`：

```python
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
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7
)

# ===== 链 1：翻译链 =====
print("=" * 60)
print("  链 1：翻译链")
print("=" * 60)

translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业翻译，只输出翻译结果"),
    ("human", "将以下内容翻译为{language}：\n{text}")
])

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
    classify_prompt
    | llm.with_structured_output(ClassificationResult)
)

# 第二步：根据分类生成回复
reply_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是专业客服，根据工单和分类结果生成回复，50字以内"),
    ("human", "工单：{ticket}\n分类：{category}\n紧急程度：{urgency}")
])

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
        "urgency": classification.urgency
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
```

### 5.2 思考

- `prompt | llm | StrOutputParser()` 这条链里，数据是怎么流动的？（提示：prompt 输出消息 → llm 接收消息输出 AIMessage → StrOutputParser 接收 AIMessage 输出字符串）
- 如果中间某一步出错了，你怎么调试？（提示：可以拆开一步步调用，不用链）
- 你觉得 LCEL 的 `|` 管道符和 Linux 的 `|` 管道符像不像？

---

## 任务 6 ★☆☆ 必做：第 1 周总结笔记（约 30 分钟）

创建文件 `day07/week1_summary.md`，用你自己的话写。**不许抄下面的提示，自己组织语言。**

### 总结框架（填空式）

```markdown
# 第 1 周总结（Day 1-7）

## 我学会了什么

### 1. LLM 的本质
（用 2-3 句话说说 LLM 和传统编程的区别）

### 2. API 调用
（说说从发 HTTP 请求到拿到模型回复，中间经历了哪些步骤）

### 3. 流式输出
（SSE 是什么？为什么要流式？）

### 4. 异步并发
（asyncio.gather 和 Semaphore 解决什么问题？）

### 5. Prompt 工程
（你觉得最重要的 3 个 Prompt 技巧是什么？为什么）

### 6. LangChain
（LangChain 帮你封装了什么？代价是什么？）

## 我踩过的坑
（列 3 个你印象最深的 bug 或坑）

## 我还不懂的地方
（列 2-3 个你觉得还没完全理解的概念）

## 手写代码练习记录
- extract_json：2 轮改对，学会了 re.search 返回 None 要判断
- count_words：3 轮改对，学会了列表和字典的区别
- MessageBuilder：1 次过，掌握了 class 基本写法
- async 并发：2 轮改对，掌握了 async/await/gather

## 下周目标
（用一句话说说你期待 RAG 能解决什么问题）
```

---

## 10 道面试题（第 1 周综合）

### Q1：LLM 是确定性系统还是概率性系统？这对开发有什么影响？

**参考答案**：
LLM 是**概率性系统**——同样的输入可能得到不同输出（temperature > 0 时）。

对开发的影响：
1. **不能直接信任输出**：必须做格式校验、重试、降级兜底
2. **需要稳定性手段**：temperature=0 降低随机性、Few-shot 固定输出格式
3. **测试方式不同**：传统软件测"输入 A 是否输出 B"，AI 应用测"输出是否在合理范围内"

---

### Q2：Token 是什么？为什么开发 AI 应用必须关注 Token？

**参考答案**：
Token 是 LLM 处理文本的最小单位。中文约 1 字 ≈ 1-2 Token，英文约 1 词 ≈ 1-1.5 Token。

必须关注的原因：
1. **成本**：API 按 Token 计费，Prompt + Completion 都算
2. **上下文窗口限制**：模型一次能处理的最大 Token 数有限（如 8K、32K、128K），超出会截断
3. **性能**：Token 越多，响应越慢
4. **记忆管理**：多轮对话时，历史消息全部计入 Token，需要在适当时候截断

---

### Q3：SSE 流式输出的原理是什么？

**参考答案**：
SSE（Server-Sent Events）是 HTTP 长连接协议。客户端发一个请求，服务端不一次性返回，而是持续推送数据块。

流程：
1. 客户端发送请求，`stream=True`
2. 服务端返回 `text/event-stream` 类型的响应
3. 数据以 `data: {JSON}\n\n` 格式逐块推送
4. 每个 chunk 包含一小段生成的文本（delta）
5. 最后推送 `data: [DONE]` 表示结束

好处：用户不用等全部生成完才看到内容，逐字显示，体验好。

---

### Q4：asyncio.gather 和 for 循环调用有什么区别？

**参考答案**：
- **for 循环串行**：一个任务完成才开始下一个，总耗时 = 所有任务耗时之和
- **asyncio.gather 并发**：所有任务同时开始，总耗时 ≈ 最慢那个任务的耗时

```python
# 串行：3 个任务各 2 秒 → 总耗时 6 秒
for task in tasks:
    await task

# 并发：3 个任务各 2 秒 → 总耗时 2 秒
await asyncio.gather(*tasks)
```

注意：`asyncio.gather` 只是"同时等待"，不是"同时执行"——Python 的 asyncio 是单线程协程，IO 等待时切换到其他任务。

---

### Q5：asyncio.Semaphore 的作用是什么？不加会怎样？

**参考答案**：
Semaphore（信号量）限制同时运行的协程数量。

```python
semaphore = asyncio.Semaphore(3)  # 最多 3 个同时执行

async def task():
    async with semaphore:  # 超过 3 个就排队等
        await api_call()
```

不加的后果：
- 同时发 100 个请求 → API 返回 429（Too Many Requests）
- 被封 IP 或 API Key
- 内存暴涨

---

### Q6：Few-shot 和 Zero-shot 的区别？给几个例子最合适？

**参考答案**：
- **Zero-shot**：不给例子，直接让模型回答
- **Few-shot**：在 Prompt 中给几个输入-输出示例，让模型照着模式回答

Few-shot 例子数量建议：
- 1-3 个：大多数场景够用
- 3-5 个：复杂分类、格式严格场景
- 5 个以上：边际收益递减，Token 成本高

选择原则：覆盖不同类别、格式完全一致、不要太过相似。

---

### Q7：CoT（思维链）的原理是什么？什么时候该用？

**参考答案**：
CoT 是让模型在给出最终答案前，先输出中间推理步骤。

原理：LLM 是自回归模型，每个 Token 的生成依赖前面的 Token。直接回答时模型没有"思考空间"；加 CoT 后，推理过程的 Token 成为后续答案的上下文，帮助模型"理清思路"。

使用场景：
- ✅ 数学计算、逻辑推理、多步骤问题
- ❌ 简单分类、翻译、事实问答（不需要，浪费 Token）

代价：消耗更多 Token 和时间。

---

### Q8：如何让模型稳定输出 JSON？

**参考答案**：
从低级到高级的方法：

1. **明确要求**：Prompt 中写"只输出 JSON，不要其他内容"
2. **给示例**：Few-shot 方式给一个标准 JSON 示例
3. **temperature=0**：减少随机性
4. **代码兜底**：try/except + 正则提取（`re.search(r'\{.*\}', text)`）
5. **Pydantic 校验**：定义 Schema，解析后自动校验字段
6. **LangChain structured output**：`llm.with_structured_output(Schema)`，框架自动处理

生产环境推荐 5+6，但必须理解 1-4 的原理。

---

### Q9：LangChain 解决了什么问题？不用 LangChain 行不行？

**参考答案**：
LangChain 解决的核心问题：**减少重复代码，提供标准化抽象**。

| 你手写的 | LangChain 的 |
|---------|-------------|
| `requests.post()` 拼 URL/headers/body | `ChatOpenAI` 一行搞定 |
| 手动解析 SSE 协议 | `.stream()` 自动处理 |
| 自定义 PromptTemplate | `ChatPromptTemplate` |
| 手写 extract_json | `with_structured_output()` |
| 手写并发 + 重试 | 内置支持 |

**不用 LangChain 行不行？** 完全行。你 Day 1-6 已经证明了。但项目大了之后，手写维护成本高，LangChain 提供了成熟的生态（文档加载器、向量库集成、Agent 编排等）。

代价：学习成本、黑箱程度增加、版本更新频繁。

---

### Q10：你手写的 LLMClient 和 LangChain 的 ChatOpenAI 有什么区别？

**参考答案**：
你手写的 `LLMClient`（Day 5）：
- ✅ 你完全理解每一行代码
- ✅ 支持多模型 Fallback（主模型挂了自动切备用）
- ✅ 支持 Semaphore 并发控制
- ✅ 支持重试
- ❌ 功能有限，需要自己扩展

LangChain 的 `ChatOpenAI`：
- ✅ 自动重试、错误处理
- ✅ 支持 structured output
- ✅ 支持流式、异步、批量
- ✅ 与 LangChain 生态无缝集成
- ❌ 内部逻辑复杂，出 bug 不容易调

**核心结论**：你手写的版本是 LangChain 的简化版。理解了你自己的版本，就能看懂 LangChain 在做什么。先手写再框架，这是正确的学习路径。

---

## 今日完成检查清单

- [ ] 完成快速复习 quiz，能答出 7 道以上
- [ ] 安装 LangChain，运行 `01_langchain_basic.py`
- [ ] 运行 `02_langchain_stream.py`，对比流式输出
- [ ] 运行 `03_langchain_prompt.py`，对比 Prompt 模板
- [ ] 运行 `04_langchain_structured.py`，对比结构化输出
- [ ] （选做）完成挑战：`05_langchain_chain.py`
- [ ] 写完 `week1_summary.md`（第 1 周总结）
- [ ] 看一遍 10 道面试题，能答出至少 7 道
- [ ] 代码 push 到 GitHub：
  ```bash
  git add day07/
  git commit -m "day07: LangChain入门-手写vs框架对比-第1周总结"
  git push
  ```

---

> 📌 今天做完，第 1 周就全部结束了！
> Day 1-7 你完成了：LLM 认知 → API 调用 → 流式 → 并发封装 → Prompt 工程 → LangChain 对比
> 明天 Day 8 开始第 2 周：**RAG 知识库**——让 AI 基于你的私有知识回答问题！
