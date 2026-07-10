"""
Day 6 挑战：智能客服分类器
组合：角色 + Few-shot + CoT + JSON + 模板管理 + 并发
"""

import os
import json
import asyncio
import sys
from typing import List, Dict, Any, Optional

# 添加父目录到路径，以便导入 LLMClient
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from day05.llm_client import LLMClient


class PromptTemplate:
    """提示词模板类"""

    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        """使用自定义占位符 {{变量名}} 替换，避免与 JSON 大括号冲突"""
        result = self.template
        for key, value in kwargs.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
        return result

    def to_messages(self, role: str = "user", **kwargs) -> List[Dict[str, str]]:
        """转换为消息格式"""
        return [{"role": role, "content": self.format(**kwargs)}]


class PromptManager:
    """提示词管理器"""

    def __init__(self):
        self.templates = {}
        self._init_templates()

    def _init_templates(self):
        """初始化所有模板"""

        # 1. 客服分类器模板
        self.templates["customer_service_classifier"] = PromptTemplate("""
你是一个专业的客服工单分类专家。

# 任务
请对用户提交的客服工单进行分类，判断属于哪个类别。

# 类别定义
1. 退货：用户要求退货退款
2. 换货：用户要求更换商品
3. 维修：用户要求维修服务
4. 咨询：用户咨询产品或服务问题
5. 投诉：用户表达不满或投诉

# 分类规则
- 如果用户明确提到"退货""退款""不要了"，分类为"退货"
- 如果用户提到"换""更换""换个新的"，分类为"换货"
- 如果用户提到"修""维修""坏了"，分类为"维修"
- 如果用户只是询问信息，分类为"咨询"
- 如果用户情绪激动且表达不满，分类为"投诉"

# Few-shot 示例
示例1：
工单：我的手机开机一直卡在logo界面，无法使用，需要维修
分类：维修
理由：明确提到"维修"，且描述了功能故障

示例2：
工单：这个颜色和图片不一样，我要退货
分类：退货
理由：明确提到"退货"

示例3：
工单：请问这个产品支持7天无理由退货吗？
分类：咨询
理由：用户在询问政策，没有要求具体操作

# 思考过程（CoT）
请按以下步骤思考：
1. 理解用户的核心诉求是什么
2. 识别关键词（退货/换货/维修/咨询/投诉）
3. 判断情绪倾向（中性/不满/愤怒）
4. 给出最终分类

# 工单内容
{{ticket}}

# 输出格式（必须是严格的JSON格式）
{
  "类别": "退货/换货/维修/咨询/投诉",
  "置信度": 0.0-1.0,
  "理由": "一句话说明分类依据",
  "关键词": ["关键词1", "关键词2"],
  "情绪": "中性/不满/愤怒"
}

请只输出JSON，不要包含其他内容。
""")

        # 2. 回复生成模板
        self.templates["reply_generator"] = PromptTemplate("""
你是一个专业的客服人员，需要根据工单分类生成回复。

# 工单内容
{{ticket}}

# 分类结果
{{classification}}

# 生成要求
- 语气专业、礼貌、耐心
- 针对分类给出具体解决方案
- 如果投诉，先道歉再解决
- 字数控制在50-100字

请直接输出回复内容。
""")

    def get(self, name: str) -> PromptTemplate:
        """获取模板"""
        if name not in self.templates:
            raise KeyError(
                f"模板 '{name}' 不存在，可用模板: {list(self.templates.keys())}"
            )
        return self.templates[name]


def extract_json(content: str) -> dict:
    """
    从模型输出中提取 JSON，处理三种情况：
    1. markdown 代码块包裹（```json ... ``` 或 ``` ... ```）
    2. 纯 JSON 字符串
    3. JSON 前后有废话文本
    """
    content = content.strip()

    # 情况1：markdown 代码块
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    # 情况2：直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 情况3：兜底——找第一个 { 和最后一个 } 之间的内容
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(content[start:end + 1])

    raise json.JSONDecodeError("无法从输出中提取JSON", content, 0)


async def classify_ticket(
        client: LLMClient,
        pm: PromptManager,
        ticket: str,
        ticket_id: int = 1,
) -> Optional[Dict]:
    """分类单个工单"""
    print(f"\n【工单 {ticket_id}】{ticket}")

    messages = pm.get("customer_service_classifier").to_messages(
        role="user",
        ticket=ticket,
    )

    try:
        response = await client.chat(
            messages=messages,
            temperature=0.3,
            max_tokens=500,
        )

        # 用统一的 JSON 提取函数
        result = extract_json(response.content)

        print(f"  📊 分类结果: {result.get('类别', '未知')}")
        print(f"  📈 置信度: {result.get('置信度', 0)}")
        print(f"  💡 理由: {result.get('理由', '无')}")
        print(f"  🔑 关键词: {', '.join(result.get('关键词', []))}")
        print(f"  😊 情绪: {result.get('情绪', '中性')}")

        return result

    except json.JSONDecodeError as e:
        print(f"  ❌ JSON解析失败: {e}")
        print(f"  📝 原始输出: {response.content[:200]}...")
        return None
    except Exception as e:
        print(f"  ❌ 分类失败: {e}")
        return None


async def classify_all(
        client: LLMClient,
        pm: PromptManager,
        tickets: List[str],
        max_concurrent: int = 3,
) -> List[Optional[Dict]]:
    """
    并发分类所有工单，使用 Semaphore 控制并发数。
    相比串行 for 循环，4 个工单同时发，总耗时大幅降低。
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def classify_with_limit(ticket: str, idx: int) -> Optional[Dict]:
        async with semaphore:
            return await classify_ticket(client, pm, ticket, idx)

    tasks = [
        classify_with_limit(ticket, i)
        for i, ticket in enumerate(tickets, 1)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # gather 返回的结果可能包含异常对象，统一处理
    cleaned = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  ❌ 工单 {i + 1} 发生异常: {result}")
            cleaned.append(None)
        else:
            cleaned.append(result)

    return cleaned


async def generate_reply(
        client: LLMClient,
        pm: PromptManager,
        ticket: str,
        classification: Optional[Dict],
) -> str:
    """根据工单和分类结果生成客服回复"""
    if not classification:
        return "（分类结果缺失，无法生成回复）"

    messages = pm.get("reply_generator").to_messages(
        role="user",
        ticket=ticket,
        classification=json.dumps(classification, ensure_ascii=False),
    )

    try:
        response = await client.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
        return response.content
    except Exception as e:
        return f"（生成回复失败：{e}）"


async def main():
    """主函数"""
    print("=" * 70)
    print("  Day 6 挑战：智能客服分类器")
    print("  组合：角色 + Few-shot + CoT + JSON + 模板管理 + 并发")
    print("=" * 70)

    # 初始化
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 请设置环境变量 DASHSCOPE_API_KEY")
        return

    client = LLMClient(
        api_key=api_key,
        models=["qwen-turbo", "qwen-plus"],
        max_concurrent=3,
        max_retries=2,
    )

    pm = PromptManager()

    # 测试工单
    tickets = [
        "我买的手机用了两天屏幕就花了，这是什么质量？我要退货！",
        "电脑开机黑屏，风扇一直转，请问怎么处理？",
        "这个产品比官网贵了200块，你们价格保护吗？",
        "收到货发现是坏的，我要换一个！",
    ]

    # ---- 第一阶段：并发分类所有工单 ----
    print("\n" + "=" * 70)
    print("  第一阶段：并发工单分类")
    print("=" * 70)

    results = await classify_all(client, pm, tickets, max_concurrent=3)

    # 组装分类成功的工单
    classified = []
    for i, (ticket, result) in enumerate(zip(tickets, results), 1):
        if result:
            classified.append({
                "ticket": ticket,
                "classification": result,
                "ticket_id": i,
            })
    print("-" * 70)
    print(f"  分类完成：{len(classified)}/{len(tickets)} 个工单成功")

    # ---- 第二阶段：并发生成回复 ----
    print("\n" + "=" * 70)
    print("  第二阶段：生成客服回复")
    print("=" * 70)

    async def process_reply(item):
        ticket = item["ticket"]
        classification = item["classification"]
        ticket_id = item["ticket_id"]

        print(f"\n【工单 {ticket_id}】{ticket[:30]}...")
        print(f"  分类: {classification.get('类别')}")

        reply = await generate_reply(client, pm, ticket, classification)
        print(f"  💬 回复: {reply}")
        print("-" * 70)
        return reply

    # 回复生成也可以并发
    reply_tasks = [process_reply(item) for item in classified]
    await asyncio.gather(*reply_tasks)

    # 关闭客户端
    await client.close()
    print("\n✅ 所有任务完成！")


if __name__ == "__main__":
    asyncio.run(main())
