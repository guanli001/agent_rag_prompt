from openai import OpenAI
import os

# 读取环境变量中的 api_key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    # 如果用通义千问，加 base_url：
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def get_completion(prompt, model="qwen3-max"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 值越低则输出文本随机性越低
    )
    return response.choices[0].message.content


# 示例：产品说明书
fact_sheet_chair = """
概述

    美丽的中世纪风格办公家具系列的一部分，包括文件柜、办公桌、书柜、会议桌等。
    多种外壳颜色和底座涂层可选。
    可选塑料前后靠背装饰（SWC-100）或10种面料和6种皮革的全面装饰（SWC-110）。
    底座涂层选项为：不锈钢、哑光黑色、光泽白色或铬。
    椅子可带或不带扶手。
    适用于家庭或商业场所。
    符合合同使用资格。

结构

    五个轮子的塑料涂层铝底座。
    气动椅子调节，方便升降。

尺寸

    宽度53厘米|20.87英寸
    深度51厘米|20.08英寸
    高度80厘米|31.50英寸
    座椅高度44厘米|17.32英寸
    座椅深度41厘米|16.14英寸

选项

    软地板或硬地板滚轮选项。
    两种座椅泡沫密度可选：中等（1.8磅/立方英尺）或高（2.8磅/立方英尺）。
    无扶手或8个位置PU扶手。

材料
外壳底座滑动件

    改性尼龙PA6/PA66涂层的铸铝。
    外壳厚度：10毫米。
    座椅
    HD36泡沫

原产国

    意大利
"""

# 提示：基于说明书创建营销描述
# prompt = f"""
# 你的任务是帮助营销团队基于技术说明书创建一个产品的营销描述。
#
# 根据```标记的技术说明书中提供的信息，编写一个产品描述。
#
# 技术说明: ```{fact_sheet_chair}```
# """
# response = get_completion(prompt)
# print(response)

# 优化后的 Prompt，说明面向对象，应具有什么性质且侧重于什么方面
prompt = f"""
您的任务是帮助营销团队基于技术说明书创建一个产品的零售网站描述。

根据```标记的技术说明书中提供的信息，编写一个产品描述。

该描述面向家具零售商，因此应具有技术性质，并侧重于产品的材料构造。

使用最多50个单词。

技术规格： ```{fact_sheet_chair}```
"""
response = get_completion(prompt)
print(response)
