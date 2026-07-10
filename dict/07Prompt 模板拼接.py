"""
场景：做 Prompt 工程时，经常要把用户输入、系统指令、
对话历史拼接成一个完整的 prompt 发送给模型。这个练习就是写一个简单的 prompt 模板引擎。

📝 题目
写一个函数 build_prompt(template: str, **kwargs) -> str：

template 是包含占位符的模板字符串，占位符格式为 {变量名}
**kwargs 是变量名=值的键值对
把模板里的占位符替换为对应的值
如果 kwargs 里缺少模板需要的某个变量，不要让它报 KeyError，而是把那个占位符保留原样（不替换）
"""


def build_prompt(template: str, **kwargs) -> str:
    result = template
    for key, value in kwargs.items():
        placeholder = f"{{{{{key}}}}}"
        result = result.replace(placeholder, str(value))
    return result
