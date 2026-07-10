"""
安全解析 API 响应
写一个函数 safe_parse_response(response_text: str) -> dict：

情况	返回值
正常 JSON（如 '{"status":"ok"}'）	解析后的字典
输入是 None	{"error": "响应为空，请检查网络连接"}
输入是空字符串 ""	{"error": "响应为空字符串"}
输入不是合法 JSON	{"error": f"JSON 解析失败: {具体错误信息}"}
JSON 解析成功但不是 dict（比如是个列表 "[1,2,3]"）	{"error": "响应格式异常：期望字典类型，实际得到 list"}
"""
import json


def safe_parse_response(response_text: str) -> dict:
    try:
        if response_text is None:
            return {"error": "响应为空，请检查网络连接"}
        elif response_text == "":
            return {"error": "响应为空字符串"}
        else:
            response_json = json.loads(response_text)
            if not isinstance(response_json, dict):
                return {"error": f"响应格式异常：期望字典类型，实际得到 {type(response_json).__name__}"}
            return response_json
    except json.JSONDecodeError as e:
        return {"error": f"JSON 解析失败: {e}"}
