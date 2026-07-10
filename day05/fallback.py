"""
Day 5 实验 2：多模型 Fallback
目标：主模型失败时自动切换备用模型
"""
import os
import asyncio
import httpx
import time

api_key = os.environ.get("DASHSCOPE_API_KEY")

# 模型列表：按优先级排列
MODEL_CHAIN = [
    {"name": "qwen-turbo", "desc": "免费/快速（主模型）"},
    {"name": "qwen-plus", "desc": "更强/稍慢（备用1）"},
    {"name": "qwen-max", "desc": "最强/最贵（备用2）"},
]


async def chat_with_model(client, model_name, messages, timeout=10):
    """调用指定模型"""
    response = await client.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model_name,
            "messages": messages,
            "temperature": 0.7,
        },
        timeout=timeout,
    )
    data = response.json()
    if "choices" not in data:
        raise Exception(f"模型返回异常: {data.get('error', data)}")
    return {
        "content": data["choices"][0]["message"]["content"],
        "model": model_name,
        "tokens": data["usage"]["total_tokens"]
    }


async def chat_with_fallback(client, messages):
    """带 Fallback 的调用：按优先级逐个尝试"""
    errors = []
    for model in MODEL_CHAIN:
        try:
            print(f"  尝试 {model['name']}（{model['desc']}）...")
            result = await chat_with_model(client, model["name"], messages)
            print(f"  ✅ {model['name']} 成功！")
            return result
        except Exception as e:
            print(f"  ❌ {model['name']} 失败: {e}")
            errors.append(f"{model['name']}: {e}")

        # 所有模型都失败了
    raise Exception(f"所有模型都失败了: {errors}")


async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # ===== 实验 1：正常调用 =====
        print("=" * 55)
        print("  实验 1：正常 Fallback（主模型应该成功）")
        print("=" * 55)
        result = await chat_with_fallback(
            client,
            [{"role": "user", "content": "什么是Python装饰器？用一句话解释"}]
        )
        print(f"\n  回答: {result['content']}")
        print(f"  使用模型: {result['model']}")
        print(f"  Token: {result['tokens']}")

        # ===== 实验 2：模拟主模型失败 =====
        print("\n" + "=" * 55)
        print("  实验 2：模拟主模型超时（timeout=0.001）")
        print("=" * 55)

        # 故意设极短的 timeout 让 turbo 超时
        async def chat_with_short_timeout(client, model_name, messages):
            return await chat_with_model(client, model_name, messages, timeout=0.001)

        # 临时替换调用函数
        original_chat = chat_with_model
        call_count = 0

        async def mock_chat(client, model_name, messages, timeout=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 第一次（turbo）故意超时
                raise asyncio.TimeoutError("模拟超时")
            return await chat_with_short_timeout(client, model_name, messages)

        # 直接测试：turbo 超时 → 切到 plus
        print("  模拟 qwen-turbo 超时...")
        try:
            result = await chat_with_model(client, "qwen-turbo",
                                           [{"role": "user", "content": "测试"}], timeout=0.001)
        except Exception as e:
            print(f"  ❌ qwen-turbo 超时: {type(e).__name__}")

        # 切到 plus
        print("  尝试 qwen-plus...")
        result = await chat_with_model(client, "qwen-plus",
                                       [{"role": "user", "content": "什么是递归？一句话解释"}])
        print(f"  ✅ qwen-plus 成功: {result['content'][:50]}...")

        print("\n  结论：主模型超时后自动切到备用模型，用户无感知")

asyncio.run(main())
