"""
写一个函数 extract_tags(articles: list, min_score: float) -> set，
从文章列表中提取所有得分 ≥ min_score 的 tag，去重后返回。
"""

articles = [
    {"title": "GPT-4 发布", "tags": ["AI", "GPT", "LLM"], "score": 9.5},
    {"title": "Python 3.13", "tags": ["Python", "编程"], "score": 7.0},
    {"title": "RAG 入门", "tags": ["AI", "RAG", "LLM"], "score": 8.2},
    {"title": "旧闻", "tags": ["杂谈"], "score": 3.0},
]


def extract_tags(articles: list, min_score: float) -> set:
    return {
        tag
        for a in articles if a["score"] >= min_score
        for tag in a["tags"]
    }
