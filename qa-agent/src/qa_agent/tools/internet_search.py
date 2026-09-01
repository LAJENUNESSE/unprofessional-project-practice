"""联网搜索工具：Tavily 优先（免费 Key、国内可直连），DuckDuckGo 兜底。

- 配置了 TAVILY_API_KEY 时调用 Tavily REST API（httpx）；
- 未配置或 Tavily 失败时回退 ddgs（DuckDuckGo，需代理）。

注意：工具名特意用 internet_search 而非 web_search——智谱 GLM 系列
API 存在与 web_search 同名的内置工具类型，函数工具重名会被上游拦截，
模型看不到该工具（表现为"我只有计算器"）。命名需避开内置工具名。
"""

import os

import httpx
from agents import function_tool
from ddgs import DDGS

from ..config import load_env

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_MAX_RESULTS = 5
_TIMEOUT_SECONDS = 15


def _tavily_search(query: str, api_key: str) -> str:
    """调用 Tavily API，返回格式化结果或抛出异常。"""
    resp = httpx.post(
        _TAVILY_ENDPOINT,
        json={
            "api_key": api_key,
            "query": query,
            "max_results": _MAX_RESULTS,
            "search_depth": "basic",
        },
        timeout=_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()

    answer = (data.get("answer") or "").strip()
    results = data.get("results") or []
    if not results and not answer:
        return "没有搜索到相关结果，请换个关键词再试。"

    lines = []
    for i, item in enumerate(results, start=1):
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        body = (item.get("content") or "").strip()
        lines.append(f"{i}. {title}\n   链接: {url}\n   摘要: {body}")
    text = "\n".join(lines)
    if answer:
        text = f"综合回答：{answer}\n\n来源：\n{text}"
    return text


def _ddgs_search(query: str) -> str:
    """调用 DuckDuckGo（ddgs），返回格式化结果或抛出异常。"""
    with DDGS(timeout=_TIMEOUT_SECONDS) as ddgs:
        hits = list(ddgs.text(query, max_results=_MAX_RESULTS))
    if not hits:
        return "没有搜索到相关结果，请换个关键词再试。"
    lines = []
    for i, hit in enumerate(hits, start=1):
        title = (hit.get("title") or "").strip()
        url = (hit.get("href") or hit.get("url") or "").strip()
        body = (hit.get("body") or hit.get("description") or "").strip()
        lines.append(f"{i}. {title}\n   链接: {url}\n   摘要: {body}")
    return "\n".join(lines)


def _do_search(query: str) -> str:
    load_env()  # 幂等；工具可能被独立调用（如冒烟脚本），不能假设主入口已加载 .env
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if api_key:
        try:
            return _tavily_search(query, api_key)
        except Exception as exc:
            # Tavily 失败不直接报错给模型，回退 DuckDuckGo 再试一次
            fallback = _ddgs_search(query)
            note = f"（提示：Tavily 搜索失败已自动回退 DuckDuckGo，原因：{exc.__class__.__name__}）\n\n"
            return note + fallback
    return _ddgs_search(query)


@function_tool
def internet_search(query: str) -> str:
    """在互联网上搜索与查询词相关的信息。

    当问题涉及实时信息、新闻、天气、价格、最新事件等你不确定的事实时，
    调用本工具获取网页摘要，并在回答中注明信息来源。

    Args:
        query: 搜索关键词，例如 "2026 年春节是几月几号"。
    """
    try:
        return _do_search(query)
    except Exception as exc:  # 双后端都失败时，把原因交还给模型处理
        return f"搜索失败：{exc.__class__.__name__}: {exc}"
