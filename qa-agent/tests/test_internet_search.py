"""internet_search 工具单元测试：mock 网络，不发起真实请求。"""

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest

from agents.tool_context import ToolContext

from qa_agent.tools import internet_search as ws


@pytest.fixture(autouse=True)
def _default_ddgs_backend(monkeypatch):
    """默认屏蔽 Tavily，保证 DuckDuckGo 路径的测试不受本机 .env 影响。"""
    monkeypatch.setenv("TAVILY_API_KEY", "")


class FakeDDGS:
    """模拟 ddgs.DDGS 的最小接口。"""

    def __init__(self, results, raise_exc=None):
        self._results = results
        self._raise = raise_exc

    def __enter__(self):
        if self._raise:
            raise self._raise
        return self

    def __exit__(self, *args):
        return False

    def text(self, query, max_results=5):
        return self._results[:max_results]


class FakeResponse:
    """模拟 httpx.Response 的最小接口。"""

    def __init__(self, payload, status_error=None):
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error:
            raise self._status_error

    def json(self):
        return self._payload


def _fake_ddgs_results():
    return [
        {"title": "结果一", "href": "https://example.com/1", "body": "摘要一"},
        {"title": "结果二", "href": "https://example.com/2", "body": "摘要二"},
    ]


def _tavily_payload():
    return {
        "answer": "综合结论",
        "results": [
            {"title": "Tavily 结果一", "url": "https://example.com/t1", "content": "内容一"},
            {"title": "Tavily 结果二", "url": "https://example.com/t2", "content": "内容二"},
        ],
    }


def _invoke(query: str) -> str:
    """以模型视角调用工具并解包返回值。"""
    args_json = json.dumps({"query": query})
    ctx = ToolContext(
        context=None,
        tool_name="internet_search",
        tool_call_id="test-call-1",
        tool_arguments=args_json,
    )
    raw = asyncio.run(ws.internet_search.on_invoke_tool(ctx, args_json))
    # 字符串返回值会被 SDK 再包一层 JSON 引号
    return json.loads(raw) if isinstance(raw, str) and raw.startswith('"') else raw


class TestDuckDuckGoBackend:
    def test_search_formats_results(self):
        with patch.object(ws, "DDGS", lambda timeout: FakeDDGS(_fake_ddgs_results())):
            text = _invoke("测试")
        assert "结果一" in text
        assert "https://example.com/1" in text
        assert "结果二" in text

    def test_search_empty_results(self):
        with patch.object(ws, "DDGS", lambda timeout: FakeDDGS([])):
            assert "没有搜索到相关结果" in _invoke("测试")

    def test_search_network_error_returns_message(self):
        with patch.object(
            ws,
            "DDGS",
            lambda timeout: FakeDDGS([], raise_exc=ConnectionError("网络不通")),
        ):
            text = _invoke("测试")
        assert "搜索失败" in text
        assert "ConnectionError" in text


class TestTavilyBackend:
    def test_tavily_used_when_key_set(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        with patch.object(
            ws.httpx, "post", return_value=FakeResponse(_tavily_payload())
        ) as mock_post:
            text = _invoke("测试")
        assert "Tavily 结果一" in text
        assert "https://example.com/t1" in text
        assert "综合回答" in text  # Tavily 的 answer 字段被利用
        # 请求体应包含 api_key 与 query
        body = mock_post.call_args.kwargs["json"]
        assert body["api_key"] == "tvly-test-key"
        assert body["query"] == "测试"

    def test_tavily_empty_results(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        with patch.object(ws.httpx, "post", return_value=FakeResponse({"results": []})):
            assert "没有搜索到相关结果" in _invoke("测试")

    def test_tavily_failure_falls_back_to_ddgs(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        with patch.object(
            ws.httpx, "post", side_effect=httpx.HTTPError("Tavily 挂了")
        ), patch.object(ws, "DDGS", lambda timeout: FakeDDGS(_fake_ddgs_results())):
            text = _invoke("测试")
        assert "自动回退 DuckDuckGo" in text
        assert "结果一" in text  # 兜底结果正常返回

    def test_both_backends_fail_returns_message(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        with patch.object(
            ws.httpx, "post", side_effect=httpx.HTTPError("Tavily 挂了")
        ), patch.object(
            ws,
            "DDGS",
            lambda timeout: FakeDDGS([], raise_exc=ConnectionError("也不通")),
        ):
            text = _invoke("测试")
        assert "搜索失败" in text


def test_tool_metadata():
    """SDK 应正确生成工具的 JSON Schema，供模型调用。"""
    params = ws.internet_search.params_json_schema
    assert params["properties"]["query"]["type"] == "string"
    assert "query" in params.get("required", [])
