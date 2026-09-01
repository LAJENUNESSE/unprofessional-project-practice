"""config 配置加载单元测试。"""

from unittest.mock import patch

import pytest

from qa_agent.config import ConfigError, load_settings


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("qa_agent.config.load_dotenv"):  # 阻止读取真实 .env
        with pytest.raises(ConfigError, match="API Key"):
            load_settings()


def test_placeholder_api_key_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-xxxx")
    with patch("qa_agent.config.load_dotenv"):
        with pytest.raises(ConfigError):
            load_settings()


def test_valid_config_with_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("QA_AGENT_MODEL", raising=False)
    with patch("qa_agent.config.load_dotenv"):
        s = load_settings()
    assert s.api_key == "sk-test-123"
    assert s.base_url == "https://api.openai.com/v1"
    assert s.model == "gpt-4o-mini"


def test_valid_config_custom_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abc")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("QA_AGENT_MODEL", "deepseek-chat")
    with patch("qa_agent.config.load_dotenv"):
        s = load_settings()
    assert s.base_url == "https://api.deepseek.com/v1"
    assert s.model == "deepseek-chat"


def test_tavily_key_loaded(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abc")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-xyz")
    with patch("qa_agent.config.load_dotenv"):
        assert load_settings().tavily_api_key == "tvly-xyz"
    monkeypatch.delenv("TAVILY_API_KEY")
    with patch("qa_agent.config.load_dotenv"):
        assert load_settings().tavily_api_key == ""
