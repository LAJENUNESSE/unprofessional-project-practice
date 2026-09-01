"""配置加载：从 .env 或环境变量读取接口配置。"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_DEFAULT_MODEL = "gpt-4o-mini"


def load_env() -> None:
    """加载 .env（幂等）。

    PyInstaller 冻结模式下模块在临时目录运行，dotenv 默认搜索会失效，
    改为从 exe 同目录读取 .env；源码运行保持默认搜索行为。
    """
    if getattr(sys, "frozen", False):
        load_dotenv(Path(sys.executable).resolve().parent / ".env")
    else:
        load_dotenv()


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    tavily_api_key: str = ""  # 为空时搜索工具回退 DuckDuckGo


class ConfigError(Exception):
    """配置缺失或不合法。"""


def load_settings() -> Settings:
    """加载配置；缺少必要项时抛出带中文指引的 ConfigError。"""
    load_env()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "sk-xxxx":
        raise ConfigError(
            "未配置 API Key。请复制 .env.example 为 .env，"
            "填入 OPENAI_API_KEY 后重试。"
        )

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"
    model = os.getenv("QA_AGENT_MODEL", "").strip() or _DEFAULT_MODEL
    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
    return Settings(
        base_url=base_url,
        api_key=api_key,
        model=model,
        tavily_api_key=tavily_api_key,
    )
