"""智能体定义：问答助手 + 工具装配。"""

from agents import Agent, set_tracing_disabled
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

from .config import Settings
from .tools.calculator import calculator
from .tools.internet_search import internet_search

# 自定义端点无法上传 trace，禁用以免每次运行报错
set_tracing_disabled(True)

_INSTRUCTIONS = """你是「问答助手」，一个乐于助人的中文智能体。

职责：
1. 用简体中文清晰、准确地回答用户问题。
2. 遇到数学计算（哪怕很简单）时，调用 calculator 工具，不要心算。
3. 遇到时效性问题（新闻、天气、价格、近期事件）或你不确定的事实时，
   调用 internet_search 工具查证，并在回答末尾注明信息来源链接。
4. 一般知识问题直接回答，不必每次都调用工具。
5. 回答保持简洁；用户没有要求时不要展开长篇大论。
"""


def build_agent(settings: Settings) -> Agent:
    """根据配置创建问答助手 Agent。

    显式使用 OpenAIChatCompletionsModel：Agent 默认走 OpenAI 官方
    Responses API，而第三方兼容端点（DeepSeek/通义/Ollama 等）通常只
    实现 Chat Completions，必须显式降级才能通用。
    """
    client = AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key)
    model = OpenAIChatCompletionsModel(
        model=settings.model,
        openai_client=client,
    )
    return Agent(
        name="问答助手",
        instructions=_INSTRUCTIONS,
        model=model,
        tools=[calculator, internet_search],
    )
