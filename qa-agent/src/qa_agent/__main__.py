"""CLI 入口：多轮对话 + 连通性自检。

用法：
    python -m qa_agent            # 进入多轮对话
    python -m qa_agent --check    # 检查配置与接口连通性后退出
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from agents import Runner
from openai import AsyncOpenAI

from .agent import build_agent
from .config import ConfigError, Settings, load_settings

_BANNER = """\
==============================
  问答助手 · 训练任务 3
  模型: {model}
  接口: {base_url}
  工具: calculator, internet_search({search_backend})
  输入 /quit 退出，/reset 清空对话
==============================
"""

_RESET_HINT = "（对话历史已清空）"


def _print_settings(settings: Settings) -> None:
    backend = "tavily" if settings.tavily_api_key else "duckduckgo"
    print(_BANNER.format(
        model=settings.model,
        base_url=settings.base_url,
        search_backend=backend,
    ))


async def _run_check(settings: Settings) -> int:
    """向接口发送一条最小请求验证连通性。"""
    print(f"正在检测接口 {settings.base_url}（模型 {settings.model}）...")
    try:
        client = AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key)
        resp = await client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": "回复“OK”两个字母即可。"}],
            # 推理模型（如 glm 系列）会先输出 reasoning_content，
            # 上限太小会导致正文为空，这里给足余量
            max_tokens=512,
        )
        reply = (resp.choices[0].message.content or "").strip()
        print(f"接口连通成功，模型回复：{reply!r}")
        return 0
    except Exception as exc:
        print(f"接口连通失败：{exc.__class__.__name__}: {exc}")
        return 1


async def _chat_loop(settings: Settings) -> int:
    agent = build_agent(settings)
    history: list = []  # 完整对话历史（含工具调用记录），跨轮次传递
    _print_settings(settings)
    print("你好，我是问答助手。可以问我问题，或让我算数、查资料。\n")

    while True:
        try:
            user_input = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return 0

        if not user_input:
            continue
        if user_input in {"/quit", "/exit", "/q"}:
            print("再见！")
            return 0
        if user_input == "/reset":
            history.clear()
            print(_RESET_HINT)
            continue

        print("助手> ", end="", flush=True)
        try:
            result = Runner.run_streamed(
                agent,
                input=history + [{"role": "user", "content": user_input}],
                max_turns=8,
            )
            async for event in result.stream_events():
                # 只打印正文增量；推理模型的思考过程
                # (response.reasoning_summary_text.delta) 不打印
                if (
                    event.type == "raw_response_event"
                    and getattr(event.data, "type", "") == "response.output_text.delta"
                ):
                    print(event.data.delta, end="", flush=True)
            print("\n")
            history = result.to_input_list()
        except KeyboardInterrupt:
            print("\n（已中断本轮回答）")
        except Exception as exc:
            print(f"\n[出错] {exc.__class__.__name__}: {exc}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="问答助手（训练任务 3）")
    parser.add_argument(
        "--check",
        action="store_true",
        help="检查 .env 配置与接口连通性后退出",
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"[配置错误] {exc}")
        return 2

    if args.check:
        return asyncio.run(_run_check(settings))
    return asyncio.run(_chat_loop(settings))


if __name__ == "__main__":
    sys.exit(main())
