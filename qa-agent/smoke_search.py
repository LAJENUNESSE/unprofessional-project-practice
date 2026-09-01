"""联网搜索冒烟测试：不消耗 API 额度，只验证搜索通路。

用法（需先挂好代理并设置 HTTP_PROXY/HTTPS_PROXY 环境变量）：
    .venv\\Scripts\\python smoke_search.py [搜索词]
"""

import asyncio
import json
import sys

from agents.tool_context import ToolContext

from qa_agent.tools.internet_search import internet_search


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "Python 3.14 新特性"
    args = json.dumps({"query": query})
    ctx = ToolContext(
        context=None,
        tool_name="internet_search",
        tool_call_id="smoke",
        tool_arguments=args,
    )
    raw = await internet_search.on_invoke_tool(ctx, args)
    out = json.loads(raw) if raw.startswith('"') else raw
    print(out)


if __name__ == "__main__":
    asyncio.run(main())
