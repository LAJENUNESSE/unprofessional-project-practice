# 问答助手（训练任务 3）

基于 [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) 开发的简单智能体：
**AI 对话推理 + 工具调用（计算器 / 联网搜索）**，支持多轮对话与流式输出。

> 任何 OpenAI Chat Completions 兼容接口均可使用（OpenAI / DeepSeek / 通义 / Kimi / GLM / Ollama 等），
> 全部配置由 `.env` 驱动，不改代码即可切换供应商。

## 功能总览

| 能力 | 说明 |
|------|------|
| 多轮对话 | 保留完整上下文（含工具调用记录），`/reset` 清空，`/quit` 退出 |
| 计算器工具 | 四则运算、幂、整除、取模、括号、pi/e 常量；AST 白名单解析，防注入防指数爆炸 |
| 联网搜索 | **Tavily 优先**（配 Key 后自动启用，国内可直连）；未配 Key 或失败自动回退 DuckDuckGo（需代理）；回答附来源链接 |
| 流式输出 | 回答逐字打印；自动过滤推理模型的思考过程，只显示正文 |
| 连通性自检 | `--check` 一键验证 LLM 接口配置；`smoke_search.py` 单独验证搜索通路 |
| 测试 | 38 个 pytest 单测，无需 API Key，秒级完成 |

## 快速开始

### 1. 安装依赖

需要 Python 3.10+，推荐 [uv](https://docs.astral.sh/uv/)：

```bash
# 方式一：uv（推荐；国内直连 PyPI 会失败，必须带镜像）
uv venv
UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple" uv pip install -e ".[dev]"

# 方式二：pip
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置接口（.env）

```bash
cp .env.example .env    # PowerShell: Copy-Item .env.example .env
```

编辑 `.env`，四个字段：

| 字段 | 必填 | 说明 |
|------|------|------|
| `OPENAI_BASE_URL` | ✅ | LLM 接口地址（OpenAI 兼容端点，见下表） |
| `OPENAI_API_KEY` | ✅ | 对应接口的 API Key |
| `QA_AGENT_MODEL` | ✅ | 模型名，写接口方自己的名字 |
| `TAVILY_API_KEY` | 推荐 | [app.tavily.com](https://app.tavily.com) 免费注册；留空则搜索回退 DuckDuckGo（需代理） |

常用接口地址参考：

| 供应商 | `OPENAI_BASE_URL` | 模型名示例 |
|--------|-------------------|-----------|
| OpenAI 官方 | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| Kimi | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| 智谱 | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |
| 本地 Ollama | `http://localhost:11434/v1` | `qwen2.5:7b` |

`.env` 在 `.gitignore` 中，不会被提交；仓库里只有不含真实密钥的 `.env.example`。

### 3. 运行

```bash
# 先自检连通性（推荐）
.venv\Scripts\python -m qa_agent --check

# 进入多轮对话
.venv\Scripts\python -m qa_agent
```

真实示例（一次提问同时触发两个工具）：

```
你> 帮我算一下 (128+72)*3^2，然后搜一下智谱GLM最新的模型是什么
助手> ## 计算结果
     (128+72)×3² = 200×9 = 1800
     ## 智谱 GLM 最新模型
     根据搜索结果，智谱目前最新的旗舰模型是 GLM-5.2，……
     信息来源：
     - [智谱AI官方：GLM-5.2上线并开源](https://www.zhipuai.cn/zh/research/161)
     - [智谱AI开放文档 - 新品发布](https://docs.bigmodel.cn/cn/update/new-releases)
```

### 4. 运行测试（无需 API Key）

```bash
.venv\Scripts\python -m pytest -q      # 38 passed
```

## 直接调用 Tavily（不走智能体）

项目依赖已含官方 SDK（`pip install -e ".[dev,tavily]"` 安装 extras）：

```python
import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()  # 读 .env 里的 TAVILY_API_KEY
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

r = client.search(
    "智谱GLM 最新模型",
    max_results=3,         # 0-20
    search_depth="basic",  # basic/fast=1 credit，advanced=2 credits（更准）
    include_answer=True,   # 附带 AI 综合回答
)
print(r["answer"])         # AI 综合回答
for x in r["results"]:     # 结构化结果
    print(x["title"], x["url"], x["score"], x["content"])
```

等价的 REST 调用（项目内部即此方式，见 `src/qa_agent/tools/internet_search.py`）：

```bash
curl -X POST https://api.tavily.com/search \
  -H "Authorization: Bearer 你的tvly-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "搜索词", "max_results": 5}'
```

官方文档：<https://docs.tavily.com/>

## 常见问题（真实案例排查）

| 现象 | 原因 | 解决 |
|------|------|------|
| pip 报 `from versions: none` / 超时 | 国内直连 PyPI 被墙 | 加镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`（uv 用 `UV_DEFAULT_INDEX`） |
| `--check` 回复是空字符串 | 模型是推理模型，思考耗尽了小 token 上限 | 已修复（内部用 512）；如自定义请保持 max_tokens ≥ 512 |
| 回答里混着大段"思考过程" | 推理模型流式思考事件 | 已修复（只打印正文增量） |
| 搜索超时 / 报 google.com 超时 | 未配 Tavily，回退的 DuckDuckGo 被墙 | `.env` 填 `TAVILY_API_KEY`（推荐）；或挂代理设 `HTTPS_PROXY` |
| 模型说"我没有搜索工具" | 工具名与供应商内置工具重名被拦截（智谱 `web_search` 是内置类型） | 已修复（改名 `internet_search`）；自定义新工具时避开内置名 |
| 接口连通失败 | base_url/模型名不匹配 | `--check` 看具体报错；模型名必须写该平台自己的（如 DeepSeek 用 `deepseek-chat`） |

## 项目结构

```
qa-agent/
├── src/qa_agent/
│   ├── __main__.py            # CLI：多轮对话、流式输出、--check 自检
│   ├── config.py              # .env 配置加载与校验
│   ├── agent.py               # Agent 定义（系统指令 + 工具装配）
│   └── tools/
│       ├── calculator.py      # 计算器工具（AST 安全求值）
│       └── internet_search.py # 联网搜索（Tavily 优先 + DuckDuckGo 兜底）
├── tests/                     # 38 个单元测试（无需 API Key）
├── docs/技术文档.md           # 架构、时序、设计决策与踩坑、测试设计
├── smoke_search.py            # 搜索通路冒烟脚本
├── .env.example               # 配置模板
└── pyproject.toml             # 依赖声明（extras: dev / tavily）
```

## 如何新增一个工具

以"当前时间"工具为例，三步：

1. 新建 `src/qa_agent/tools/clock.py`：
   ```python
   from datetime import datetime
   from agents import function_tool

   @function_tool
   def now() -> str:
       """获取当前本地日期时间。当用户询问今天几号、现在几点时调用。"""
       return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")
   ```
   docstring 和类型注解会自动生成给模型看的 JSON Schema。
2. `agent.py` 里导入并加入 `tools=[calculator, internet_search, now]`。
3. `tests/` 里加对应单测（参考 `test_internet_search.py` 的 mock 写法）。

更多设计细节（架构图、调用时序、8 项设计决策与踩坑记录）见
[docs/技术文档.md](docs/技术文档.md)。
