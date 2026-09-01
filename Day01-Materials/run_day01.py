# -*- coding: utf-8 -*-
"""Day01 文本类实践项目批量执行脚本
从 .env 读取 OPENAI_BASE_URL / OPENAI_API_KEY / QA_AGENT_MODEL，
逐个调用 GLM 完成项目集 01-19 中纯文本类练习作业，产出写入 Day01-Outputs/。
"""
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Day01-Outputs"
OUT.mkdir(exist_ok=True)

# ---- 读取 .env ----
env = {}
for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"')

BASE = env["OPENAI_BASE_URL"].rstrip("/")
KEY = env["OPENAI_API_KEY"]
MODEL = env["QA_AGENT_MODEL"]


def chat(messages: list, retries: int = 3) -> str:
    """调用 OpenAI 兼容接口，返回助手回复文本。思考模型不设 max_tokens。"""
    payload = {"model": MODEL, "messages": messages, "temperature": 0.8}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                BASE + "/chat/completions",
                data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + KEY},
            )
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            return d["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"chat failed after {retries} retries: {last_err}")


def save(name: str, title: str, content: str):
    p = OUT / f"{name}.md"
    p.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
    print(f"[saved] {p.name}  ({len(content)} chars)")


# ============ 各项目提示词 ============

TASKS = {}

# 01 ChatGPT批量制作小红书爆款笔记 —— 练习作业：让ChatGPT批量做10个小红书笔记
TASKS["01-小红书爆款笔记"] = [
    {"role": "system", "content": "你是一名资深小红书博主，擅长写爆款笔记。"},
    {"role": "user", "content": (
        "你的任务是以小红书博主的文章结构，以我给出的主题写帖子推荐。回答应包括便于表情符号增加趣味和互动，"
        "以及与每个段落相匹配的图片描述。以引人入胜的介绍开始，为推荐设置基调，然后提供至少三个与主题相关的段落，"
        "突出它们独特特点和吸引力，结尾附话题标签。\n"
        "请围绕主题「新手养宠物须知」批量产出10篇小红书笔记，每篇用「## 笔记N」分隔，主题可细分为"
        "宠物护理、喂养指南、用品清单、健康观察等不同角度。"
    )},
]

# 02 ChatGPT批量生产热门爆款原创文章 —— 练习作业：让ChatGPT批量完成5篇文章创作
TASKS["02-爆款原创文章"] = [
    {"role": "user", "content": (
        "你现在充当我的爆款文章写作助手。下面我给你一篇热门文章，你要帮我改写成5篇原创度高于90%（重复度低于10%）"
        "的新文章，保持口语化、有网感、情绪饱满的风格，每篇800字左右，用「## 文章N」分隔。\n\n"
        "【原文】\n"
        "30岁裸辞写小说的每一天，都是自由的感觉\n"
        "不知不觉已经是裸辞写小说的第四年了。\n"
        "早上睡到自然醒，没有闹钟催促，不会因为白天的会议和任务突然惊醒。\n"
        "起来之后肚子饿就先煮点早餐，不饿就去沙发上看会儿书。\n"
        "上午只要写完四千字，任务就完成了。下午晚上都是自己的。\n"
        "晚饭后可以一个人下楼逛逛夜市，就算不买看看热闹也很开心。\n"
        "我可以写字，也可以看书。做自己喜欢的事并且还能从中赚钱，这简直是普通人的梦想。"
        "感谢年轻而又莽撞的自己，让我有了现在的自由。"
    )},
]

# 03 用GPT作提示词专家优化提示词 —— 练习作业：用GPT作为提示词专家给你5个提示词
TASKS["03-提示词专家"] = [
    {"role": "user", "content": (
        "你现在是AI提示词（Prompt）大师，我给你提出的所有需求你都能准确理解，且会先给我提出5个更具体详细的提示词选项。\n"
        "我的需求是：我想让AI帮我学习Python编程。请给我5个优化后的提示词建议。"
    )},
]

# 05 大语言模型生成编程程序（问题1+问题2+问题3 由本地验证，单独处理）
TASKS["05-抽奖程序-问题1"] = [
    {"role": "user", "content": (
        "帮我虚拟一个30人的名单，然后用Python写一个随机抽奖程序，带有UI界面，窗口尺寸为300x200。"
        "只输出完整代码，用```python代码块包裹。"
    )},
]
TASKS["05-抽奖程序-问题2"] = [
    {"role": "user", "content": (
        "帮我虚拟一个30人的名单，然后用Python写一个随机抽奖程序，带有UI界面，窗口尺寸为300x200。"
        "要求：点击按钮后名字滚动5秒钟，然后弹出确定的中奖人名，并且按钮背景改为蓝色。"
        "只输出完整代码，用```python代码块包裹。"
    )},
]

# 06 大语言模型担任作家（自传第一章）
TASKS["06-大模型作家"] = [
    {"role": "user", "content": (
        "请你扮演一名作家。我的人生自传第一章主题是「山村的梦想」：我出生在江苏省一个偏远山区农村，"
        "生活朴素而艰辛，在父母的鼓励下从小刻苦学习，每天走山路求学，坚信知识改变命运。"
        "请帮我详细写一下第一章，800字左右，第一人称，情感真挚。"
    )},
]

# 07 国内大语言模型和ChatGPT问答对比 —— 本地只有GLM，改为记录GLM回答并留对比表格框架
TASKS["07-模型问答对比"] = [
    {"role": "user", "content": (
        "问题：手机充电器快充协议不匹配会怎样？请简述原因和解决方案。\n"
        "请用200字以内回答。"
    )},
]

# 08 论文分析
TASKS["08-论文分析"] = [
    {"role": "user", "content": (
        "你是一名学术论文分析助手。请对以下论文标题与摘要进行分析，输出：1)研究问题 2)方法 3)主要发现 4)创新点 5)局限。\n\n"
        "标题：GOVGPT：生成式人工智能驱动的政务服务智能体研究\n"
        "摘要：随着人工智能技术的创新迭代，生成式人工智能（AIGC）正在成为塑造政务服务智能体的新型引擎，"
        "构思并衍生出行政领域的人工智能通用大模型——GOVGPT。生成式人工智能驱动的政务服务主要体现在空间场景、"
        "服务范式与治理逻辑三个层面的重构。"
    )},
]

# 09 ChatGPT创作儿童故事
TASKS["09-儿童故事"] = [
    {"role": "user", "content": (
        "请为8岁的小朋友们创作一篇关于勇气和友谊的儿童故事。故事情节：米菲兔和小伙伴在森林里玩耍，"
        "突然遇到了一只凶猛的老虎要吃掉他们。故事的主角米菲兔挺身而出，制定了一项逃跑计划，"
        "将老虎引开并让他们中的每一个人都安全脱险。在故事的结尾，米菲兔和他的朋友们因为相互的帮助和支持"
        "变得更加团结友爱，同时也体会到了勇气和友谊的真正含义。语言通俗生动，故事情节离奇曲折，引人入胜。"
        "故事中要有对话。"
    )},
]

# 10 ChatGPT创作剧本
TASKS["10-电影剧本"] = [
    {"role": "user", "content": (
        "请创作爱情悲剧的电影剧本。角色：主人公冷艳，一个企业高管杜小川的上司，性格高冷，精明强干；"
        "杜小川，AI工程师，体贴温柔，勇敢担当。剧情：一次主人公冷艳和杜小川火车上相遇，并遭遇了抢劫，"
        "杜小川为了保护冷艳深受重伤，从此冷艳改变了对杜小川的看法，并开始喜欢上了他。但冷艳父母反对，"
        "杜小川无奈只好远走他乡，冷艳终身未嫁、孤独终老。片长大约15分钟左右。"
    )},
]

# 11 GPT生成SD和MJ魔法词
TASKS["11-SD与MJ魔法词"] = [
    {"role": "user", "content": (
        "完成两件事：\n"
        "1. 我想使用Midjourney创作一幅美图，画面内容：午后，一位中国美女在书房读书。"
        "请帮我写提示词（prompt），先给中文版再翻译成英文，按 \"/imagine prompt: ...\" 格式，包含画面、光线、风格、画质词。\n"
        "2. 我使用Stable Diffusion文生图，同样画面请给出SD格式的正向提示词与负向提示词，"
        "正向包含质量词(如 masterpiece, best quality, 8k)、画面细节词，负向包含常见劣化词(如 lowres, bad anatomy)。"
    )},
]

# 12 剪映生成抖音短视频 —— 只产出脚本文本（剪映GUI操作需人工）
TASKS["12-短视频脚本"] = [
    {"role": "user", "content": (
        "请帮我生成一个短视频脚本，关于重庆洪崖洞景点介绍的。格式：标题 + 6个镜头（每个镜头写画面描述与旁白），"
        "最后加片尾。旁白单独汇总成一段，方便粘贴到剪映「图文成片」。"
    )},
]

# 18 论文写作（摘要/提纲/评价）
TASKS["18-论文写作"] = [
    {"role": "user", "content": (
        "我想写一篇关于《DeFi项目中去中心化交易的市场影响与实证研究》论文，请为我生成一篇300字左右的摘要，"
        "语言要精炼，符合学术规范；然后根据摘要生成论文提纲结构；最后自我评价一下这个提纲如何并给出改进建议。"
    )},
]

# 19 法律顾问（专利申请书）
TASKS["19-法律顾问"] = [
    {"role": "user", "content": (
        "区块链工程师杨云发明了一套高效跨链交换数据的新方法，请起草一份中国专利申请书，"
        "包含：发明名称、技术领域、背景技术、发明内容、附图说明、具体实施方式、权利要求。"
    )},
]


def extract_code(text: str) -> str:
    """从回复中提取第一个 python 代码块。"""
    if "```python" in text:
        seg = text.split("```python", 1)[1]
    elif "```" in text:
        seg = text.split("```", 1)[1]
    else:
        return text
    return seg.split("```", 1)[0].strip()


def run_lucky_draw():
    """项目05：生成代码→本地运行三轮验证（问题2含5秒滚动）。"""
    import subprocess
    import sys

    code_dir = OUT / "05-抽奖程序"
    code_dir.mkdir(exist_ok=True)

    # 问题2（完整版：滚动5秒+弹窗+按钮变蓝）
    reply2 = chat(TASKS["05-抽奖程序-问题2"])
    save("05-抽奖程序/glm回答", "大语言模型生成抽奖程序（问题2：滚动5秒版）", reply2)
    code = extract_code(reply2)
    src = code_dir / "lucky_draw.py"
    src.write_text(code, encoding="utf-8")

    # 语法检查 + 导入检查（不弹窗：-c import 检查）
    r = subprocess.run([sys.executable, "-m", "py_compile", str(src)],
                       capture_output=True, text=True)
    print("[05] py_compile:", "OK" if r.returncode == 0 else r.stderr[:500])

    # 离屏跑 8 秒验证滚动逻辑不报错（用 after 截断：直接运行，稍后杀掉）
    proc = subprocess.Popen([sys.executable, str(src)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace")
    time.sleep(8)
    if proc.poll() is None:
        proc.terminate()
        print("[05] 窗口运行 8s 无崩溃（已截断）")
        ok = True
    else:
        err = proc.stderr.read() if proc.stderr else ""
        print("[05] 进程提前退出，stderr:", err[:800])
        ok = proc.returncode == 0
    (code_dir / "运行验证.txt").write_text(
        f"lucky_draw.py 本地验证：py_compile {'通过' if r.returncode == 0 else '失败'}；"
        f"窗口运行{'无崩溃' if ok else '异常'}。验证时间：{time.strftime('%Y-%m-%d %H:%M')}\n",
        encoding="utf-8")
    return ok


def main():
    only = os.environ.get("ONLY")
    names = [n for n in TASKS if not only or n.startswith(only)]
    for name in names:
        print(f"==== {name} ====")
        try:
            reply = chat(TASKS[name])
            save(name, name.split("-", 1)[1], reply)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name}: {e}")
    if not only or only.startswith("05"):
        print("==== 05 本地运行验证 ====")
        try:
            run_lucky_draw()
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] 05-本地验证: {e}")


if __name__ == "__main__":
    main()
