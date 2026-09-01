# Day01 任务清单与完成方案

> 依据：`01-AIGC应用与实践-PPT.pdf`（86页）、`AI实践项目集19个.pdf`（64页）、`环境与编译器安装.pdf`（7页），全文逐页读图整理。
> 三个 PDF 均为扫描/字体缺映射，无法复制文本，已渲染 PNG 存于 `_extract/`（proj 64 张、ppt 86 张、env 7 张）。

## 一、三份材料是什么

| 材料 | 内容 | 是否有硬性作业 |
|---|---|---|
| 01-AIGC应用与实践-PPT | 第1天通识课，8章：①AIGC概念与发展史 ②行业应用 ③提示词方法 ④文本大模型应用 ⑤AI绘画(MJ/SD) ⑥图像大模型应用(即梦/Dreamina、室内设计工作流) ⑦音视频与智能体(Suno/Sora/数字人/coze) ⑧资源分享(魔搭/HuggingFace/锡布哩布/Civitai/FaceChain) | 无明确课后作业页，属听课材料 |
| AI实践项目集19个 | 19个上手项目，每个含"练习作业"提示与操作步骤截图 | **是，本Day核心作业** |
| 环境与编译器安装 | Python 安装 + Jupyter Lab 安装/快捷键/导出 | 环境验证即完成 |

## 二、环境要求（env PDF）与当前状态

1. Python 安装并勾选 Add to PATH，`cmd` 输 `python` 验证 —— ✅ 已有 Python 3.14（scoop），远高于文档示例的 3.12
2. Jupyter Lab：`pip install jupyter lab -i https://pypi.mirrors.ustc.edu.cn/simple/`，`WIN+R → jupyter lab` 启动 —— ⬜ 需验证/安装（注意：文档命令 `pip install jupyter lab` 少了中划线，正确包名为 `jupyterlab`，`pip install jupyterlab` 或 `pip install jupyter lab` 均可解析）
3. 掌握快捷键（H 帮助、Ctrl-Enter 运行、M/Y 切换 md/code、A/B 插入单元格、D,D 删除）—— 听课内容
4. .ipynb 可导出 pdf/py/html/md —— 听课内容

## 三、19 个实践项目清单与可行方案

| # | 项目 | 用到的工具 | 可行性 / 替代方案 |
|---|---|---|---|
| 1 | ChatGPT批量制作小红书爆款笔记 | ChatGPT | ✅ 智谱清言/GLM API 直接做 |
| 2 | ChatGPT批量生产热门爆款原创文章 | ChatGPT | ✅ 同上 |
| 3 | 用GPT当提示词专家优化提示词 | ChatGPT | ✅ 同上 |
| 4 | 制作一张漂浮穿着人类衣服的宠物照片 | DALL·E | 🔶 需绘图模型：即梦/通义万相，或 GLM 的 imagegen |
| 5 | 大语言模型生成编程程序 | ChatGPT | ✅ GLM 生成，本地跑验证（Tkinter 抽奖程序三轮迭代） |
| 6 | 大语言模型担任作家 | ChatGPT | ✅ |
| 7 | 国内大语言模型和ChatGPT问答对比 | 多家模型 | ✅ 智谱/Kimi/文心等网页各问一遍 |
| 8 | 论文分析 | ChatGPT | ✅ GLM 长文本/网页版传文件 |
| 9 | ChatGPT创作儿童故事 | ChatGPT | ✅ |
| 10 | ChatGPT创作剧本 | ChatGPT | ✅ |
| 11 | GPT生成SD和MJ魔法词 | ChatGPT | ✅ 只需生成提示词文本 |
| 12 | 剪映生成抖音短视频 | 剪映(图文成片) | 🔶 需装剪映桌面版，人工操作GUI |
| 13 | 室内设计方案-线稿出图 | Stable Diffusion | 🔶 需 SD 本地部署或哩布在线出图 |
| 14 | 室内设计方案-毛坯房直出 | Stable Diffusion | 🔶 同上 |
| 15 | 制作个人写真 | FaceChain/妙鸭 | 🔶 魔搭 FaceChain 可在线跑，需注册 |
| 16 | AI音乐生成实践-水调歌头词牌生成音乐 | Suno | 🔶 Suno 需注册(可谷歌账号)；替代：网易天音/天工SkyMusic |
| 17 | AI生成视频 | Sora/即梦/noise.ai | 🔶 即梦国内可直接用 |
| 18 | 论文写作 | ChatGPT | ✅ |
| 19 | 法律顾问 | ChatGPT | ✅ |

图例：✅ 现有条件可直接做（智谱清言网页 / GLM API）；🔶 需要装软件或注册账号后人工操作。

## 四、建议完成路径

1. **环境验证**（半小时内）：装/验 Jupyter Lab → 起 `jupyter lab` 截图存档。
2. **文本类 12 项**（#1,2,3,5,6,7,8,9,10,11,18,19）：全部用智谱清言网页或 GLM API 脚本完成，提示词与产出整理到 `Day01-Outputs/`，每项一个 md/截图；#5 额外把抽奖程序代码在本地跑通（`.venv` + Tkinter，三轮迭代过程记录）。
3. **多模态 5 项**（#4,13,14,15,17）：首选即梦（字节，免注册可用网页版）+ GLM 绘图；若课程验收要求截图带工具界面，需要你注册对应账号操作，我可准备好每项的提示词与操作指引。
4. **音视频 2 项**（#12,16）：剪映桌面版需人工 GUI 操作；音乐优先试 Suno（需账号）或网易天音。同样由我备好文案（歌词/脚本），你粘贴生成即可。
5. **产出归档**：`Day01-Outputs/` 下按 `01-小红书笔记.md` … `19-法律顾问.md` 编号存放 + 截图，最后合并一份实验说明（对齐 Day04 的格式）。

## 五、备注

- PPT 第7章提到的"使用智谱清言和剪映生成抖音短视频"、coze 智能体，与本项目集 #12 相互印证——课程方默认国内可访问智谱全家桶，与我们现有 GLM API 栈一致。
- 课程资料来源水印为"365天深度学习训练营@K同学啊"。
