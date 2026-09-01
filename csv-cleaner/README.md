# CSV 数据清洗报表工具

一个用 Python + Tkinter 开发的桌面数据工具：导入 CSV → 数据体检 → 一键清洗 → 统计图表 → 导出干净数据与分析报告。提供打包好的 exe，双击即用。

> 本项目为「AI 编程挑战」训练任务成果，全程 AI 辅助开发，见 [docs/AI使用说明.md](docs/AI使用说明.md) 与 [docs/项目报告.md](docs/项目报告.md)。详细操作手册见 [docs/使用说明.md](docs/使用说明.md)，课堂演示脚本见 [docs/演示文档.md](docs/演示文档.md)。

## 功能一览

| 模块 | 功能 |
| --- | --- |
| 导入 | CSV 编码自动识别（UTF-8 / GBK），兼容混合分隔符与坏行 |
| 数据体检 | 行列规模、重复行、每列类型推断、缺失统计、文本列可转换性、数值列 IQR 异常值 |
| 行级清洗 | 去完全重复行、删缺失行、文本去首尾空格 |
| 缺失值 | 均值 / 中位数 / 众数 / 固定值填充 |
| 类型转换 | 货币文本列（`¥1,299.00`）转数值；混合格式日期转日期 |
| 异常值 | IQR 法，支持缩进边界或删除行 |
| 列管理 | 重命名、删除列 |
| 统计图表 | 直方图、箱线图、类别频次 TOP10、相关性热力图 |
| 导出 | 清洗后 CSV（带 BOM，Excel 直接打开不乱码）、Markdown 分析报告（体检 + 清洗日志 + 统计摘要） |

每个清洗步骤都记入「清洗日志」，导出的报告自动包含完整操作记录，保证清洗过程可追溯。

## 快速开始

### 方式一：直接运行 exe（免安装）

双击 `dist/CsvCleaner.exe`（首次启动需数秒解压，属正常现象）。

打开本仓库自带的 `samples/messy_sales.csv` 体验完整流程：

1. 点 **① 数据体检 → 重新体检**，查看问题清单
2. 依次点 **去除完全重复行**、**文本列去首尾空格**
3. ③ 选 `单价` 列 → ④ 类型转换选 `数值` → **转换所选列**（把 `¥1,299.00` 这类文本转成数字）
4. ③ 选 `客户年龄` → 策略 `median(中位数)` → **填充所选列**
5. ⑤ 选 `数量` → **处理所选列**（IQR 缩进异常值）
6. 切到 **图表** 页签选列绘制；最后菜单 **文件 → 导出分析报告**

### 方式二：从源码运行

要求 Python ≥ 3.12。虚拟环境已含在仓库 `.venv/` 中（复用无需重建）：

```bash
# 已有 .venv（Windows）
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m csv_cleaner

# 或新建环境
python -m venv .venv
.venv\Scripts\pip install -e .[dev]
```

### 运行测试

```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

### 重新打包 exe

```bash
.venv\Scripts\python.exe -m PyInstaller --onefile --windowed --noconfirm \
    --name CsvCleaner --paths src launcher.py
```

## 项目结构

```
csv-cleaner/
├── launcher.py              # 打包入口
├── pyproject.toml           # 项目元数据与依赖
├── src/csv_cleaner/
│   ├── app.py               # Tkinter 主窗口与事件
│   ├── core/
│   │   ├── loader.py        # 编码识别与读取
│   │   ├── inspector.py     # 数据体检
│   │   ├── cleaner.py       # 清洗操作 + 操作日志
│   │   ├── stats.py         # 统计摘要
│   │   └── report.py        # Markdown 报告导出
│   └── gui/
│       ├── table_view.py    # 表格视图
│       └── charts_panel.py  # matplotlib 图表面板
├── tests/                   # pytest 单元测试（19 个）
├── samples/                 # 示例脏数据（UTF-8 与 GBK 各一份）
└── docs/                    # 使用说明、演示文档、项目报告、AI 使用说明、截图
```

## 已知说明

- exe 体积约 15~80 MB：PyInstaller 单文件模式打包了 pandas + matplotlib 运行时，属正常现象
- 个别杀毒软件可能对未签名的 PyInstaller exe 误报，可加入信任或改从源码运行
- 大于 200 行的数据在预览页只显示前 200 行（防界面卡顿），清洗与导出始终针对全量数据
