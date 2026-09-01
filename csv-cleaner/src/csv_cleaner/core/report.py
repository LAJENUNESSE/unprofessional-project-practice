"""Markdown 分析报告导出：体检结果 + 清洗日志 + 统计摘要 + 数据预览。"""
from __future__ import annotations

import datetime as _dt
import pathlib

import pandas as pd

from .cleaner import Cleaner
from .inspector import InspectionReport
from . import stats


def build_report(df: pd.DataFrame, inspection: InspectionReport, cleaner: Cleaner,
                 source_name: str = "") -> str:
    """把清洗后的数据、体检报告、操作日志组装成 Markdown 报告。"""
    lines: list[str] = []
    lines.append("# CSV 数据清洗分析报告")
    lines.append("")
    lines.append(f"- 生成时间：{_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    if source_name:
        lines.append(f"- 数据来源：{source_name}")
    lines.append(f"- 清洗后规模：{len(df)} 行 × {len(df.columns)} 列")
    lines.append("")

    lines.append("## 一、清洗前数据体检")
    lines.append("")
    lines.append("```")
    lines.extend(inspection.to_lines())
    lines.append("```")
    lines.append("")

    lines.append("## 二、清洗操作记录")
    lines.append("")
    if cleaner.log:
        lines.append("| 步骤 | 操作 | 说明 | 影响行数 |")
        lines.append("| --- | --- | --- | --- |")
        for a in cleaner.log:
            lines.append(f"| {a.step} | {a.action} | {a.detail} | {a.rows_affected} |")
    else:
        lines.append("（未执行任何清洗操作）")
    lines.append("")

    lines.append("## 三、清洗后统计摘要")
    lines.append("")
    desc = stats.numeric_summary(df)
    if not desc.empty:
        lines.append("### 数值列统计")
        lines.append("")
        lines.append(desc.to_markdown())
        lines.append("")
    for col in df.columns:
        vc = stats.category_top(df, str(col), n=5)
        # 数值列的频次意义不大，只展示非数值列
        if not pd.api.types.is_numeric_dtype(df[col]) and not vc.empty:
            lines.append(f"### 列「{col}」频次 TOP5")
            lines.append("")
            lines.append(vc.to_markdown())
            lines.append("")

    lines.append("## 四、清洗后数据预览（前 20 行）")
    lines.append("")
    lines.append(df.head(20).to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


def export_report(df: pd.DataFrame, inspection: InspectionReport, cleaner: Cleaner,
                  out_path: str | pathlib.Path, source_name: str = "") -> pathlib.Path:
    """生成并写出 Markdown 报告，返回文件路径。"""
    text = build_report(df, inspection, cleaner, source_name)
    path = pathlib.Path(out_path)
    path.write_text(text, encoding="utf-8")
    return path
