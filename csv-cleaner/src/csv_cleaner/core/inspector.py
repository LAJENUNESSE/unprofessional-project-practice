"""数据体检：扫描 DataFrame 的质量问题，输出结构化报告。"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .loader import guess_column_kind


@dataclass
class ColumnReport:
    name: str
    kind: str  # numeric / datetime / text
    dtype: str
    missing: int
    missing_ratio: float
    n_unique: int
    outliers: int = 0  # 仅数值列：IQR 法异常值个数
    convertible: bool = False  # 文本列：去货币符号后大多能转数值


@dataclass
class InspectionReport:
    n_rows: int
    n_cols: int
    duplicate_rows: int
    columns: list[ColumnReport] = field(default_factory=list)

    def to_lines(self) -> list[str]:
        """生成面向用户的问题清单（GUI 体检页直接展示）。"""
        lines = [
            f"数据规模：{self.n_rows} 行 × {self.n_cols} 列，重复行 {self.duplicate_rows} 行",
            "",
            f"{'列名':<12}{'类型':<10}{'缺失':<8}{'缺失率':<9}{'唯一值':<8}{'异常值':<8}备注",
        ]
        for c in self.columns:
            notes = []
            if c.convertible:
                notes.append("文本列含数值，可转换")
            if c.missing:
                notes.append("有缺失")
            lines.append(
                f"{c.name:<12}{c.kind:<10}{c.missing:<8}{c.missing_ratio:>6.1%}  "
                f"{c.n_unique:<8}{c.outliers:<8}{'；'.join(notes)}"
            )
        return lines


def _count_outliers(values: pd.Series) -> int:
    """IQR 法：低于 Q1-1.5IQR 或高于 Q3+1.5IQR 视为异常。"""
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((values < low) | (values > high)).sum())


def inspect(df: pd.DataFrame) -> InspectionReport:
    """对当前数据做全面体检。"""
    report = InspectionReport(
        n_rows=len(df),
        n_cols=len(df.columns),
        duplicate_rows=int(df.duplicated().sum()),
    )
    for name in df.columns:
        s = df[name]
        kind = guess_column_kind(s)
        convertible = False
        outliers = 0
        if kind == "numeric":
            outliers = _count_outliers(s.dropna())
        else:
            convertible = _mostly_numeric(s)
        report.columns.append(
            ColumnReport(
                name=str(name),
                kind=kind,
                dtype=str(s.dtype),
                missing=int(s.isna().sum()),
                missing_ratio=float(s.isna().mean()) if len(s) else 0.0,
                n_unique=int(s.nunique()),
                outliers=outliers,
                convertible=convertible,
            )
        )
    return report


def _mostly_numeric(s: pd.Series) -> bool:
    """判断文本列去掉货币符号/千分位后是否能大部分转成数值。"""
    sample = s.dropna().astype(str).head(200)
    if not len(sample):
        return False
    cleaned = sample.str.replace(r"[¥$€£,\s]", "", regex=True)
    converted = pd.to_numeric(cleaned, errors="coerce")
    return bool(converted.notna().mean() > 0.6)
