"""清洗操作：每个动作都记录到操作日志，供报告导出与界面回显。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

_CURRENCY_RE = re.compile(r"[¥$€£,\s]")


@dataclass
class CleaningAction:
    step: int
    action: str
    detail: str
    rows_affected: int


class Cleaner:
    """包装 DataFrame 并执行清洗动作，所有动作写入 log。"""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.log: list[CleaningAction] = []
        self._step = 0

    def _record(self, action: str, detail: str, rows_affected: int) -> None:
        self._step += 1
        self.log.append(CleaningAction(self._step, action, detail, rows_affected))

    # ---------- 行级操作 ----------

    def drop_duplicates(self) -> int:
        """删除完全重复行，返回删除行数。"""
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        removed = before - len(self.df)
        self._record("去除重复行", f"删除 {removed} 行", removed)
        return removed

    def drop_missing_rows(self, subset: list[str] | None = None) -> int:
        """删除含缺失值的行（可指定列），返回删除行数。"""
        before = len(self.df)
        self.df = self.df.dropna(subset=subset).reset_index(drop=True)
        removed = before - len(self.df)
        cols = "、".join(subset) if subset else "任意列"
        self._record("删除缺失行", f"按 {cols} 删除含缺失行 {removed} 行", removed)
        return removed

    def fill_missing(
        self,
        column: str,
        strategy: str,
        fill_value: str | float | None = None,
    ) -> int:
        """填充缺失值。strategy: mean / median / mode / constant。

        返回填充的单元格数；列无缺失时为合法空操作。
        """
        s = self.df[column]
        n_missing = int(s.isna().sum())
        if n_missing == 0:
            self._record("填充缺失值", f"列「{column}」无缺失，跳过", 0)
            return 0

        if strategy == "mean":
            value = float(pd.to_numeric(s, errors="coerce").mean())
            self.df[column] = s.fillna(value)
            detail = f"列「{column}」以均值 {value:.2f} 填充"
        elif strategy == "median":
            value = float(pd.to_numeric(s, errors="coerce").median())
            self.df[column] = s.fillna(value)
            detail = f"列「{column}」以中位数 {value:.2f} 填充"
        elif strategy == "mode":
            mode = s.mode(dropna=True)
            value = mode.iloc[0] if not mode.empty else None
            self.df[column] = s.fillna(value)
            detail = f"列「{column}」以众数「{value}」填充"
        elif strategy == "constant":
            if fill_value is None:
                raise ValueError("constant 策略需要 fill_value")
            self.df[column] = s.fillna(fill_value)
            value = fill_value
            detail = f"列「{column}」以固定值「{fill_value}」填充"
        else:
            raise ValueError(f"未知填充策略：{strategy}")

        self._record("填充缺失值", detail, n_missing)
        return n_missing

    # ---------- 列级操作 ----------

    def convert_numeric(self, column: str) -> int:
        """文本列去货币符号/千分位后转数值，返回成功转换的单元格数。"""
        original = self.df[column]
        cleaned = original.astype("string").str.replace(_CURRENCY_RE, "", regex=True)
        converted = pd.to_numeric(cleaned, errors="coerce")
        # 原本非空、转换后仍为空的视为脏数据，保留为缺失
        n = int((original.notna() & converted.notna()).sum())
        self.df[column] = converted
        self._record("转数值列", f"列「{column}」转换为数值，成功 {n} 个", n)
        return n

    def convert_datetime(self, column: str) -> int:
        """解析混合格式日期列，返回成功解析的单元格数。"""
        parsed = pd.to_datetime(self.df[column], errors="coerce", format="mixed")
        n = int(parsed.notna().sum())
        self.df[column] = parsed
        self._record("转日期列", f"列「{column}」转换为日期，成功 {n} 个", n)
        return n

    def rename_column(self, old: str, new: str) -> None:
        if old not in self.df.columns:
            raise KeyError(f"列不存在：{old}")
        self.df = self.df.rename(columns={old: new})
        self._record("重命名列", f"「{old}」→「{new}」", len(self.df))

    def drop_columns(self, columns: list[str]) -> int:
        exist = [c for c in columns if c in self.df.columns]
        self.df = self.df.drop(columns=exist)
        self._record("删除列", "删除列：" + "、".join(exist), len(self.df))
        return len(exist)

    # ---------- 异常值 ----------

    def handle_outliers(self, column: str, action: str = "clip") -> int:
        """IQR 法处理数值列异常值。action: clip（缩进边界）/ remove（删除行）。"""
        s = pd.to_numeric(self.df[column], errors="coerce")
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < low) | (s > high)
        n_out = int(mask.sum())
        if n_out == 0:
            self._record("处理异常值", f"列「{column}」未发现异常值，跳过", 0)
            return 0

        if action == "clip":
            self.df[column] = s.clip(lower=low, upper=high)
            detail = f"列「{column}」缩进 {n_out} 个异常值到 [{low:.2f}, {high:.2f}]"
        elif action == "remove":
            self.df = self.df[~mask.fillna(False)].reset_index(drop=True)
            detail = f"列「{column}」删除 {n_out} 行异常值"
        else:
            raise ValueError(f"未知异常值处理方式：{action}")
        self._record("处理异常值", detail, n_out)
        return n_out

    # ---------- 文本规范化 ----------

    def strip_whitespace(self) -> int:
        """全部文本列去除首尾空格，返回受影响单元格数。"""
        n = 0
        for col in self.df.columns:
            s = self.df[col]
            if pd.api.types.is_string_dtype(s) or s.dtype == object:
                stripped = s.astype("string").str.strip()
                changed = int((s.notna() & (stripped != s.astype("string"))).sum())
                if changed:
                    self.df[col] = stripped
                    n += changed
        self._record("去首尾空格", f"清理 {n} 个单元格", n)
        return n
