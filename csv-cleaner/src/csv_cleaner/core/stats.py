"""统计摘要：数值描述、类别频次、相关性。"""
from __future__ import annotations

import pandas as pd


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """数值列 describe 转置（列为统计指标）。"""
    numeric = df.select_dtypes("number")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T.round(3)


def category_top(df: pd.DataFrame, column: str, n: int = 10) -> pd.Series:
    """类别列频次 TOP N。"""
    return df[column].value_counts().head(n)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """数值列相关系数矩阵。"""
    return df.select_dtypes("number").corr().round(3)
