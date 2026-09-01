"""CSV 文件读取与编码识别。"""
from __future__ import annotations

import csv
import pathlib

import chardet
import pandas as pd


def read_csv_any(path: str | pathlib.Path) -> pd.DataFrame:
    """读取 CSV，自动识别编码（utf-8 / gbk 等）并兼容常见脏格式。

    识别顺序：chardet 探测编码 -> 尝试 python engine + sep=None 嗅探分隔符；
    若个别行字段数不一致则跳过坏行，保证尽量能打开。
    """
    path = pathlib.Path(path)
    raw = path.read_bytes()

    # 小文件直接探测全部字节；空文件返回空表
    if not raw.strip():
        return pd.DataFrame()

    guess = chardet.detect(raw[:200_000])["encoding"] or "utf-8"
    # chardet 对 GBK 常报 GB2312，统一放宽到 gbk 以覆盖更多字符
    encoding = "gbk" if guess.lower().replace("-", "") in {"gb2312", "gbk", "gb18030"} else guess

    try:
        df = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
    except (UnicodeDecodeError, pd.errors.ParserError):
        # 兜底：跳过坏行、替换无法解码的字节
        df = pd.read_csv(
            path, encoding=encoding, sep=None, engine="python",
            on_bad_lines="skip", encoding_errors="replace",
        )
    return df


def sniff_encoding(path: str | pathlib.Path) -> str:
    """仅探测文件编码，供界面展示。"""
    raw = pathlib.Path(path).read_bytes()[:200_000]
    guess = chardet.detect(raw)["encoding"] or "unknown"
    return "gbk" if guess.lower().replace("-", "") in {"gb2312", "gbk", "gb18030"} else guess


def guess_column_kind(series: pd.Series) -> str:
    """把列粗分为 numeric / datetime / text 三类，供体检和图表选择使用。"""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    # 尝试解析为日期：抽样非空值大多能解析即认定
    sample = series.dropna().astype(str).head(50)
    if len(sample) and (pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() > 0.8):
        return "datetime"
    return "text"
