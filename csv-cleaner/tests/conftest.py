"""测试公共夹具：构造脏数据 DataFrame。"""
import sys
import pathlib

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

SAMPLES = pathlib.Path(__file__).resolve().parents[1] / "samples"


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    """小型脏数据：缺失、重复、货币符号、异常值、空格。"""
    return pd.DataFrame({
        "商品": ["水", "可乐", "薯片", "水", "面包", "可乐", "牙膏"],
        "数量": [2.0, 1.0, None, 2.0, 9999.0, 1.0, 3.0],
        "单价": ["¥2.5", "3.5", "6.8", "¥2.5", " 8.0 ", "3.5", "-9.9"],
        "城市": [" 北京", "上海", "广州", " 北京", "深圳", "上海", "成都"],
    })


@pytest.fixture
def samples_dir() -> pathlib.Path:
    return SAMPLES
