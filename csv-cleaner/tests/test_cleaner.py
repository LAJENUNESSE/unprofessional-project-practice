"""cleaner 模块测试：每个清洗动作与日志记录。"""
import pytest

from csv_cleaner.core.cleaner import Cleaner


def test_drop_duplicates(dirty_df):
    c = Cleaner(dirty_df)
    removed = c.drop_duplicates()
    assert removed == 2
    assert len(c.df) == 5
    assert c.log[-1].action == "去除重复行"


def test_fill_missing_mean(dirty_df):
    c = Cleaner(dirty_df)
    n = c.fill_missing("数量", "mean")
    assert n == 1
    assert c.df["数量"].isna().sum() == 0
    filled = c.df.loc[2, "数量"]
    assert filled == pytest.approx((2 + 1 + 2 + 9999 + 1 + 3) / 6)


def test_fill_missing_mode_and_constant(dirty_df):
    c = Cleaner(dirty_df)
    c.fill_missing("数量", "mode")
    assert c.df["数量"].isna().sum() == 0

    c2 = Cleaner(dirty_df)
    c2.fill_missing("数量", "constant", fill_value=0)
    assert (c2.df["数量"] == 0).sum() == 1


def test_fill_missing_constant_requires_value(dirty_df):
    with pytest.raises(ValueError):
        Cleaner(dirty_df).fill_missing("数量", "constant")


def test_convert_numeric_strips_currency(dirty_df):
    c = Cleaner(dirty_df)
    c.convert_numeric("单价")
    s = c.df["单价"]
    assert s.iloc[0] == pytest.approx(2.5)
    assert s.iloc[4] == pytest.approx(8.0)  # 带空格的数值
    # -9.9 负数保留（异常值交给异常值处理）
    assert s.iloc[6] == pytest.approx(-9.9)


def test_drop_missing_rows(dirty_df):
    c = Cleaner(dirty_df)
    removed = c.drop_missing_rows(subset=["数量"])
    assert removed == 1
    assert c.df["数量"].isna().sum() == 0


def test_handle_outliers_clip(dirty_df):
    c = Cleaner(dirty_df)
    n = c.handle_outliers("数量", action="clip")
    assert n == 1
    assert c.df["数量"].max() < 9999


def test_handle_outliers_remove(dirty_df):
    c = Cleaner(dirty_df)
    before = len(c.df)
    n = c.handle_outliers("数量", action="remove")
    assert n == 1
    assert len(c.df) == before - 1


def test_strip_whitespace(dirty_df):
    c = Cleaner(dirty_df)
    changed = c.strip_whitespace()
    assert changed >= 2
    assert (c.df["城市"] == " 北京").sum() == 0


def test_rename_and_drop_columns(dirty_df):
    c = Cleaner(dirty_df)
    c.rename_column("商品", "品名")
    assert "品名" in c.df.columns and "商品" not in c.df.columns
    c.drop_columns(["品名", "不存在的列"])
    assert "品名" not in c.df.columns


def test_log_records_steps_in_order(dirty_df):
    c = Cleaner(dirty_df)
    c.drop_duplicates()
    c.strip_whitespace()
    assert [a.step for a in c.log] == [1, 2]
    assert c.log[0].action == "去除重复行"
    assert c.log[1].action == "去首尾空格"
