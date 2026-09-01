"""stats 与 report 模块测试。"""
from csv_cleaner.core.cleaner import Cleaner
from csv_cleaner.core.inspector import inspect
from csv_cleaner.core.report import export_report
from csv_cleaner.core import stats


def test_numeric_summary_and_correlation(dirty_df):
    c = Cleaner(dirty_df)
    c.convert_numeric("单价")
    desc = stats.numeric_summary(c.df)
    assert "数量" in desc.index
    assert "mean" in desc.columns

    corr = stats.correlation_matrix(c.df)
    assert corr.loc["数量", "单价"] is not None


def test_category_top(dirty_df):
    top = stats.category_top(dirty_df, "商品", n=3)
    assert len(top) <= 3
    assert top.index[0] in {"水", "可乐"}


def test_export_report_writes_markdown(dirty_df, tmp_path):
    c = Cleaner(dirty_df)
    c.drop_duplicates()
    c.convert_numeric("单价")
    inspection = inspect(c.df)
    out = export_report(c.df, inspection, c, tmp_path / "r.md", source_name="test.csv")
    text = out.read_text(encoding="utf-8")
    assert "# CSV 数据清洗分析报告" in text
    assert "## 二、清洗操作记录" in text
    assert "去除重复行" in text
    assert "test.csv" in text
