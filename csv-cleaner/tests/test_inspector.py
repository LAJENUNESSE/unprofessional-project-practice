"""inspector 模块测试。"""
from csv_cleaner.core.inspector import inspect


def test_inspect_reports_duplicates_and_missing(dirty_df):
    report = inspect(dirty_df)
    assert report.n_rows == 7
    assert report.duplicate_rows == 2  # 「水」和「可乐」各有一对重复行

    by_name = {c.name: c for c in report.columns}
    assert by_name["数量"].missing == 1
    assert by_name["单价"].convertible  # 带货币符号的文本列应被识别为可转数值
    assert by_name["商品"].kind == "text"


def test_inspect_counts_outliers(dirty_df):
    report = inspect(dirty_df)
    by_name = {c.name: c for c in report.columns}
    # 数量列 9999 是明显异常值（其余为 1~3）
    assert by_name["数量"].outliers >= 1


def test_to_lines_contains_problem_notes(dirty_df):
    lines = inspect(dirty_df).to_lines()
    text = "\n".join(lines)
    assert "重复行" in text and "2 行" in text
