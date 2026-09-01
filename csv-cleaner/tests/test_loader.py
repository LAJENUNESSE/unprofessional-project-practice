"""loader 模块测试：编码识别与读取。"""
from csv_cleaner.core.loader import read_csv_any, sniff_encoding


def test_read_utf8_csv(samples_dir):
    df = read_csv_any(samples_dir / "messy_sales.csv")
    assert not df.empty
    assert "订单ID" in df.columns
    assert len(df) == 34


def test_read_gbk_csv(samples_dir):
    """GBK 编码文件应能自动识别并正确读出中文。"""
    path = samples_dir / "messy_sales_gbk.csv"
    assert sniff_encoding(path) == "gbk"
    df = read_csv_any(path)
    assert not df.empty
    assert "商品名称" in df.columns
    assert df["商品名称"].iloc[0] == "矿泉水"
