"""calculator 工具单元测试（不依赖 API Key）。"""

import pytest

from qa_agent.tools.calculator import ExpressionError, evaluate


class TestBasicArithmetic:
    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("1+2", "3"),
            ("23*7", "161"),
            ("10/4", "2.5"),
            ("7-9", "-2"),
            ("2**10", "1024"),
            ("2^10", "1024"),  # ^ 兼容为幂
            ("(1+2)*3^2", "27"),
            ("10//3", "3"),
            ("10%3", "1"),
            ("-5+3", "-2"),
            ("+5*2", "10"),
            ("pi*2", "6.283185307179586"),
            ("e*1", "2.718281828459045"),
            ("  ( 1 + 2 ) * 3  ", "9"),
        ],
    )
    def test_values(self, expr, expected):
        assert evaluate(expr) == expected

    def test_float_result_keeps_decimals(self):
        assert evaluate("1/3") == "0.3333333333333333"

    def test_float_integer_result_normalized(self):
        assert evaluate("4/2") == "2"  # 2.0 -> "2"


class TestRejections:
    @pytest.mark.parametrize(
        "expr",
        [
            "",            # 空表达式
            "__import__('os')",  # 注入尝试
            "print(1)",    # 非数学元素
            "1 +",         # 语法错误
            "a+b",         # 未知变量
            "9" + "**9" * 2,  # 9**9**9 指数爆炸
        ],
    )
    def test_raises(self, expr):
        with pytest.raises(Exception):
            evaluate(expr)

    def test_injection_is_valueerror_not_execution(self):
        # 注入尝试应被拒绝而不是被执行
        with pytest.raises(ExpressionError):
            evaluate("__import__('os').system('echo hacked')")

    def test_overlong_rejected(self):
        with pytest.raises(ExpressionError):
            evaluate("1+" * 200 + "1")


class TestDivisionByZero:
    def test_zero_division(self):
        with pytest.raises(ZeroDivisionError):
            evaluate("1/0")
