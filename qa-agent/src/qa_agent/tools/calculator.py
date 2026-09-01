"""计算器工具：安全表达式求值。

只允许白名单字符（数字、四则运算符、括号、小数点、幂符号），
通过 ast 解析为受限节点后求值，绝不使用 eval 直接执行用户输入。
"""

import ast
import operator

from agents import function_tool

# 二元运算符 -> 实际函数
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# 一元运算符（正负号）
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# 允许的常量名
_CONSTS = {
    "pi": 3.141592653589793,
    "e": 2.718281828459045,
}

_MAX_EXPR_LEN = 200
_MAX_POW_EXP = 1000  # 幂运算指数上限，防止 9**9**9 这类爆炸输入


class ExpressionError(ValueError):
    """表达式不合法或超出支持范围。"""


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXP:
            raise ExpressionError(f"幂指数过大：{right}")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ExpressionError(f"不支持的表达式元素：{ast.dump(node)[:60]}")


def evaluate(expression: str) -> str:
    """计算数学表达式，返回格式化结果字符串。"""
    expr = expression.strip()
    if not expr or len(expr) > _MAX_EXPR_LEN:
        raise ExpressionError("表达式为空或过长")
    if "^" in expr:
        expr = expr.replace("^", "**")  # 常见写法兼容
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"表达式语法错误：{exc.msg}") from exc
    result = _eval_node(tree)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


@function_tool
def calculator(expression: str) -> str:
    """计算一个数学表达式并返回结果。

    支持四则运算（+ - * /）、整除（//）、取模（%）、幂（** 或 ^）、
    括号，以及常量 pi 和 e。示例："23*7"、"（1+2)*3^2"、"sqrt 不支持，请用 **0.5"。

    Args:
        expression: 数学表达式，例如 "3+4*2" 或 "(1+2)*3^2"。
    """
    try:
        return evaluate(expression)
    except ExpressionError as exc:
        return f"计算失败：{exc}"
    except ZeroDivisionError:
        return "计算失败：除数为零"
    except OverflowError:
        return "计算失败：结果超出可表示范围"
