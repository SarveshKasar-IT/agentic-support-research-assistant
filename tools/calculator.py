import ast
import operator as op

from langchain_core.tools import tool


# Allowed arithmetic operations
OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _calculate(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):
        operator = OPERATORS.get(type(node.op))

        if operator is None:
            raise ValueError("Unsupported operator.")

        left = _calculate(node.left)
        right = _calculate(node.right)

        return operator(left, right)

    if isinstance(node, ast.UnaryOp):
        operator = OPERATORS.get(type(node.op))

        if operator is None:
            raise ValueError("Unsupported operator.")

        return operator(_calculate(node.operand))

    raise ValueError("Invalid mathematical expression.")


@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Use this tool for arithmetic calculations such as:
    15 * 850
    100 / 4
    (25 + 15) * 2
    2 ** 10
    """

    try:
        tree = ast.parse(expression, mode="eval")
        result = _calculate(tree.body)

        return str(result)

    except Exception as e:
        return f"Calculation error: {e}"