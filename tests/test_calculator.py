import pytest

from tools.calculator import calculator


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("15 * 850", "12750"),
        ("100 / 4", "25.0"),
        ("(25 + 15) * 2", "80"),
        ("2 ** 10", "1024"),
        ("17 % 5", "2"),
        ("20 - 8", "12"),
        ("-5 + 3", "-2"),
        ("+7", "7"),
    ],
)
def test_valid_calculations(expression, expected):
    assert calculator.invoke({"expression": expression}) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "2 +",
        "10 / 0",
        "hello",
        "__import__('os').system('echo unsafe')",
        "[1, 2, 3]",
    ],
)
def test_invalid_calculations(expression):
    result = calculator.invoke({"expression": expression})
    assert result.startswith("Calculation error:")


def test_calculator_rejects_unsupported_operator():
    result = calculator.invoke({"expression": "10 // 3"})
    assert "Calculation error:" in result
