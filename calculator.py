"""
calculator.py
-------------
Gives the Analyst Agent exact arithmetic instead of letting the LLM
"guess" numbers. Every method returns a small dict with the inputs,
operation, and result so it can be logged/cited (e.g. "Calculator:
average of [92, 95, 89] = 92.0").
"""

import ast
import operator
import statistics
from typing import Dict, List, Union

Number = Union[int, float]

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


class CalculatorError(ValueError):
    pass


class Calculator:
    """Deterministic numeric tool used by the Analyst Agent."""

    # ---- basic ops -----------------------------------------------------
    def add(self, *values: Number) -> Dict:
        return self._wrap("add", values, sum(values))

    def subtract(self, a: Number, b: Number) -> Dict:
        return self._wrap("subtract", (a, b), a - b)

    def multiply(self, *values: Number) -> Dict:
        result = 1
        for v in values:
            result *= v
        return self._wrap("multiply", values, result)

    def divide(self, a: Number, b: Number) -> Dict:
        if b == 0:
            raise CalculatorError("Division by zero")
        return self._wrap("divide", (a, b), a / b)

    # ---- stats / comparisons -------------------------------------------
    def average(self, values: List[Number]) -> Dict:
        if not values:
            raise CalculatorError("average() needs at least one value")
        return self._wrap("average", values, statistics.mean(values))

    def median(self, values: List[Number]) -> Dict:
        if not values:
            raise CalculatorError("median() needs at least one value")
        return self._wrap("median", values, statistics.median(values))

    def stdev(self, values: List[Number]) -> Dict:
        if len(values) < 2:
            raise CalculatorError("stdev() needs at least two values")
        return self._wrap("stdev", values, statistics.stdev(values))

    def percentage(self, part: Number, whole: Number) -> Dict:
        if whole == 0:
            raise CalculatorError("percentage() whole cannot be zero")
        return self._wrap("percentage", (part, whole), (part / whole) * 100)

    def percent_change(self, old: Number, new: Number) -> Dict:
        if old == 0:
            raise CalculatorError("percent_change() old value cannot be zero")
        return self._wrap("percent_change", (old, new), ((new - old) / old) * 100)

    def ratio(self, a: Number, b: Number) -> Dict:
        if b == 0:
            raise CalculatorError("ratio() b cannot be zero")
        return self._wrap("ratio", (a, b), a / b)

    def min_max(self, values: List[Number]) -> Dict:
        if not values:
            raise CalculatorError("min_max() needs at least one value")
        return self._wrap("min_max", values, {"min": min(values), "max": max(values)})

    # ---- free-form expressions -------------------------------------------
    def evaluate(self, expression: str) -> Dict:
        """
        Safely evaluate a numeric expression like "(92+95+89)/3".
        Only numbers and +-*/%() are allowed (no names, calls, attrs).
        """
        try:
            tree = ast.parse(expression, mode="eval")
            result = self._eval_node(tree.body)
        except (SyntaxError, TypeError, ZeroDivisionError) as e:
            raise CalculatorError(f"Could not evaluate '{expression}': {e}")
        return self._wrap("evaluate", expression, result)

    def _eval_node(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](
                self._eval_node(node.left), self._eval_node(node.right)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](self._eval_node(node.operand))
        raise CalculatorError("Expression contains disallowed syntax")

    # ---- helper -----------------------------------------------------------
    @staticmethod
    def _wrap(op: str, inputs, result) -> Dict:
        return {"tool": "calculator", "operation": op, "inputs": inputs, "result": result}
