# SPDX-License-Identifier: Apache-2.0
"""Safe, trailing-only expression trees for formulaic factor research.

The parser accepts a small expression grammar and builds its own immutable
tree. Evaluation never compiles or executes Python source. Time-series
operators use only the current and earlier rows; callers still own the
training-window and holdout boundary.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import cast

import numpy as np

from signalquarry._internal.canonical import canonical_hash
from signalquarry.sdk import xs
from signalquarry.sdk.factors import FactorCtx

GRAMMAR_VERSION = 1
_SCHEMA = "signalquarry.factor-expression/v1"
_MAX_SOURCE_CHARS = 512
_MAX_NODES = 64
_MAX_DEPTH = 16
_MAX_PERIOD = 252
_MAX_CONSTANT = 1_000_000.0
_PANEL_FIELDS = frozenset({"open", "high", "low", "close", "volume"})
_SIMPLE_CALLS = frozenset({"log", "abs", "sign", "rank", "zscore"})
_WINDOW_CALLS = frozenset({"delay", "delta", "ts_mean", "ts_std", "ts_rank"})
_BINARY_OPERATORS = {
    ast.Add: "add",
    ast.Sub: "subtract",
    ast.Mult: "multiply",
    ast.Div: "divide",
}


@dataclass(frozen=True)
class _Node:
    op: str
    args: tuple[_Node | str | float | int, ...]


@dataclass(frozen=True)
class FactorExpression:
    """Parsed factor expression with a versioned, stable source identity."""

    canonical: str
    identity: str
    _root: _Node


def _invalid() -> ValueError:
    return ValueError("FACTOR_EXPRESSION_INVALID")


def _constant(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid()
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        raise _invalid() from None
    if not math.isfinite(numeric) or abs(numeric) > _MAX_CONSTANT:
        raise _invalid()
    return numeric


def _period_node(node: ast.expr, state: list[int]) -> int:
    state[0] += 1
    if state[0] > _MAX_NODES or not isinstance(node, ast.Constant) or type(node.value) is not int:
        raise _invalid()
    value = node.value
    if not 1 <= value <= _MAX_PERIOD:
        raise _invalid()
    return value


def _parse_node(node: ast.expr, *, depth: int, state: list[int]) -> _Node:
    state[0] += 1
    if state[0] > _MAX_NODES or depth > _MAX_DEPTH:
        raise _invalid()

    if isinstance(node, ast.Name):
        if node.id not in _PANEL_FIELDS:
            raise _invalid()
        return _Node("panel", (node.id,))

    if isinstance(node, ast.Constant):
        return _Node("constant", (_constant(node.value),))

    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
        op = "positive" if type(node.op) is ast.UAdd else "negative"
        return _Node(op, (_parse_node(node.operand, depth=depth + 1, state=state),))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        op = _BINARY_OPERATORS[type(node.op)]
        return _Node(
            op,
            (
                _parse_node(node.left, depth=depth + 1, state=state),
                _parse_node(node.right, depth=depth + 1, state=state),
            ),
        )

    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.keywords:
        raise _invalid()
    name = node.func.id
    if name in _SIMPLE_CALLS and len(node.args) == 1:
        return _Node(name, (_parse_node(node.args[0], depth=depth + 1, state=state),))
    if name in _WINDOW_CALLS and len(node.args) == 2:
        value = _parse_node(node.args[0], depth=depth + 1, state=state)
        period = _period_node(node.args[1], state)
        return _Node(name, (value, period))
    if name == "ts_corr" and len(node.args) == 3:
        left = _parse_node(node.args[0], depth=depth + 1, state=state)
        right = _parse_node(node.args[1], depth=depth + 1, state=state)
        period = _period_node(node.args[2], state)
        return _Node(name, (left, right, period))
    raise _invalid()


def _render(node: _Node) -> str:
    if node.op == "panel":
        return cast(str, node.args[0])
    if node.op == "constant":
        return repr(cast(float, node.args[0]))
    if node.op in ("positive", "negative"):
        symbol = "+" if node.op == "positive" else "-"
        return f"({symbol}{_render(cast(_Node, node.args[0]))})"
    if node.op in ("add", "subtract", "multiply", "divide"):
        symbols = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
        return (
            f"({_render(cast(_Node, node.args[0]))} {symbols[node.op]} {_render(cast(_Node, node.args[1]))})"
        )
    rendered = [_render(arg) if isinstance(arg, _Node) else str(arg) for arg in node.args]
    return f"{node.op}({', '.join(rendered)})"


def _document(node: _Node) -> list[object]:
    return [
        node.op,
        *[_document(arg) if isinstance(arg, _Node) else arg for arg in node.args],
    ]


def _has_panel(node: _Node) -> bool:
    return node.op == "panel" or any(isinstance(arg, _Node) and _has_panel(arg) for arg in node.args)


def parse_expression(source: str) -> FactorExpression:
    """Parse one bounded expression; reject every Python construct outside the grammar."""
    if not isinstance(source, str) or not source.strip() or len(source) > _MAX_SOURCE_CHARS:
        raise _invalid()
    try:
        parsed = ast.parse(source.strip(), mode="eval")
        root = _parse_node(parsed.body, depth=1, state=[0])
    except (SyntaxError, RecursionError):
        raise _invalid() from None
    if not _has_panel(root):
        raise _invalid()
    canonical = _render(root)
    identity = canonical_hash(
        {"schema": _SCHEMA, "grammar_version": GRAMMAR_VERSION, "expression": _document(root)}
    )
    return FactorExpression(canonical, identity, root)


def _node_complexity(node: _Node) -> int:
    return 1 + sum(_node_complexity(arg) for arg in node.args if isinstance(arg, _Node))


def expression_complexity(expression: FactorExpression) -> int:
    """Return the number of immutable AST nodes in a parsed expression."""
    if not isinstance(expression, FactorExpression):
        raise TypeError("FACTOR_EXPRESSION_INVALID")
    return _node_complexity(expression._root)


def _matrix(value: float | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.full(shape, value, dtype=np.float64)


def _clean(value: float | np.ndarray) -> float | np.ndarray:
    if isinstance(value, np.ndarray):
        result = np.asarray(value, dtype=np.float64)
        if np.isfinite(result).all():
            return result
        return np.where(np.isfinite(result), result, np.nan)
    return value if math.isfinite(value) else float("nan")


def _rowwise(values: np.ndarray, operation: str, eligible: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    function = xs.rank if operation == "rank" else xs.zscore
    for row in range(len(values)):
        result[row] = function(np.where(eligible[row], values[row], np.nan))
    return result


def _delay(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if period < len(values):
        result[period:] = values[:-period]
    return result


def _rolling(values: np.ndarray, period: int, operation: str, other: np.ndarray | None = None) -> np.ndarray:
    rows, columns = values.shape
    result = np.full(values.shape, np.nan, dtype=np.float64)
    counts = np.zeros(columns, dtype=np.int16)
    sum_x = np.zeros(columns, dtype=np.float64)
    sum_x2 = np.zeros(columns, dtype=np.float64)
    sum_y = np.zeros(columns, dtype=np.float64)
    sum_y2 = np.zeros(columns, dtype=np.float64)
    sum_xy = np.zeros(columns, dtype=np.float64)

    for end in range(rows):
        current_x = values[end]
        finite_x = np.isfinite(current_x)
        if other is None:
            counts += finite_x
            sum_x += np.where(finite_x, current_x, 0.0)
            sum_x2 += np.where(finite_x, current_x * current_x, 0.0)
        else:
            current_y = other[end]
            finite = finite_x & np.isfinite(current_y)
            counts += finite
            sum_x += np.where(finite, current_x, 0.0)
            sum_x2 += np.where(finite, current_x * current_x, 0.0)
            sum_y += np.where(finite, current_y, 0.0)
            sum_y2 += np.where(finite, current_y * current_y, 0.0)
            sum_xy += np.where(finite, current_x * current_y, 0.0)

        if end >= period:
            outgoing_x = values[end - period]
            outgoing_finite_x = np.isfinite(outgoing_x)
            if other is None:
                counts -= outgoing_finite_x
                sum_x -= np.where(outgoing_finite_x, outgoing_x, 0.0)
                sum_x2 -= np.where(outgoing_finite_x, outgoing_x * outgoing_x, 0.0)
            else:
                outgoing_y = other[end - period]
                outgoing_finite = outgoing_finite_x & np.isfinite(outgoing_y)
                counts -= outgoing_finite
                sum_x -= np.where(outgoing_finite, outgoing_x, 0.0)
                sum_x2 -= np.where(outgoing_finite, outgoing_x * outgoing_x, 0.0)
                sum_y -= np.where(outgoing_finite, outgoing_y, 0.0)
                sum_y2 -= np.where(outgoing_finite, outgoing_y * outgoing_y, 0.0)
                sum_xy -= np.where(outgoing_finite, outgoing_x * outgoing_y, 0.0)

        if end + 1 < period:
            continue
        valid = counts == period
        if operation == "ts_mean":
            result[end, valid] = sum_x[valid] / period
        elif operation == "ts_std":
            mean = sum_x[valid] / period
            variance = np.maximum(sum_x2[valid] / period - mean * mean, 0.0)
            result[end, valid] = np.sqrt(variance)
        else:
            mean_x = sum_x[valid] / period
            mean_y = sum_y[valid] / period
            variance_x = sum_x2[valid] / period - mean_x * mean_x
            variance_y = sum_y2[valid] / period - mean_y * mean_y
            tolerance_x = np.finfo(np.float64).eps * np.maximum(sum_x2[valid] / period, 1.0)
            tolerance_y = np.finfo(np.float64).eps * np.maximum(sum_y2[valid] / period, 1.0)
            defined = (variance_x > tolerance_x) & (variance_y > tolerance_y)
            if defined.any():
                covariance = sum_xy[valid] / period - mean_x * mean_y
                correlation = covariance[defined] / np.sqrt(variance_x[defined] * variance_y[defined])
                result[end, np.flatnonzero(valid)[defined]] = np.clip(correlation, -1.0, 1.0)
    return result


def _rolling_rank(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for end in range(period - 1, len(values)):
        window = values[end - period + 1 : end + 1]
        valid = np.isfinite(window).all(axis=0)
        if not valid.any():
            continue
        current = window[-1]
        less = np.count_nonzero(window < current[None, :], axis=0)
        equal = np.count_nonzero(window == current[None, :], axis=0)
        result[end, valid] = (less[valid] + equal[valid] / 2.0) / period
    return result


def _evaluate_node(
    node: _Node, panels: dict[str, np.ndarray], shape: tuple[int, int], eligible: np.ndarray
) -> float | np.ndarray:
    if node.op == "panel":
        return panels[cast(str, node.args[0])]
    if node.op == "constant":
        return cast(float, node.args[0])

    if node.op in ("positive", "negative", "log", "abs", "sign", "rank", "zscore"):
        value = _evaluate_node(cast(_Node, node.args[0]), panels, shape, eligible)
        if node.op == "positive":
            result = value
        elif node.op == "negative":
            result = -value
        elif node.op == "log":
            matrix = _matrix(value, shape)
            with np.errstate(all="ignore"):
                result = np.where(matrix > 0, np.log(np.where(matrix > 0, matrix, 1.0)), np.nan)
        elif node.op == "abs":
            result = np.abs(value)
        elif node.op == "sign":
            result = np.sign(value)
        else:
            result = _rowwise(_matrix(value, shape), node.op, eligible)
        return _clean(result)

    if node.op in ("delay", "delta", "ts_mean", "ts_std", "ts_rank"):
        value = _matrix(_evaluate_node(cast(_Node, node.args[0]), panels, shape, eligible), shape)
        period = cast(int, node.args[1])
        if node.op == "delay":
            result = _delay(value, period)
        elif node.op == "delta":
            with np.errstate(all="ignore"):
                result = value - _delay(value, period)
        elif node.op == "ts_rank":
            result = _rolling_rank(value, period)
        else:
            result = _rolling(value, period, node.op)
        return _clean(result)

    if node.op == "ts_corr":
        left = _matrix(_evaluate_node(cast(_Node, node.args[0]), panels, shape, eligible), shape)
        right = _matrix(_evaluate_node(cast(_Node, node.args[1]), panels, shape, eligible), shape)
        result = _rolling(left, cast(int, node.args[2]), node.op, right)
        return _clean(result)

    left = _matrix(_evaluate_node(cast(_Node, node.args[0]), panels, shape, eligible), shape)
    right = _matrix(_evaluate_node(cast(_Node, node.args[1]), panels, shape, eligible), shape)
    with np.errstate(all="ignore"):
        if node.op == "add":
            result = left + right
        elif node.op == "subtract":
            result = left - right
        elif node.op == "multiply":
            result = left * right
        else:
            result = left / right
    return _clean(result)


def _panel_names(node: _Node) -> set[str]:
    names = {cast(str, node.args[0])} if node.op == "panel" else set()
    for arg in node.args:
        if isinstance(arg, _Node):
            names.update(_panel_names(arg))
    return names


def _immutable(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.tobytes(order="C"), dtype=values.dtype).reshape(values.shape)


def evaluate_expression(
    expression: FactorExpression, context: FactorCtx, *, eligible: np.ndarray
) -> np.ndarray:
    """Evaluate every row using only that row and its trailing history.

    Eligibility is a dated boolean membership mask with the same shape as the
    panel. Cross-sectional operators rank only eligible symbols in each row;
    output scores for ineligible symbols are NaN. Its last row is the current
    pre-decision cross-section when the context came from the factor engine.
    """
    if not isinstance(expression, FactorExpression):
        raise TypeError("FACTOR_EXPRESSION_INVALID")
    sessions = context.sessions
    universe = context.universe
    shape = (len(sessions), len(universe))
    membership = np.asarray(eligible)
    if (
        not sessions
        or not universe
        or sessions != tuple(sorted(set(sessions)))
        or any(session >= context.decision_session for session in sessions)
        or len(universe) != len(set(universe))
        or membership.dtype != np.bool_
        or membership.shape != shape
    ):
        raise ValueError("FACTOR_EXPRESSION_CONTEXT_INVALID")

    panels: dict[str, np.ndarray] = {}
    for field in _panel_names(expression._root):
        try:
            values = np.asarray(context.panel(field), dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            raise ValueError("FACTOR_EXPRESSION_CONTEXT_INVALID") from None
        if values.shape != shape or np.isinf(values).any():
            raise ValueError("FACTOR_EXPRESSION_CONTEXT_INVALID")
        panels[field] = values

    result = _evaluate_node(expression._root, panels, shape, membership)
    output = _matrix(result, shape)
    output = np.where(membership & np.isfinite(output), output, np.nan)
    return _immutable(np.asarray(output, dtype=np.float64))


__all__ = ["FactorExpression", "evaluate_expression", "expression_complexity", "parse_expression"]
