# SPDX-License-Identifier: Apache-2.0
"""Deterministic, training-only formula search over the safe expression grammar.

This pilot accepts only explicitly synthetic outcomes. It estimates a
multiple-testing-adjusted discovery score for candidate ordering, but writes no
trial ledger, issues no evidence grade, and cannot open a holdout.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from statistics import NormalDist
from types import MappingProxyType
from typing import Literal, cast

import numpy as np

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.factors.evaluate import (
    ScorePanel,
    SyntheticLabels,
    rank_ic,
    redundancy,
)
from signalquarry._internal.factors.expr import (
    FactorExpression,
    _has_panel,
    _Node,
    _render,
    evaluate_expression,
    expression_complexity,
    parse_expression,
)
from signalquarry.sdk.factors import FactorCtx

_IDENTITY_PREFIX = "sha256:"
_IDENTITY_HEX_LENGTH = 64
_FIELDS = ("open", "high", "low", "close", "volume")
_UNARY = ("positive", "negative", "log", "abs", "sign", "rank", "zscore")
_BINARY = ("add", "subtract", "multiply", "divide")
_WINDOW = ("delay", "delta", "ts_mean", "ts_std", "ts_rank")
_PERIODS = (1, 2, 3, 5, 10, 20)
_CONSTANTS = (-1.0, -0.5, 0.5, 1.0, 2.0)
_NORMAL = NormalDist()
_EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class FormulaTrainingInput:
    """One immutable, explicitly dated training panel and synthetic labels."""

    dataset_identity: str
    universe_identity: str
    context: FactorCtx
    eligible: np.ndarray
    labels: SyntheticLabels


@dataclass(frozen=True)
class FormulaSearchConfig:
    training_cutoff: date
    horizon: int
    seed: int
    budget: int
    family_budget: int = 50
    family_trials_used: int = 0
    project_trials_used: int = 0
    population_size: int = 12
    chronological_blocks: int = 6
    complexity_penalty: float = 0.01
    redundancy_penalty: float = 0.25
    fdr_alpha: float = 0.05
    previous_family_p_values: tuple[float, ...] = ()


@dataclass(frozen=True)
class FormulaCandidate:
    expression: str
    identity: str
    complexity: int
    observations: int
    effective_observations: float
    mean_ic: float | None
    icir: float | None
    t_statistic: float | None
    deflated_t_statistic: float | None
    p_value: float
    q_value: float
    max_abs_library_correlation: float
    fitness: float
    discovery: bool


@dataclass(frozen=True)
class FormulaSearchReport:
    scope: Literal["synthetic"]
    dataset_identity: str
    universe_identity: str
    training_label_identity: str
    training_start: date
    training_cutoff: date
    horizon: int
    seed: int
    budget: int
    family_trials_used_before: int
    project_trials_used_before: int
    trials_evaluated: int
    family_trials_after_batch: int
    discoveries: tuple[str, ...]
    candidates: tuple[FormulaCandidate, ...]
    evidence_grade: Literal["none"]
    trial_ledger_written: Literal[False]
    holdout_accessed: Literal[False]


@dataclass(frozen=True)
class _ValidatedInput:
    dataset_identity: str
    universe_identity: str
    context: FactorCtx
    eligible: np.ndarray
    labels: SyntheticLabels


@dataclass(frozen=True)
class _Scored:
    expression: FactorExpression
    complexity: int
    observations: int
    effective_observations: float
    mean_ic: float | None
    icir: float | None
    t_statistic: float | None
    max_abs_library_correlation: float
    preliminary_fitness: float


def _identity(value: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(_IDENTITY_PREFIX)
        and len(value) == len(_IDENTITY_PREFIX) + _IDENTITY_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value.removeprefix(_IDENTITY_PREFIX))
    )


def _immutable(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.tobytes(order="C"), dtype=values.dtype).reshape(values.shape)


def _nullable_rows(values: np.ndarray) -> list[list[float | None]]:
    return [[float(value) if math.isfinite(float(value)) else None for value in row] for row in values]


def _validate_config(config: FormulaSearchConfig) -> None:
    if not isinstance(config, FormulaSearchConfig):
        raise ValueError("FACTOR_SEARCH_CONFIG_INVALID")
    integer_values = (
        config.horizon,
        config.seed,
        config.budget,
        config.family_budget,
        config.family_trials_used,
        config.project_trials_used,
        config.population_size,
        config.chronological_blocks,
    )
    numeric_values = (config.complexity_penalty, config.redundancy_penalty, config.fdr_alpha)
    if (
        type(config.training_cutoff) is not date
        or any(type(value) is not int for value in integer_values)
        or any(type(value) not in (int, float) for value in numeric_values)
        or not 1 <= config.horizon <= 252
        or not 0 <= config.seed < 2**64
        or not 1 <= config.budget <= 10_000
        or not 1 <= config.family_budget <= 10_000
        or not 0 <= config.family_trials_used <= config.family_budget
        or config.project_trials_used < config.family_trials_used
        or not 2 <= config.population_size <= 128
        or not 1 <= config.chronological_blocks <= 100
        or not math.isfinite(config.complexity_penalty)
        or not 0 <= config.complexity_penalty <= 10
        or not math.isfinite(config.redundancy_penalty)
        or not 0 <= config.redundancy_penalty <= 10
        or not math.isfinite(config.fdr_alpha)
        or not 0 < config.fdr_alpha <= 0.2
        or not isinstance(config.previous_family_p_values, tuple)
        or len(config.previous_family_p_values) > config.family_trials_used
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in config.previous_family_p_values
        )
    ):
        raise ValueError("FACTOR_SEARCH_CONFIG_INVALID")
    if config.budget > config.family_budget - config.family_trials_used:
        raise ValueError("FACTOR_SEARCH_BUDGET_EXHAUSTED")


def _validate_input(data: FormulaTrainingInput, config: FormulaSearchConfig) -> _ValidatedInput:
    if not isinstance(data, FormulaTrainingInput):
        raise ValueError("FACTOR_SEARCH_INPUT_INVALID")
    context = data.context
    labels = data.labels
    if (
        not _identity(data.dataset_identity)
        or not _identity(data.universe_identity)
        or not isinstance(context, FactorCtx)
        or not isinstance(labels, SyntheticLabels)
        or not _identity(labels.dataset_identity)
        or not _identity(labels.label_identity)
        or data.dataset_identity != labels.dataset_identity
    ):
        raise ValueError("FACTOR_SEARCH_INPUT_INVALID")

    sessions = context.sessions
    symbols = context.universe
    if (
        type(context.decision_session) is not date
        or not isinstance(sessions, tuple)
        or any(type(session) is not date for session in sessions)
        or not sessions
        or sessions != tuple(sorted(set(sessions)))
        or any(session >= context.decision_session for session in sessions)
        or sessions[-1] != config.training_cutoff
        or not symbols
        or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
        or len(symbols) != len(set(symbols))
        or labels.sessions[: len(sessions)] != sessions
        or labels.symbols != symbols
        or config.horizon not in labels.forward_returns
        or labels.outcome_end_sessions is None
        or config.horizon not in labels.outcome_end_sessions
    ):
        raise ValueError("FACTOR_SEARCH_INPUT_INVALID")

    shape = (len(sessions), len(symbols))
    membership = np.asarray(data.eligible)
    if membership.dtype != np.bool_ or membership.shape != shape:
        raise ValueError("FACTOR_SEARCH_INPUT_INVALID")

    panels: dict[str, np.ndarray] = {}
    try:
        for field in _FIELDS:
            values = np.asarray(context.panel(field), dtype=np.float64)
            if values.shape != shape or np.isinf(values).any():
                raise ValueError("FACTOR_SEARCH_INPUT_INVALID")
            panels[field] = _immutable(values)
    except (KeyError, TypeError, ValueError):
        raise ValueError("FACTOR_SEARCH_INPUT_INVALID") from None

    raw_outcomes = np.asarray(labels.forward_returns[config.horizon])
    all_sessions = labels.sessions
    if (
        not isinstance(all_sessions, tuple)
        or any(type(session) is not date for session in all_sessions)
        or len(all_sessions) < len(sessions)
        or all_sessions != tuple(sorted(set(all_sessions)))
        or raw_outcomes.shape != (len(all_sessions), len(symbols))
    ):
        raise ValueError("FACTOR_SEARCH_INPUT_INVALID")

    all_ends = labels.outcome_end_sessions[config.horizon]
    if not isinstance(all_ends, tuple) or len(all_ends) != len(all_sessions):
        raise ValueError("FACTOR_SEARCH_INPUT_INVALID")
    train_outcomes = np.array(raw_outcomes[: len(sessions)], dtype=np.float64, copy=True)
    if np.isinf(train_outcomes).any() or np.any(np.isfinite(train_outcomes) & (train_outcomes < -1)):
        raise ValueError("FACTOR_SEARCH_INPUT_INVALID")
    train_ends: list[date | None] = []
    for row, (session, end_session) in enumerate(zip(sessions, all_ends, strict=False)):
        if end_session is not None and (type(end_session) is not date or end_session <= session):
            raise ValueError("FACTOR_SEARCH_INPUT_INVALID")
        if end_session is None:
            if np.isfinite(train_outcomes[row]).any():
                raise ValueError("FACTOR_SEARCH_INPUT_INVALID")
            train_outcomes[row] = np.nan
        elif end_session > config.training_cutoff:
            train_outcomes[row] = np.nan
        train_ends.append(end_session)

    adv = None
    if labels.predecision_adv is not None:
        raw_adv = np.asarray(labels.predecision_adv)
        if raw_adv.shape != (len(all_sessions), len(symbols)):
            raise ValueError("FACTOR_SEARCH_INPUT_INVALID")
        adv = _immutable(np.array(raw_adv[: len(sessions)], dtype=np.float64, copy=True))

    training_label_identity = canonical_hash(
        {
            "schema": "signalquarry.synthetic-factor-training-labels/v1",
            "dataset_identity": data.dataset_identity,
            "universe_identity": data.universe_identity,
            "horizon": config.horizon,
            "training_cutoff": config.training_cutoff.isoformat(),
            "sessions": [session.isoformat() for session in sessions],
            "symbols": symbols,
            "outcome_end_sessions": [item.isoformat() if item else None for item in train_ends],
            "forward_returns": _nullable_rows(train_outcomes),
        }
    )
    training_labels = SyntheticLabels(
        dataset_identity=data.dataset_identity,
        label_identity=training_label_identity,
        sessions=sessions,
        symbols=symbols,
        forward_returns=MappingProxyType({config.horizon: _immutable(train_outcomes)}),
        predecision_adv=adv,
        outcome_end_sessions=MappingProxyType({config.horizon: tuple(train_ends)}),
    )
    training_context = FactorCtx(
        decision_session=context.decision_session,
        sessions=sessions,
        universe=symbols,
        _panels=MappingProxyType(panels),
    )
    return _ValidatedInput(
        data.dataset_identity,
        data.universe_identity,
        training_context,
        _immutable(np.array(membership, copy=True)),
        training_labels,
    )


def _paths(node: _Node, path: tuple[int, ...] = ()) -> list[tuple[tuple[int, ...], _Node]]:
    found = [(path, node)]
    for index, argument in enumerate(node.args):
        if isinstance(argument, _Node):
            found.extend(_paths(argument, (*path, index)))
    return found


def _replace(node: _Node, path: tuple[int, ...], replacement: _Node) -> _Node:
    if not path:
        return replacement
    index = path[0]
    arguments = list(node.args)
    child = arguments[index]
    if not isinstance(child, _Node):
        return node
    arguments[index] = _replace(child, path[1:], replacement)
    return _Node(node.op, tuple(arguments))


def _terminal(rng: np.random.Generator) -> _Node:
    if rng.random() < 0.82:
        return _Node("panel", (str(rng.choice(_FIELDS)),))
    return _Node("constant", (float(rng.choice(_CONSTANTS)),))


def _random_tree(rng: np.random.Generator, depth: int, max_depth: int) -> _Node:
    if depth >= max_depth or rng.random() < 0.28:
        return _terminal(rng)
    kind = int(rng.integers(0, 4))
    if kind == 0:
        return _Node(str(rng.choice(_UNARY)), (_random_tree(rng, depth + 1, max_depth),))
    if kind == 1:
        return _Node(
            str(rng.choice(_WINDOW)),
            (_random_tree(rng, depth + 1, max_depth), int(rng.choice(_PERIODS))),
        )
    if kind == 2:
        return _Node(
            str(rng.choice(_BINARY)),
            (
                _random_tree(rng, depth + 1, max_depth),
                _random_tree(rng, depth + 1, max_depth),
            ),
        )
    return _Node(
        "ts_corr",
        (
            _random_tree(rng, depth + 1, max_depth),
            _random_tree(rng, depth + 1, max_depth),
            int(rng.choice(_PERIODS)),
        ),
    )


def _parse_root(root: _Node) -> FactorExpression | None:
    if not _has_panel(root):
        root = _Node("add", (root, _Node("panel", ("close",))))
    try:
        return parse_expression(_render(root))
    except (ValueError, RecursionError):
        return None


def _random_expression(rng: np.random.Generator, max_depth: int = 4) -> FactorExpression | None:
    return _parse_root(_random_tree(rng, 1, max_depth))


def _mutate(expression: FactorExpression, rng: np.random.Generator) -> FactorExpression | None:
    paths = _paths(expression._root)
    path, node = paths[int(rng.integers(0, len(paths)))]
    if rng.random() < 0.35:
        replacement = _random_tree(rng, 1, 3)
    elif node.op == "panel":
        choices = tuple(field for field in _FIELDS if field != node.args[0])
        replacement = _Node("panel", (str(rng.choice(choices)),))
    elif node.op == "constant":
        choices = tuple(value for value in _CONSTANTS if value != node.args[0])
        replacement = _Node("constant", (float(rng.choice(choices)),))
    elif node.op in _BINARY:
        replacement = _Node(str(rng.choice(tuple(op for op in _BINARY if op != node.op))), node.args)
    elif node.op in _UNARY:
        replacement = _Node(str(rng.choice(tuple(op for op in _UNARY if op != node.op))), node.args)
    elif node.op in _WINDOW:
        if rng.random() < 0.5:
            replacement = _Node(str(rng.choice(tuple(op for op in _WINDOW if op != node.op))), node.args)
        else:
            current_period = cast(int, node.args[1])
            period = int(rng.choice(tuple(value for value in _PERIODS if value != current_period)))
            replacement = _Node(node.op, (cast(_Node, node.args[0]), period))
    elif node.op == "ts_corr":
        current_period = cast(int, node.args[2])
        period = int(rng.choice(tuple(value for value in _PERIODS if value != current_period)))
        replacement = _Node(node.op, (cast(_Node, node.args[0]), cast(_Node, node.args[1]), period))
    else:
        replacement = _random_tree(rng, 1, 3)
    return _parse_root(_replace(expression._root, path, replacement))


def _crossover(
    left: FactorExpression, right: FactorExpression, rng: np.random.Generator
) -> FactorExpression | None:
    left_path, _ = _paths(left._root)[int(rng.integers(0, len(_paths(left._root))))]
    _, right_node = _paths(right._root)[int(rng.integers(0, len(_paths(right._root))))]
    return _parse_root(_replace(left._root, left_path, right_node))


def _seed_expressions() -> tuple[FactorExpression, ...]:
    sources = (
        "rank(close)",
        "rank(delta(close, 1))",
        "rank((close / delay(close, 1)) - 1)",
        "rank(delta(close, 5))",
        "rank(volume / ts_mean(volume, 5))",
        "rank((ts_mean(close, 5) / ts_mean(close, 20)) - 1)",
        "rank((high - low) / close)",
    )
    return tuple(parse_expression(source) for source in sources)


def _effective_observations(values: np.ndarray) -> float:
    count = len(values)
    if count < 3:
        return float(count)
    centered = values - float(values.mean())
    variance = float(np.dot(centered, centered))
    if variance <= 0:
        return float(count)
    bandwidth = min(20, max(1, int(4 * (count / 100) ** (2 / 9))))
    inflation = 1.0
    for lag in range(1, min(bandwidth, count - 1) + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / variance)
        weight = 1 - lag / (bandwidth + 1)
        inflation += 2 * weight * covariance
    inflation = max(1.0, inflation)
    return max(1.0, min(float(count), count / inflation))


def _t_statistic(daily_ic: tuple[float | None, ...]) -> tuple[int, float, float | None, float | None]:
    values = np.asarray([value for value in daily_ic if value is not None], dtype=np.float64)
    count = len(values)
    if count < 30:
        return count, float(count), None, None
    deviation = float(values.std(ddof=1))
    effective = _effective_observations(values)
    if not math.isfinite(deviation) or deviation <= 0:
        return count, effective, None, None
    icir = float(values.mean() / deviation)
    statistic = icir * math.sqrt(effective)
    return count, effective, icir, statistic if math.isfinite(statistic) else None


def _expected_max_t(trials: int) -> float:
    if trials <= 1:
        return 0.0
    first = _NORMAL.inv_cdf(1 - 1 / trials)
    second = _NORMAL.inv_cdf(1 - 1 / (trials * math.e))
    return (1 - _EULER_GAMMA) * first + _EULER_GAMMA * second


def _benjamini_hochberg(p_values: tuple[float, ...]) -> tuple[float, ...]:
    if not p_values:
        return ()
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    adjusted = [1.0] * len(p_values)
    running = 1.0
    for rank_index in range(len(order) - 1, -1, -1):
        position = rank_index + 1
        index = order[rank_index]
        running = min(running, p_values[index] * len(p_values) / position)
        adjusted[index] = min(1.0, running)
    return tuple(adjusted)


def _validate_library(data: _ValidatedInput, accepted: Mapping[str, ScorePanel]) -> None:
    if not isinstance(accepted, Mapping):
        raise ValueError("FACTOR_SEARCH_INPUT_INVALID")
    for name, scores in accepted.items():
        if (
            not name
            or not isinstance(scores, ScorePanel)
            or scores.dataset_identity != data.dataset_identity
            or scores.universe_identity != data.universe_identity
            or scores.sessions != data.context.sessions
            or scores.symbols != data.context.universe
            or scores.scores.shape != data.eligible.shape
            or scores.eligible.shape != data.eligible.shape
        ):
            raise ValueError("FACTOR_SEARCH_INPUT_INVALID")


def _score(
    expression: FactorExpression,
    data: _ValidatedInput,
    config: FormulaSearchConfig,
    accepted: Mapping[str, ScorePanel],
) -> _Scored:
    values = evaluate_expression(expression, data.context, eligible=data.eligible)
    panel = ScorePanel(
        data.dataset_identity,
        data.universe_identity,
        data.context.sessions,
        data.context.universe,
        values,
        data.eligible,
    )
    diagnostic = rank_ic(panel, data.labels, blocks=config.chronological_blocks)
    horizon = next(item for item in diagnostic.horizons if item.horizon == config.horizon)
    observations, effective, icir, statistic = _t_statistic(horizon.daily_ic)
    correlations = redundancy(panel, accepted) if accepted else ()
    maximum_correlation = max(
        (abs(item.mean_rank_correlation) for item in correlations if item.mean_rank_correlation is not None),
        default=0.0,
    )
    complexity = expression_complexity(expression)
    statistic_for_fitness = statistic if statistic is not None else -1_000_000.0
    preliminary = (
        statistic_for_fitness
        - config.complexity_penalty * max(0, complexity - 1)
        - config.redundancy_penalty * maximum_correlation
    )
    return _Scored(
        expression,
        complexity,
        observations,
        effective,
        horizon.mean_ic,
        icir,
        statistic,
        maximum_correlation,
        preliminary,
    )


def _tournament(population: list[_Scored], rng: np.random.Generator) -> _Scored:
    ranked = sorted(population, key=lambda item: (-item.preliminary_fitness, item.expression.identity))
    candidates = [
        ranked[int(index)] for index in rng.choice(len(ranked), size=min(3, len(ranked)), replace=False)
    ]
    return max(candidates, key=lambda item: (item.preliminary_fitness, item.expression.identity))


def _evolve(
    data: _ValidatedInput,
    config: FormulaSearchConfig,
    accepted: Mapping[str, ScorePanel],
) -> list[_Scored]:
    rng = np.random.Generator(np.random.PCG64(config.seed))
    seeds = _seed_expressions()
    next_seed = 0
    results: dict[str, _Scored] = {}
    attempts = 0
    while len(results) < config.budget and attempts < config.budget * 20:
        attempts += 1
        if next_seed < len(seeds):
            candidate = seeds[next_seed]
            next_seed += 1
        elif len(results) < config.population_size or rng.random() < 0.12:
            candidate = _random_expression(rng)
        else:
            population = sorted(
                results.values(),
                key=lambda item: (-item.preliminary_fitness, item.expression.identity),
            )[: config.population_size]
            parent = _tournament(population, rng)
            if rng.random() < 0.28:
                second = _tournament(population, rng)
                candidate = _crossover(parent.expression, second.expression, rng)
            else:
                candidate = _mutate(parent.expression, rng)
        if candidate is None or candidate.identity in results:
            continue
        results[candidate.identity] = _score(candidate, data, config, accepted)
    return list(results.values())


def search_expressions(
    data: FormulaTrainingInput,
    config: FormulaSearchConfig,
    *,
    accepted: Mapping[str, ScorePanel] | None = None,
) -> FormulaSearchReport:
    """Search only through the supplied training cutoff and label outcomes ending by it."""
    _validate_config(config)
    validated = _validate_input(data, config)
    library = {} if accepted is None else accepted
    _validate_library(validated, library)
    scored = _evolve(validated, config, library)

    total_tests = config.project_trials_used + len(scored)
    haircut = _expected_max_t(total_tests)
    current_p_values = tuple(
        _NORMAL.cdf(-(item.t_statistic - haircut)) if item.t_statistic is not None else 1.0 for item in scored
    )
    prior = (1.0,) * (config.family_trials_used - len(config.previous_family_p_values))
    combined_q_values = _benjamini_hochberg((*prior, *config.previous_family_p_values, *current_p_values))
    start = len(combined_q_values) - len(current_p_values)
    current_q_values = combined_q_values[start:]

    candidates = []
    for item, p_value, q_value in zip(scored, current_p_values, current_q_values, strict=True):
        deflated = item.t_statistic - haircut if item.t_statistic is not None else None
        fitness = (
            deflated
            - config.complexity_penalty * max(0, item.complexity - 1)
            - config.redundancy_penalty * item.max_abs_library_correlation
            if deflated is not None
            else -1_000_000.0
        )
        discovery = item.mean_ic is not None and item.mean_ic > 0 and q_value <= config.fdr_alpha
        candidates.append(
            FormulaCandidate(
                expression=item.expression.canonical,
                identity=item.expression.identity,
                complexity=item.complexity,
                observations=item.observations,
                effective_observations=round(item.effective_observations, 6),
                mean_ic=item.mean_ic,
                icir=item.icir,
                t_statistic=item.t_statistic,
                deflated_t_statistic=deflated,
                p_value=p_value,
                q_value=q_value,
                max_abs_library_correlation=item.max_abs_library_correlation,
                fitness=fitness,
                discovery=discovery,
            )
        )
    candidates.sort(key=lambda item: (-item.fitness, item.identity))
    discoveries = tuple(item.identity for item in candidates if item.discovery)
    return FormulaSearchReport(
        scope="synthetic",
        dataset_identity=validated.dataset_identity,
        universe_identity=validated.universe_identity,
        training_label_identity=validated.labels.label_identity,
        training_start=validated.context.sessions[0],
        training_cutoff=config.training_cutoff,
        horizon=config.horizon,
        seed=config.seed,
        budget=config.budget,
        family_trials_used_before=config.family_trials_used,
        project_trials_used_before=config.project_trials_used,
        trials_evaluated=len(candidates),
        family_trials_after_batch=config.family_trials_used + len(candidates),
        discoveries=discoveries,
        candidates=tuple(candidates),
        evidence_grade="none",
        trial_ledger_written=False,
        holdout_accessed=False,
    )


__all__ = [
    "FormulaCandidate",
    "FormulaSearchConfig",
    "FormulaSearchReport",
    "FormulaTrainingInput",
    "search_expressions",
]
