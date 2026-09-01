"""Immutable model for a linear PBO instance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

ComparisonOperator = Literal[">=", "=", "<="]


@dataclass(frozen=True)
class LinearExpression:
    """Integer linear expression over binary variables."""

    coefficients: Mapping[str, int] = field(default_factory=dict)
    constant: int = 0

    def __post_init__(self) -> None:
        normalized = {name: int(value) for name, value in self.coefficients.items() if value}
        object.__setattr__(self, "coefficients", normalized)

    @property
    def variables(self) -> frozenset[str]:
        return frozenset(self.coefficients)

    def evaluate(self, assignment: Mapping[str, int | bool]) -> int:
        missing = self.variables.difference(assignment)
        if missing:
            names = ", ".join(sorted(missing, key=_variable_sort_key))
            raise ValueError(f"Missing assignment values for: {names}")

        total = self.constant
        for name, coefficient in self.coefficients.items():
            value = assignment[name]
            if value not in (0, 1, False, True):
                raise ValueError(f"{name} must be binary; received {value!r}")
            total += coefficient * int(value)
        return total


@dataclass(frozen=True)
class Constraint:
    """Linear pseudo-Boolean constraint."""

    expression: LinearExpression
    operator: ComparisonOperator
    rhs: int

    def __post_init__(self) -> None:
        if self.operator not in {">=", "=", "<="}:
            raise ValueError(f"Unsupported operator: {self.operator}")

    def is_satisfied(self, assignment: Mapping[str, int | bool]) -> bool:
        lhs = self.expression.evaluate(assignment)
        if self.operator == ">=":
            return lhs >= self.rhs
        if self.operator == "<=":
            return lhs <= self.rhs
        return lhs == self.rhs


@dataclass(frozen=True)
class PBOInstance:
    """Linear PBO/PBS instance with optional comments."""

    constraints: tuple[Constraint, ...]
    objective: LinearExpression | None = None
    comments: tuple[str, ...] = ()

    @property
    def variables(self) -> frozenset[str]:
        variables: set[str] = set()
        if self.objective is not None:
            variables.update(self.objective.variables)
        for constraint in self.constraints:
            variables.update(constraint.expression.variables)
        return frozenset(variables)

    def is_feasible(self, assignment: Mapping[str, int | bool]) -> bool:
        return all(constraint.is_satisfied(assignment) for constraint in self.constraints)

    def objective_value(self, assignment: Mapping[str, int | bool]) -> int | None:
        if self.objective is None:
            return None
        return self.objective.evaluate(assignment)


def variable_sort_key(name: str) -> tuple[int, str]:
    """Sort x2 before x10 and place non-standard names at the end."""

    return _variable_sort_key(name)


def _variable_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("x") and name[1:].isdigit():
        return (int(name[1:]), "")
    return (2**32, name)
