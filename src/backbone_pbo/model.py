"""Modelo inmutable de una instancia PBO lineal."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

ComparisonOperator = Literal[">=", "=", "<="]


@dataclass(frozen=True)
class LinearExpression:
    """Expresión lineal entera sobre variables binarias."""

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
            raise ValueError(f"Faltan valores para: {names}")

        total = self.constant
        for name, coefficient in self.coefficients.items():
            value = assignment[name]
            if value not in (0, 1, False, True):
                raise ValueError(f"{name} debe ser binaria; se recibió {value!r}")
            total += coefficient * int(value)
        return total


@dataclass(frozen=True)
class Constraint:
    """Restricción pseudo-booleana lineal."""

    expression: LinearExpression
    operator: ComparisonOperator
    rhs: int

    def __post_init__(self) -> None:
        if self.operator not in {">=", "=", "<="}:
            raise ValueError(f"Operador no soportado: {self.operator}")

    def is_satisfied(self, assignment: Mapping[str, int | bool]) -> bool:
        lhs = self.expression.evaluate(assignment)
        if self.operator == ">=":
            return lhs >= self.rhs
        if self.operator == "<=":
            return lhs <= self.rhs
        return lhs == self.rhs


@dataclass(frozen=True)
class PBOInstance:
    """Instancia PBO/PBS lineal con comentarios opcionales."""

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
    """Ordena x2 antes de x10 y deja nombres no estándar al final."""

    return _variable_sort_key(name)


def _variable_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("x") and name[1:].isdigit():
        return (int(name[1:]), "")
    return (2**32, name)
