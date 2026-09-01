"""Polarity inversion through the substitution x = 1 - y."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from backbone_pbo.model import Constraint, LinearExpression, PBOInstance


def invert_polarity(
    instance: PBOInstance,
    variables: Iterable[str] | None = None,
) -> PBOInstance:
    """Invert an instance's polarity globally or selectively.

    If ``variables`` is ``None``, invert every present variable. In constraints,
    generated constants move to the right-hand side. The objective offset is
    preserved to compare objective values exactly.
    """

    selected = instance.variables if variables is None else frozenset(variables)
    unknown = selected.difference(instance.variables)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Variables missing from the instance: {names}")

    objective = None
    if instance.objective is not None:
        objective = _substitute(instance.objective, selected)

    constraints: list[Constraint] = []
    for constraint in instance.constraints:
        substituted = _substitute(constraint.expression, selected)
        constraints.append(
            Constraint(
                expression=LinearExpression(substituted.coefficients),
                operator=constraint.operator,
                rhs=constraint.rhs - substituted.constant,
            )
        )

    return PBOInstance(
        objective=objective,
        constraints=tuple(constraints),
        comments=instance.comments,
    )


def invert_assignment(
    assignment: Mapping[str, int | bool],
    variables: Iterable[str] | None = None,
) -> dict[str, int]:
    """Apply the same bijection used by augmentation to an assignment."""

    selected = set(assignment) if variables is None else set(variables)
    unknown = selected.difference(assignment)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Variables missing from the assignment: {names}")

    transformed: dict[str, int] = {}
    for name, raw_value in assignment.items():
        if raw_value not in (0, 1, False, True):
            raise ValueError(f"{name} must be binary; received {raw_value!r}")
        value = int(raw_value)
        transformed[name] = 1 - value if name in selected else value
    return transformed


def _substitute(expression: LinearExpression, selected: frozenset[str]) -> LinearExpression:
    coefficients = dict(expression.coefficients)
    constant = expression.constant
    for name in selected:
        coefficient = coefficients.get(name)
        if coefficient is None:
            continue
        constant += coefficient
        coefficients[name] = -coefficient
    return LinearExpression(coefficients=coefficients, constant=constant)
