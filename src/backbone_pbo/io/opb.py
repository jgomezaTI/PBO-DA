"""Parser and writer for the restricted linear OPB subset."""

from __future__ import annotations

import re
from pathlib import Path

from backbone_pbo.model import Constraint, LinearExpression, PBOInstance, variable_sort_key

_TERM_RE = re.compile(r"(?P<coefficient>[+-]?\d+)\s+(?P<variable>x[1-9]\d*)")
_CONSTRAINT_RE = re.compile(r"^(?P<lhs>.+?)\s*(?P<operator>>=|<=|=)\s*(?P<rhs>[+-]?\d+)\s*$")
_HEADER_RE = re.compile(r"#variable=\s*(\d+).*#constraint=\s*(\d+)")
_OFFSET_RE = re.compile(r"backbone-pbo\s+objective-offset:\s*([+-]?\d+)", re.IGNORECASE)


class OPBFormatError(ValueError):
    """The file is outside the supported restricted linear OPB subset."""


def loads_opb(content: str, *, validate_header: bool = True) -> PBOInstance:
    """Read restricted linear OPB text.

    For convenience, ``<=`` and statements spanning multiple lines are also
    accepted. The writer always normalizes ``<=`` to ``>=``.
    """

    comments: list[str] = []
    source_lines: list[str] = []
    objective_offset = 0

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("*"):
            comment = stripped[1:].strip()
            comments.append(comment)
            if offset_match := _OFFSET_RE.search(comment):
                objective_offset = int(offset_match.group(1))
            continue
        source_lines.append(stripped)

    body = " ".join(source_lines)
    parts = body.split(";")
    if parts[-1].strip():
        raise OPBFormatError("Cada sentencia OPB debe terminar en ';'")
    statements = [part.strip() for part in parts[:-1] if part.strip()]
    if not statements:
        raise OPBFormatError("The instance contains neither an objective nor constraints")

    objective: LinearExpression | None = None
    if statements[0].startswith("min:"):
        objective = _parse_expression(statements.pop(0)[len("min:") :])
        objective = LinearExpression(objective.coefficients, constant=objective_offset)
    elif objective_offset:
        raise OPBFormatError(
            "An objective offset was declared for an instance without an objective"
        )

    constraints = tuple(_parse_constraint(statement) for statement in statements)
    instance = PBOInstance(
        objective=objective,
        constraints=constraints,
        comments=tuple(comments),
    )
    if validate_header:
        _validate_declared_counts(instance, comments)
    return instance


def dumps_opb(instance: PBOInstance) -> str:
    """Write a canonical representation compatible with restricted linear OPB."""

    variables = sorted(instance.variables, key=variable_sort_key)
    _validate_restricted_variable_sequence(variables)

    equal_count = sum(constraint.operator == "=" for constraint in instance.constraints)
    intsize = _required_intsize(instance)
    lines = [
        f"* #variable= {len(variables)} #constraint= {len(instance.constraints)} "
        f"#equal= {equal_count} intsize= {intsize}"
    ]

    if instance.objective is not None and instance.objective.constant:
        lines.append(f"* backbone-pbo objective-offset: {instance.objective.constant:+d}")

    for comment in instance.comments:
        if _HEADER_RE.search(comment) or _OFFSET_RE.search(comment):
            continue
        lines.append(f"* {comment}" if comment else "*")

    if instance.objective is not None:
        lines.append(f"min: {_format_expression(instance.objective)} ;")

    for constraint in instance.constraints:
        expression = constraint.expression
        rhs = constraint.rhs - expression.constant
        operator = constraint.operator
        coefficients = dict(expression.coefficients)
        if operator == "<=":
            coefficients = {name: -value for name, value in coefficients.items()}
            rhs = -rhs
            operator = ">="
        lines.append(f"{_format_expression(LinearExpression(coefficients))} {operator} {rhs:+d} ;")

    return "\n".join(lines) + "\n"


def read_opb(path: str | Path, *, validate_header: bool = True) -> PBOInstance:
    return loads_opb(Path(path).read_text(encoding="ascii"), validate_header=validate_header)


def write_opb(instance: PBOInstance, path: str | Path) -> None:
    Path(path).write_text(dumps_opb(instance), encoding="ascii", newline="\n")


def _parse_constraint(statement: str) -> Constraint:
    match = _CONSTRAINT_RE.fullmatch(statement)
    if match is None:
        raise OPBFormatError(f"Unsupported constraint: {statement!r}")
    return Constraint(
        expression=_parse_expression(match.group("lhs")),
        operator=match.group("operator"),  # type: ignore[arg-type]
        rhs=int(match.group("rhs")),
    )


def _parse_expression(source: str) -> LinearExpression:
    coefficients: dict[str, int] = {}
    position = 0
    matches = list(_TERM_RE.finditer(source))
    if not matches:
        raise OPBFormatError(f"Empty or unsupported linear expression: {source!r}")

    for match in matches:
        if source[position : match.start()].strip():
            fragment = source[position : match.start()].strip()
            raise OPBFormatError(f"Unsupported expression fragment: {fragment!r}")
        coefficient = int(match.group("coefficient"))
        variable = match.group("variable")
        coefficients[variable] = coefficients.get(variable, 0) + coefficient
        position = match.end()

    if source[position:].strip():
        raise OPBFormatError(f"Unsupported fragment: {source[position:].strip()!r}")
    return LinearExpression(coefficients)


def _format_expression(expression: LinearExpression) -> str:
    if not expression.coefficients:
        raise OPBFormatError("Restricted OPB does not allow an empty linear sum")
    terms = []
    for variable in sorted(expression.coefficients, key=variable_sort_key):
        coefficient = expression.coefficients[variable]
        terms.append(f"{coefficient:+d} {variable}")
    return " ".join(terms)


def _validate_declared_counts(instance: PBOInstance, comments: list[str]) -> None:
    header = next(
        (_HEADER_RE.search(comment) for comment in comments if _HEADER_RE.search(comment)), None
    )
    if header is None:
        return
    declared_variables, declared_constraints = map(int, header.groups())
    if declared_variables != len(instance.variables):
        raise OPBFormatError(
            f"The header declares {declared_variables} variables, but "
            f"{len(instance.variables)} were found"
        )
    if declared_constraints != len(instance.constraints):
        raise OPBFormatError(
            f"The header declares {declared_constraints} constraints, but "
            f"{len(instance.constraints)} were found"
        )


def _validate_restricted_variable_sequence(variables: list[str]) -> None:
    expected = [f"x{index}" for index in range(1, len(variables) + 1)]
    if variables != expected:
        raise OPBFormatError(
            "Restricted OPB requires contiguous variables x1..xN; "
            f"found {', '.join(variables) or 'none'}"
        )


def _required_intsize(instance: PBOInstance) -> int:
    magnitudes = []
    if instance.objective is not None:
        magnitudes.append(sum(abs(value) for value in instance.objective.coefficients.values()))
    for constraint in instance.constraints:
        magnitudes.append(
            abs(constraint.rhs - constraint.expression.constant)
            + sum(abs(value) for value in constraint.expression.coefficients.values())
        )
    return max(1, *(value.bit_length() for value in magnitudes))
