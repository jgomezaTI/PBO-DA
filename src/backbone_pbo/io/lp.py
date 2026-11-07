"""Small reader/writer helpers for the linear LP files used by MVC-easy."""

from __future__ import annotations

import re
from collections.abc import Iterable

_TERM = re.compile(r"(?P<sign>[+-])(?P<coef>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(?P<var>[A-Za-z_]\w*)")
_REL = re.compile(r"^(?P<lhs>.*)\s(?P<op>>=|<=|=)\s(?P<rhs>[+-]?\d+(?:\.\d+)?)\s*$")


def _transform_expression(text: str, selected: set[str]) -> tuple[str, float]:
    """Flip selected binary variables and return expression plus constant shift."""
    parts: list[str] = []
    shift = 0.0
    position = 0
    for match in _TERM.finditer(text):
        parts.append(text[position : match.start()])
        coefficient = float(match.group("coef"))
        if match.group("var") in selected:
            shift += coefficient
            coefficient = -coefficient
        sign = "+" if coefficient >= 0 else "-"
        parts.append(f" {sign}{abs(coefficient):g} {match.group('var')}")
        position = match.end()
    parts.append(text[position:])
    return "".join(parts), shift


def invert_lp(content: str, variables: Iterable[str] | None = None) -> str:
    """Apply x=1-y to a SCIP/CPLEX linear LP file.

    The transformation preserves the LP syntax and binary bounds. Constraints
    receive the required RHS shift; the objective's constant shift is omitted,
    which is valid for optimization and does not affect the argmin.
    """
    lines = content.splitlines()
    names = set(variables or re.findall(r"\b(?:v|x)\d+\b", content))
    output: list[str] = []
    section = ""
    for raw in lines:
        stripped = raw.strip()
        if stripped in {"Minimize", "Maximize", "Subject To", "Bounds", "Binary", "Generals", "End"}:
            section = stripped
            output.append(raw)
            continue
        if section in {"Minimize", "Maximize"} and stripped and not stripped.startswith("\\"):
            transformed, _ = _transform_expression(raw, names)
            output.append(transformed)
            continue
        if section == "Subject To" and stripped and not stripped.startswith("\\"):
            match = re.match(r"(?P<prefix>\s*[^:]+:\s*)(?P<body>.*)$", raw)
            if match:
                relation = _REL.match(match.group("body"))
                if relation:
                    lhs, shift = _transform_expression(relation.group("lhs"), names)
                    rhs = float(relation.group("rhs")) - shift
                    output.append(f"{match.group('prefix')}{lhs} {relation.group('op')} {rhs:g}")
                    continue
        output.append(raw)
    return "\n".join(output) + "\n"
