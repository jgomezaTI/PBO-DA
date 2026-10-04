"""Exact PBO solver and backbone extractor with opportunistic solution caching."""

from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, milp

from backbone_pbo.backbone import BackboneLabel
from backbone_pbo.model import PBOInstance, variable_sort_key


def solve_pbo(instance: PBOInstance) -> tuple[float | None, dict[str, int] | None]:
    """Solve a linear PBO instance and return optimal objective value and an optimal assignment."""
    if not instance.variables:
        return 0.0, {}

    variables = sorted(instance.variables, key=variable_sort_key)
    var_idx = {name: i for i, name in enumerate(variables)}
    n = len(variables)

    c = np.zeros(n, dtype=float)
    if instance.objective is not None:
        for name, coef in instance.objective.coefficients.items():
            c[var_idx[name]] = coef

    constraints_list = []
    for constraint in instance.constraints:
        row = np.zeros(n, dtype=float)
        for name, coef in constraint.expression.coefficients.items():
            row[var_idx[name]] = coef

        rhs = float(constraint.rhs - constraint.expression.constant)
        if constraint.operator == ">=":
            constraints_list.append((row, rhs, np.inf))
        elif constraint.operator == "<=":
            constraints_list.append((row, -np.inf, rhs))
        elif constraint.operator == "=":
            constraints_list.append((row, rhs, rhs))

    if constraints_list:
        A = np.vstack([item[0] for item in constraints_list])
        lhs_min = np.array([item[1] for item in constraints_list])
        lhs_max = np.array([item[2] for item in constraints_list])
        lin_constraints = LinearConstraint(A, lhs_min, lhs_max)
    else:
        lin_constraints = None

    integrality = np.ones(n)
    bounds = (np.zeros(n), np.ones(n))

    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=lin_constraints,
    )

    if not res.success or res.x is None:
        return None, None

    assignment = {name: round(res.x[i]) for i, name in enumerate(variables)}
    obj_val = float(res.fun) + (instance.objective.constant if instance.objective else 0.0)
    return obj_val, assignment


def extract_backbone(instance: PBOInstance) -> dict[str, BackboneLabel]:
    """Extract exact backbone labels with opportunistic solution caching (GuroBack-style)."""
    opt_val, opt_assign = solve_pbo(instance)
    if opt_val is None or opt_assign is None:
        raise ValueError("Cannot extract backbone from an infeasible PBO instance")

    variables = sorted(instance.variables, key=variable_sort_key)
    var_idx = {name: i for i, name in enumerate(variables)}
    n = len(variables)

    c = np.zeros(n, dtype=float)
    if instance.objective is not None:
        for name, coef in instance.objective.coefficients.items():
            c[var_idx[name]] = coef

    base_constraints = []
    for constraint in instance.constraints:
        row = np.zeros(n, dtype=float)
        for name, coef in constraint.expression.coefficients.items():
            row[var_idx[name]] = coef
        rhs = float(constraint.rhs - constraint.expression.constant)
        if constraint.operator == ">=":
            base_constraints.append((row, rhs, np.inf))
        elif constraint.operator == "<=":
            base_constraints.append((row, -np.inf, rhs))
        elif constraint.operator == "=":
            base_constraints.append((row, rhs, rhs))

    clean_opt_val = opt_val - (instance.objective.constant if instance.objective else 0.0)
    obj_row = c.copy()
    base_constraints.append((obj_row, -np.inf, clean_opt_val + 1e-4))

    A = np.vstack([item[0] for item in base_constraints])
    lhs_min = np.array([item[1] for item in base_constraints])
    lhs_max = np.array([item[2] for item in base_constraints])
    lin_constraints = LinearConstraint(A, lhs_min, lhs_max)
    integrality = np.ones(n)

    can_be_0 = [False] * n
    can_be_1 = [False] * n

    def update_from_solution(x_arr: np.ndarray) -> None:
        for i in range(n):
            val = round(x_arr[i])
            if val == 0:
                can_be_0[i] = True
            elif val == 1:
                can_be_1[i] = True

    # Seed with initial optimal solution
    init_x = np.array([opt_assign[name] for name in variables], dtype=float)
    update_from_solution(init_x)

    for i in range(n):
        # If both values are already proven possible, it's NB -> skip!
        if can_be_0[i] and can_be_1[i]:
            continue

        target_val = 0 if not can_be_0[i] else 1
        lb = np.zeros(n)
        ub = np.ones(n)
        lb[i] = target_val
        ub[i] = target_val

        res = milp(c=c, integrality=integrality, bounds=(lb, ub), constraints=lin_constraints)
        if (
            res.success
            and res.fun is not None
            and res.fun <= clean_opt_val + 1e-4
            and res.x is not None
        ):
            update_from_solution(res.x)

    labels = {}
    for i, name in enumerate(variables):
        if can_be_0[i] and not can_be_1[i]:
            labels[name] = BackboneLabel.B0
        elif can_be_1[i] and not can_be_0[i]:
            labels[name] = BackboneLabel.B1
        else:
            labels[name] = BackboneLabel.NB

    return labels
