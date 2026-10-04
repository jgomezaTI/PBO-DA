"""Graph representation converter for linear PBO instances."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch_geometric.data import Data

from backbone_pbo.backbone import BackboneLabel
from backbone_pbo.model import PBOInstance, variable_sort_key


def instance_to_bipartite_graph(
    instance: PBOInstance,
    labels: Mapping[str, BackboneLabel | int] | None = None,
    scale: str = "M",
    family: str = "pbo",
) -> Data:
    """Convert a linear PBOInstance into a PyG bipartite Data object."""
    sorted_vars = sorted(instance.variables, key=variable_sort_key)
    var_map = {name: idx for idx, name in enumerate(sorted_vars)}
    num_vars = len(sorted_vars)
    num_cons = len(instance.constraints)

    # 1. Variable features: [obj_coef_norm, degree, pos_coef_sum, neg_coef_sum]
    var_features = torch.zeros((num_vars, 4), dtype=torch.float)
    max_obj = 1.0
    if instance.objective is not None and instance.objective.coefficients:
        max_obj = max(1.0, max(abs(v) for v in instance.objective.coefficients.values()))
        for name, coef in instance.objective.coefficients.items():
            if name in var_map:
                var_features[var_map[name], 0] = float(coef) / max_obj

    # 2. Constraint features: [rhs_norm, is_ge, degree, sum_abs_coefs]
    con_features = torch.zeros((num_cons, 4), dtype=torch.float)
    max_rhs = 1.0
    if instance.constraints:
        max_rhs = max(1.0, max(abs(c.rhs - c.expression.constant) for c in instance.constraints))

    # 3. Edges & edge attributes
    edge_list_src = []
    edge_list_dst = []
    edge_attrs = []

    for con_idx, constraint in enumerate(instance.constraints):
        con_node_idx = num_vars + con_idx
        rhs = constraint.rhs - constraint.expression.constant
        is_ge = 1.0 if constraint.operator == ">=" else 0.0

        coefs = constraint.expression.coefficients
        con_features[con_idx, 0] = float(rhs) / max_rhs
        con_features[con_idx, 1] = is_ge
        con_features[con_idx, 2] = float(len(coefs))
        con_features[con_idx, 3] = float(sum(abs(v) for v in coefs.values()))

        for var_name, coef in coefs.items():
            if var_name not in var_map:
                continue
            v_idx = var_map[var_name]

            # Update variable features
            var_features[v_idx, 1] += 1.0
            if coef > 0:
                var_features[v_idx, 2] += float(coef)
            else:
                var_features[v_idx, 3] += float(abs(coef))

            # Edge v -> c
            edge_list_src.append(v_idx)
            edge_list_dst.append(con_node_idx)
            edge_attrs.append([float(coef), 1.0 if coef > 0 else -1.0])

            # Edge c -> v
            edge_list_src.append(con_node_idx)
            edge_list_dst.append(v_idx)
            edge_attrs.append([float(coef), 1.0 if coef > 0 else -1.0])

    if num_cons > 0:
        var_features[:, 1] /= float(num_cons)

    x = torch.cat([var_features, con_features], dim=0)

    if edge_list_src:
        edge_index = torch.tensor([edge_list_src, edge_list_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float)

    var_mask = torch.zeros(num_vars + num_cons, dtype=torch.bool)
    var_mask[:num_vars] = True

    if labels is not None:
        y = torch.full((num_vars + num_cons,), fill_value=-1, dtype=torch.long)
        for var_name, label in labels.items():
            if var_name in var_map:
                y[var_map[var_name]] = int(label)
    else:
        y = None

    # Map scale string to int for batching: S=0, M=1, L=2
    scale_id = 0 if scale == "S" else (1 if scale == "M" else 2)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        var_mask=var_mask,
        num_vars=num_vars,
        num_cons=num_cons,
        scale_id=scale_id,
    )
    return data
