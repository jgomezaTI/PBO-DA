"""Build deterministic solver-validated constraint/objective/synthetic datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path

from backbone_pbo.augmentation.polarity import invert_polarity
from backbone_pbo.io.lp import invert_lp
from backbone_pbo.backbone import BackboneLabel, loads_backbone
from backbone_pbo.dataset.audit import inspect_instance
from backbone_pbo.io.opb import dumps_opb, loads_opb
from backbone_pbo.model import Constraint, LinearExpression, PBOInstance, variable_sort_key


def perturb_constraint(instance: PBOInstance, rng: random.Random) -> PBOInstance:
    if not instance.constraints:
        raise ValueError("constraint perturbation requires a constraint")
    i = rng.randrange(len(instance.constraints))
    old = instance.constraints[i]
    constraints = list(instance.constraints)
    constraints[i] = Constraint(old.expression, old.operator, old.rhs + rng.choice((-1, 1)))
    return PBOInstance(tuple(constraints), instance.objective, instance.comments)


def perturb_objective(instance: PBOInstance, rng: random.Random) -> PBOInstance:
    objective = instance.objective
    if objective is None or not objective.coefficients:
        raise ValueError("objective perturbation requires a nonempty objective")
    name = rng.choice(sorted(objective.coefficients))
    coefficients = dict(objective.coefficients)
    coefficients[name] += rng.choice((-1, 1))
    if coefficients[name] == 0:
        del coefficients[name]
    if not coefficients:
        raise ValueError("objective perturbation cannot remove every objective term")
    return PBOInstance(instance.constraints, LinearExpression(coefficients, objective.constant), instance.comments)


def perturb_synthetic(instance: PBOInstance, rng: random.Random) -> PBOInstance:
    """Make a nearby synthetic candidate by perturbing constraints and objective."""
    return perturb_objective(perturb_constraint(instance, rng), rng)


def perturb_selective_polarity(instance: PBOInstance, rng: random.Random) -> PBOInstance:
    variables = sorted(instance.variables, key=variable_sort_key)
    if len(variables) < 2:
        raise ValueError("selective polarity requires at least two variables")
    count = max(1, len(variables) // 2)
    return invert_polarity(instance, rng.sample(variables, count))


_LP_TERM = re.compile(r"(?P<sign>[+-])(?P<coef>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(?P<var>[A-Za-z_]\w*)")
_LP_RELATION = re.compile(r"^(?P<prefix>\s*[^:]+:\s*)(?P<lhs>.*?)(?P<op>>=|<=|=)\s*(?P<rhs>[+-]?\d+(?:\.\d+)?)\s*$")
_MIS_EDGE = re.compile(r"^(?P<prefix>\s*[^:]+:\s*)\+\s*x(?P<u>\d+)\s+\+\s*x(?P<v>\d+)\s*<=\s*1\s*$", re.IGNORECASE)


def perturb_lp_text(content: str, kind: str, rng: random.Random) -> tuple[str, set[str]]:
    """Perturb one MVC LP constraint/objective while preserving SCIP LP syntax."""
    lines = content.splitlines()
    subject = next((i for i, line in enumerate(lines) if line.strip().lower() == "subject to"), None)
    objective = next((i for i, line in enumerate(lines) if line.strip().lower() in {"minimize", "maximize"}), None)
    if subject is None or objective is None:
        raise ValueError("Expected a linear LP file with objective and Subject to sections")

    selected: set[str] = set()
    if kind in {"constraint", "synthetic"}:
        relations = [(i, _LP_RELATION.match(line)) for i, line in enumerate(lines[subject + 1:], subject + 1)]
        relations = [(i, match) for i, match in relations if match]
        if not relations:
            raise ValueError("No single-line constraints found in LP")
        index, match = rng.choice(relations)
        rhs = float(match.group("rhs")) + rng.choice((-1, 1))
        lines[index] = f"{match.group('prefix')}{match.group('lhs')}{match.group('op')} {rhs:.15g}"
    if kind in {"objective", "synthetic"}:
        objective_end = next((i for i in range(objective + 1, len(lines))
                              if lines[i].strip().lower() in {"subject to", "subjectto", "such that", "st"}), len(lines))
        terms = [(i, match) for i in range(objective + 1, objective_end)
                 for match in _LP_TERM.finditer(lines[i]) if match.group("var") not in selected]
        if not terms:
            raise ValueError("No objective terms found in LP")
        index, match = rng.choice(terms)
        old = float(match.group("coef")) * (1 if match.group("sign") == "+" else -1)
        new = old + rng.choice((-1, 1))
        if new == 0:
            raise ValueError("Objective perturbation removed a term")
        replacement = f"{'+' if new > 0 else '-'}{abs(new):.15g} {match.group('var')}"
        lines[index] = lines[index][:match.start()] + replacement + lines[index][match.end():]
    return "\n".join(lines) + "\n", selected


def perturb_mis_edge_text(content: str, rng: random.Random) -> tuple[str, dict[str, object]]:
    """Add or remove exactly one MIS edge constraint deterministically."""
    lines = content.splitlines()
    edges: dict[tuple[int, int], int] = {}
    variables = {int(m.group("var")) for line in lines for m in
                 re.finditer(r"\+\s*1\s+x(?P<var>\d+)", line, re.IGNORECASE)}
    for index, line in enumerate(lines):
        match = _MIS_EDGE.match(line)
        if not match:
            continue
        edge = tuple(sorted((int(match.group("u")), int(match.group("v")))))
        if edge[0] != edge[1]:
            edges[edge] = index
            variables.update(edge)
    if not edges or len(variables) < 2:
        raise ValueError("MIS LP does not contain recognizable edge constraints")
    edge_set = set(edges)
    nonedges = [(u, v) for u in sorted(variables) for v in sorted(variables)
                if u < v and (u, v) not in edge_set]
    add = bool(nonedges) and (not edges or rng.random() < 0.5)
    if add:
        u, v = rng.choice(nonedges)
        insertion = next((i for i, line in enumerate(lines)
                          if line.strip().lower() == "binary"), len(lines))
        lines.insert(insertion, f"C_aug_edge_{u}_{v}: + x{u} + x{v} <= 1")
        operation = "add_edge"
    else:
        u, v = rng.choice(sorted(edge_set))
        del lines[edges[(u, v)]]
        operation = "remove_edge"
    return "\n".join(lines) + "\n", {
        "operation": operation, "edge": [u, v],
        "original_edge_count": len(edges),
        "result_edge_count": len(edges) + (1 if add else -1),
    }


def exact_backbone_lp(path: Path) -> dict[str, str] | None:
    """Compute exact optimum backbone labels for LP/MPS using Gurobi."""
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise RuntimeError("LP/MPS augmentation requires gurobipy and a valid license") from exc
    import os

    env = gp.Env(empty=True)
    model = None
    try:
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        for key, name in (("LicenseID", "PBO_GUROBI_LICENSEID"),
                          ("WLSAccessID", "PBO_GUROBI_WLSACCESSID"),
                          ("WLSSecret", "PBO_GUROBI_WLSSECRET")):
            if os.environ.get(name):
                value = int(os.environ[name]) if key == "LicenseID" else os.environ[name]
                env.setParam(key, value)
        env.start()
        model = gp.read(str(path), env=env)
        model.Params.MIPGap = 0.0
        model.Params.MIPGapAbs = 0.0
        model.Params.FeasibilityTol = 1e-9
        model.Params.DualReductions = 0
        model.Params.Threads = int(os.environ.get("PBO_GUROBI_THREADS", "4"))
        model.Params.MIPFocus = 2
        # Optional bounded preflight/production runs (unset preserves exact
        # unbounded solving). A timeout is treated as an inconclusive candidate
        # and is rejected by the materializer rather than being mislabeled.
        if os.environ.get("PBO_GUROBI_TIMELIMIT"):
            model.Params.TimeLimit = float(os.environ["PBO_GUROBI_TIMELIMIT"])
        # Detailed solver progress is written beside the run logs in Drive.
        log_dir = Path(os.environ["PBO_GUROBI_LOG_DIR"]) if os.environ.get("PBO_GUROBI_LOG_DIR") else path.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        model.Params.LogFile = str(log_dir / f"{path.stem}.gurobi.log")
        model.Params.DisplayInterval = 30
        variables = list(model.getVars())
        if any(variable.VType != gp.GRB.BINARY for variable in variables):
            raise ValueError("Exact LP backbone currently requires binary variables")
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            return None
        if not variables:
            return {}

        optimum = model.ObjVal
        objective = model.getObjective()
        sense = model.ModelSense
        tolerance = 1e-8 * max(1.0, abs(optimum))
        if sense == gp.GRB.MINIMIZE:
            model.addConstr(objective <= optimum + tolerance, name="backbone_objective_cutoff")
        else:
            model.addConstr(objective >= optimum - tolerance, name="backbone_objective_cutoff")

        reference = {v.VarName: int(v.X > 0.5) for v in variables}
        unresolved = {v.VarName: v for v in variables}
        labels: dict[str, str] = {}
        while unresolved:
            distance = gp.quicksum(
                variable if reference[name] == 0 else 1 - variable
                for name, variable in unresolved.items()
            )
            model.setObjective(distance, gp.GRB.MAXIMIZE)
            model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                return None
            changed = [name for name, variable in unresolved.items()
                       if int(variable.X > 0.5) != reference[name]]
            if not changed:
                labels.update({name: ("B1" if reference[name] else "B0")
                               for name in unresolved})
                break
            for name in changed:
                labels[name] = "NB"
                del unresolved[name]
        return labels
    except gp.GurobiError as exc:
        raise RuntimeError(f"Gurobi could not compute exact backbone for {path}") from exc
    finally:
        if model is not None:
            model.dispose()
        env.dispose()


def exact_mis_edge_preserves_backbone(candidate_path: Path, original_backbone_path: Path,
                                      edge: tuple[int, int], operation: str,
                                      original_path: Path) -> dict[str, str] | None:
    """Certify an MIS edge edit without recomputing every backbone variable.

    The original backbone is valid for the transformed instance only when the
    edit preserves the optimum value and every optimum satisfies the edited
    edge inequality.  Both conditions are checked exactly with Gurobi; the
    accepted labels are then copied from the source backbone.
    """
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise RuntimeError("MIS augmentation requires gurobipy and a valid license") from exc
    env = gp.Env(empty=True)
    models = []
    try:
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        for key, name in (("LicenseID", "PBO_GUROBI_LICENSEID"),
                          ("WLSAccessID", "PBO_GUROBI_WLSACCESSID"),
                          ("WLSSecret", "PBO_GUROBI_WLSSECRET")):
            if os.environ.get(name):
                value = int(os.environ[name]) if key == "LicenseID" else os.environ[name]
                env.setParam(key, value)
        env.start()

        def load(path: Path):
            model = gp.read(str(path), env=env)
            model.Params.MIPGap = 0.0
            model.Params.MIPGapAbs = 0.0
            model.Params.DualReductions = 0
            model.Params.Threads = 4
            # Keep MIS certification bounded in pilot/production runs too.
            # Without this, the specialized exact checker could run forever
            # even when PBO_GUROBI_TIMELIMIT was set for the generic checker.
            if os.environ.get("PBO_GUROBI_TIMELIMIT"):
                model.Params.TimeLimit = float(os.environ["PBO_GUROBI_TIMELIMIT"])
            models.append(model)
            return model

        original = load(original_path)
        original.optimize()
        if original.Status != gp.GRB.OPTIMAL:
            return None
        optimum = original.ObjVal

        candidate = load(candidate_path)
        candidate.optimize()
        if candidate.Status != gp.GRB.OPTIMAL or abs(candidate.ObjVal - optimum) > 1e-8:
            return None
        u, v = edge
        vars_by_name = {var.VarName: var for var in candidate.getVars()}
        if f"x{u}" not in vars_by_name or f"x{v}" not in vars_by_name:
            return None
        objective = candidate.getObjective()
        tolerance = 1e-8 * max(1.0, abs(optimum))
        if candidate.ModelSense == gp.GRB.MINIMIZE:
            candidate.addConstr(objective <= optimum + tolerance)
        else:
            candidate.addConstr(objective >= optimum - tolerance)
        edge_sum = vars_by_name[f"x{u}"] + vars_by_name[f"x{v}"]
        candidate.setObjective(edge_sum, gp.GRB.MAXIMIZE)
        candidate.optimize()
        if candidate.Status != gp.GRB.OPTIMAL or candidate.ObjVal > 1.0 + 1e-8:
            return None
        variables = [var.VarName for var in original.getVars()]
        labels = loads_backbone(original_backbone_path.read_text(encoding="ascii"), variables)
        return {name: ("NB" if label == BackboneLabel.NB else
                       "B1" if label == BackboneLabel.B1 else "B0")
                for name, label in labels.items()}
    except gp.GurobiError as exc:
        raise RuntimeError(f"Gurobi could not certify MIS edge edit for {candidate_path}") from exc
    finally:
        for model in models:
            model.dispose()
        env.dispose()


def extract_exact_backbone(instance: PBOInstance) -> dict[str, str] | None:
    """Return exact optimum labels with Gurobi; inconclusive solves return None."""
    variables = sorted(instance.variables, key=variable_sort_key)
    if not variables:
        return {}
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise RuntimeError("OPB augmentation requires gurobipy and a valid license") from exc

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model(env=env)
    try:
        model.Params.MIPGap = 0.0
        model.Params.MIPGapAbs = 0.0
        model.Params.FeasibilityTol = 1e-9
        model.Params.DualReductions = 0
        model.Params.Threads = 16
        model.Params.MIPFocus = 2
        decision = {name: model.addVar(vtype=gp.GRB.BINARY, name=name) for name in variables}
        model.update()

        for index, constraint in enumerate(instance.constraints):
            lhs = gp.quicksum(coef * decision[name]
                              for name, coef in constraint.expression.coefficients.items())
            lhs += constraint.expression.constant
            rhs = constraint.rhs
            if constraint.operator == ">=":
                model.addConstr(lhs >= rhs, name=f"c{index}")
            elif constraint.operator == "<=":
                model.addConstr(lhs <= rhs, name=f"c{index}")
            else:
                model.addConstr(lhs == rhs, name=f"c{index}")

        objective = instance.objective
        objective_expr = gp.quicksum(
            coef * decision[name] for name, coef in objective.coefficients.items()
        ) if objective else gp.LinExpr(0.0)
        if objective:
            objective_expr += objective.constant
            model.setObjective(objective_expr, gp.GRB.MINIMIZE)
        else:
            model.setObjective(0.0, gp.GRB.MINIMIZE)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            return None

        optimum = model.ObjVal
        sense = model.ModelSense
        tolerance = 1e-6
        labels: dict[str, str] = {}
        for name in variables:
            possible: set[int] = set()
            variable = decision[name]
            for value in (0, 1):
                old_lb, old_ub = variable.LB, variable.UB
                variable.LB = variable.UB = value
                model.update()
                model.optimize()
                if model.Status == gp.GRB.OPTIMAL:
                    if abs(model.ObjVal - optimum) <= tolerance:
                        possible.add(value)
                elif model.Status != gp.GRB.INFEASIBLE:
                    return None
                variable.LB, variable.UB = old_lb, old_ub
                model.update()
            if not possible:
                return None
            labels[name] = "NB" if len(possible) == 2 else ("B1" if possible == {1} else "B0")
        model.setObjective(objective_expr, sense)
        return labels
    finally:
        model.dispose()
        env.dispose()


def render_backbone(labels: dict[str, str]) -> str:
    lines = []
    for name in sorted(labels, key=variable_sort_key):
        label = labels[name]
        if label == "NB":
            lines.append(f"nb {name}")
        elif label == "B1":
            lines.append(f"b {name}")
        else:
            lines.append(f"b -{name}")
    return "\n".join([*lines, "b 0", ""])


def partitions_of(dataset: Path) -> dict[str, list[str]]:
    payload = json.loads((dataset / "partitions.json").read_text(encoding="utf-8"))
    partitions = payload.get("partitions", payload)
    if "validation" not in partitions and "valid" in partitions:
        partitions["validation"] = partitions.pop("valid")
    return {key: list(value) for key, value in partitions.items()}


def materialize(source: Path, output: Path, *, kind: str, seed: int,
                replicas: int, attempts: int) -> dict[str, int]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    partial = output.with_name(f".{output.name}.partial")
    recipe = {"source": str(source.resolve()), "kind": kind, "seed": seed,
              "replicas": replicas, "attempts": attempts}
    recipe_path = partial / "generation-recipe.json"
    if partial.exists():
        existing_recipe = None
        if recipe_path.is_file():
            try:
                existing_recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        if existing_recipe != recipe:
            shutil.rmtree(partial)
    partitions = partitions_of(source)
    output_instance = partial / "instance"
    output_backbone = partial / "backbone"
    partial.mkdir(parents=True, exist_ok=True)
    output_instance.mkdir(exist_ok=True)
    output_backbone.mkdir(exist_ok=True)
    recipe_tmp = recipe_path.with_suffix(".tmp")
    recipe_tmp.write_text(json.dumps(recipe, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(recipe_tmp, recipe_path)
    transforms = {"selective-polarity": perturb_selective_polarity,
                  "constraint": perturb_constraint, "objective": perturb_objective,
                  "synthetic": perturb_synthetic}
    transform = transforms[kind]
    generated, rejected = [], 0
    for split, names in partitions.items():
        for original_name in names:
            instance_path = source / "instance" / original_name
            backbone_path = source / "backbone" / f"{original_name}.backbone"
            if not instance_path.is_file() or not backbone_path.is_file():
                raise FileNotFoundError(f"Missing instance/backbone pair: {original_name}")
            original_text = instance_path.read_text(encoding="ascii")
            lp_source_text = original_text
            is_lp = instance_path.suffix.lower() in {".lp", ".mps"}
            if instance_path.suffix.lower() == ".mps":
                try:
                    import gurobipy as gp
                except ImportError as exc:
                    raise RuntimeError("MPS augmentation requires gurobipy and a valid license") from exc
                normalized_path = output_instance / f".normalized_{original_name}.lp"
                normalizer = gp.read(str(instance_path))
                try:
                    normalizer.Params.OutputFlag = 0
                    normalizer.write(str(normalized_path))
                    lp_source_text = normalized_path.read_text(encoding="utf-8")
                finally:
                    normalizer.dispose()
                    normalized_path.unlink(missing_ok=True)
            original = None if is_lp else loads_opb(original_text)
            (output_instance / original_name).write_text(original_text, encoding="ascii")
            (output_backbone / f"{original_name}.backbone").write_bytes(backbone_path.read_bytes())
            if split != "train":
                continue
            for replica in range(replicas):
                derived_name = (f"aug{replica}_{instance_path.stem}.lp"
                                if is_lp and instance_path.suffix.lower() == ".mps"
                                else f"aug{replica}_{original_name}")
                generated_instance = output_instance / derived_name
                generated_backbone = output_backbone / f"{derived_name}.backbone"
                if generated_instance.is_file() and generated_backbone.is_file():
                    generated.append(derived_name)
                    continue
                generated_instance.unlink(missing_ok=True)
                generated_backbone.unlink(missing_ok=True)
                rng_seed = int.from_bytes(
                    hashlib.sha256(f"{seed}:{original_name}:{replica}".encode("utf-8")).digest()[:8],
                    "big",
                )
                rng = random.Random(rng_seed)
                candidate = None
                for _ in range(attempts):
                    try:
                        if is_lp:
                            if kind == "selective-polarity":
                                shape = inspect_instance(instance_path)
                                variables = list(shape.variables)
                                if len(variables) < 2:
                                    raise ValueError("selective polarity requires at least two variables")
                                selected = set(rng.sample(variables, max(1, len(variables) // 2)))
                                transformed = invert_lp(lp_source_text, selected)
                                labels_raw = loads_backbone(backbone_path.read_text(encoding="ascii"), variables)
                                labels = {}
                                for name, label in labels_raw.items():
                                    if label == BackboneLabel.NB:
                                        labels[name] = "NB"
                                    elif name in selected:
                                        labels[name] = "B0" if label == BackboneLabel.B1 else "B1"
                                    else:
                                        labels[name] = "B1" if label == BackboneLabel.B1 else "B0"
                                candidate = (transformed, labels)
                                break
                            if kind == "constraint" and source.name.upper() == "MIS":
                                transformed, operation = perturb_mis_edge_text(lp_source_text, rng)
                            else:
                                transformed, _ = perturb_lp_text(lp_source_text, kind, rng)
                                operation = {"operation": "rhs_or_objective_perturbation"}
                            candidate_path = output_instance / f".candidate_{replica}_{original_name}.lp"
                            try:
                                candidate_path.write_text(transformed, encoding="ascii")
                                if kind == "constraint" and source.name.upper() == "MIS":
                                    labels = exact_mis_edge_preserves_backbone(
                                        candidate_path, backbone_path,
                                        tuple(operation["edge"]), operation["operation"],
                                        instance_path)
                                else:
                                    labels = exact_backbone_lp(candidate_path)
                            finally:
                                candidate_path.unlink(missing_ok=True)
                            if labels is None:
                                rejected += 1
                                continue
                            candidate = (transformed, labels)
                            break
                        assert original is not None
                        proposal = transform(original, rng)
                        if dumps_opb(proposal) == dumps_opb(original):
                            rejected += 1
                            continue
                        labels = extract_exact_backbone(proposal)
                        if labels is None:
                            rejected += 1
                            continue
                        candidate = proposal, labels
                        break
                    except (ValueError, RuntimeError, OverflowError):
                        rejected += 1
                if candidate is None:
                    continue
                proposal, labels = candidate
                instance_text = proposal if is_lp else dumps_opb(proposal)
                instance_tmp = output_instance / f".{derived_name}.writing"
                backbone_tmp = output_backbone / f".{derived_name}.backbone.writing"
                instance_tmp.write_text(instance_text, encoding="ascii")
                backbone_tmp.write_text(render_backbone(labels), encoding="ascii")
                os.replace(instance_tmp, generated_instance)
                os.replace(backbone_tmp, generated_backbone)
                generated.append(derived_name)
    output_partitions = {"train": partitions.get("train", []) + generated,
                         "validation": partitions.get("validation", []),
                         "test": partitions.get("test", [])}
    (partial / "partitions.json").write_text(json.dumps(output_partitions, indent=2) + "\n", encoding="utf-8")
    record = {"kind": kind, "seed": seed, "replicas_per_train_instance": replicas,
              "attempts_per_replica": attempts, "generated": len(generated), "rejected": rejected,
              "label_solver": ("exact transformation for selective polarity; MIS edge edits use "
                               "an exact optimum-preservation certificate; Gurobi backbone for "
                               "other OPB/LP/MPS edits"),
              "perturbation": ("MIS: add or remove exactly one edge constraint x_u+x_v<=1, "
                               "accepted only when the optimum and all optimum edge assignments "
                               "are preserved; other problems: single RHS +/-1; objective coefficient +/-1")}
    (partial / "augmentation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    partial.replace(output)
    return {"generated": len(generated), "rejected": rejected}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--kind", choices=("selective-polarity", "constraint", "objective", "synthetic"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=20)
    args = parser.parse_args()
    if args.replicas < 1 or args.attempts < 1:
        parser.error("--replicas and --attempts must be positive")
    result = materialize(args.source.resolve(), args.output.resolve(), kind=args.kind,
                         seed=args.seed, replicas=args.replicas, attempts=args.attempts)
    print(json.dumps({**result, "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
