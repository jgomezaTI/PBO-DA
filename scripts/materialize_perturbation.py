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

import numpy as np
from scipy.optimize import LinearConstraint, milp

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


def exact_backbone_lp(path: Path) -> dict[str, str] | None:
    """Compute exact optimum backbone labels for LP/MPS using SCIP feasibility proofs."""
    try:
        from pyscipopt import Model
    except ImportError as exc:
        raise RuntimeError("LP augmentation requires PySCIPOpt") from exc
    model = Model()
    model.hideOutput(True)
    model.setParam("limits/gap", 0.0)
    model.setParam("limits/absgap", 0.0)
    model.setParam("numerics/feastol", 1e-9)
    model.readProblem(str(path))
    variables = list(model.getVars())
    if any(variable.vtype() != "BINARY" for variable in variables):
        model.freeProb()
        raise ValueError("Exact LP backbone currently requires binary variables")
    objective = model.getObjective()
    sense = model.getObjectiveSense()
    model.optimize()
    if model.getStatus() != "optimal":
        model.freeProb()
        return None
    optimum = model.getObjVal()
    tolerance = 1e-9
    model.freeTransform()
    if sense == "minimize":
        model.addCons(objective <= optimum + tolerance, name="backbone_objective_cutoff")
    else:
        model.addCons(objective >= optimum - tolerance, name="backbone_objective_cutoff")
    model.setObjective(0.0, "minimize")
    labels: dict[str, str] = {}
    try:
        for variable in variables:
            possible: set[int] = set()
            for value in (0, 1):
                model.freeTransform()
                fixed = model.addCons(variable == value, name=f"backbone_fix_{variable.name}_{value}")
                model.optimize()
                status = model.getStatus()
                if status in {"optimal", "solutions found"}:
                    possible.add(value)
                elif status != "infeasible":
                    return None
                model.freeTransform()
                model.delCons(fixed)
            if not possible:
                return None
            labels[variable.name] = "NB" if len(possible) == 2 else ("B1" if possible == {1} else "B0")
        return labels
    finally:
        model.freeProb()


def extract_exact_backbone(instance: PBOInstance) -> dict[str, str] | None:
    """Return exact optimum labels; status 2 is treated as an unproven timeout."""
    variables = sorted(instance.variables, key=variable_sort_key)
    if not variables:
        return {}
    indices = {name: i for i, name in enumerate(variables)}
    c = np.zeros(len(variables))
    if instance.objective:
        for name, coefficient in instance.objective.coefficients.items():
            c[indices[name]] = coefficient
    rows, lower, upper = [], [], []
    for constraint in instance.constraints:
        row = np.zeros(len(variables))
        for name, coefficient in constraint.expression.coefficients.items():
            row[indices[name]] = coefficient
        rhs = constraint.rhs - constraint.expression.constant
        rows.append(row)
        if constraint.operator == ">=":
            lower.append(rhs); upper.append(np.inf)
        elif constraint.operator == "<=":
            lower.append(-np.inf); upper.append(rhs)
        else:
            lower.append(rhs); upper.append(rhs)
    matrix = np.vstack(rows) if rows else np.empty((0, len(variables)))
    base = [LinearConstraint(matrix, lower, upper)] if rows else None
    kwargs = {"c": c, "integrality": np.ones(len(variables)),
              "bounds": (np.zeros(len(variables)), np.ones(len(variables))),
              "constraints": base}
    optimum = milp(**kwargs)
    if optimum.status != 0 or optimum.fun is None or optimum.x is None:
        return None
    optimum_value = float(optimum.fun)
    if instance.objective:
        optimum_value += instance.objective.constant
    labels: dict[str, str] = {}
    tolerance = 1e-7
    for i, name in enumerate(variables):
        possible = set()
        for value in (0, 1):
            lo = np.zeros(len(variables)); hi = np.ones(len(variables))
            lo[i] = hi[i] = value
            objective_constraint = LinearConstraint(c.reshape(1, -1), -np.inf,
                optimum_value - (instance.objective.constant if instance.objective else 0) + tolerance)
            constraints = [*([base[0]] if base else []), objective_constraint]
            result = milp(c=np.zeros(len(variables)), integrality=np.ones(len(variables)),
                          bounds=(lo, hi), constraints=constraints)
            if result.status == 0 and result.x is not None:
                possible.add(value)
            elif result.status not in (2,):
                return None
        labels[name] = "NB" if len(possible) == 2 else ("B1" if possible == {1} else "B0")
    return labels


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
                    from pyscipopt import Model
                except ImportError as exc:
                    raise RuntimeError("MPS augmentation requires PySCIPOpt") from exc
                normalized_path = output_instance / f".normalized_{original_name}.lp"
                normalizer = Model()
                normalizer.hideOutput(True)
                try:
                    normalizer.readProblem(str(instance_path))
                    normalizer.writeProblem(str(normalized_path), trans=False)
                    lp_source_text = normalized_path.read_text(encoding="utf-8")
                finally:
                    normalizer.freeProb()
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
                            transformed, _ = perturb_lp_text(lp_source_text, kind, rng)
                            candidate_path = output_instance / f".candidate_{replica}_{original_name}.lp"
                            try:
                                candidate_path.write_text(transformed, encoding="ascii")
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
              "label_solver": "exact transformation for selective polarity; PySCIPOpt for LP/MPS; scipy.optimize.milp for OPB",
              "perturbation": "single RHS +/-1; objective coefficient +/-1"}
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
