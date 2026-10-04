"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from backbone_pbo.augmentation.polarity import invert_polarity
from backbone_pbo.dataset.audit import (
    DatasetAuditError,
    audit_dataset,
    write_audit_outputs,
)
from backbone_pbo.dataset.backpas import (
    BackPaSError,
    extract_backbones,
    prepare_backpas,
    train_backpas,
)
from backbone_pbo.dataset.dmiplib import DatasetFetchError, fetch_dmiplib_mvc
from backbone_pbo.io.opb import read_opb, write_opb
from backbone_pbo.model import variable_sort_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="backbone-pbo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="summarize an OPB instance")
    inspect_parser.add_argument("input", type=Path)
    inspect_parser.set_defaults(handler=_inspect)

    invert_parser = subparsers.add_parser("invert", help="invert polarities")
    invert_parser.add_argument("input", type=Path)
    invert_parser.add_argument("output", type=Path)
    invert_parser.add_argument(
        "--variables",
        nargs="+",
        help="variables to invert; if omitted, invert all variables",
    )
    invert_parser.add_argument(
        "--no-provenance",
        action="store_true",
        help="do not add a comment identifying the augmentation",
    )
    invert_parser.set_defaults(handler=_invert)

    fetch_parser = subparsers.add_parser(
        "fetch-dataset", help="fetch a pinned public benchmark distribution"
    )
    fetch_parser.add_argument("benchmark", choices=["MVC"])
    fetch_parser.add_argument("--output", type=Path, default=Path("data/backpas/MVC"))
    fetch_parser.add_argument("--revision", help="Hugging Face dataset revision")
    fetch_parser.add_argument("--force", action="store_true")
    fetch_parser.set_defaults(handler=_fetch_dataset)

    audit_parser = subparsers.add_parser(
        "audit-dataset", help="validate a BackPaS dataset and emit deterministic reports"
    )
    audit_parser.add_argument("dataset", type=Path)
    audit_parser.add_argument("--partitions", type=Path)
    audit_parser.add_argument(
        "--require-backbones",
        nargs="*",
        default=["train", "validation"],
        metavar="PARTITION",
    )
    audit_parser.add_argument(
        "--allow-missing-backbones",
        action="store_true",
        help="run the pre-extraction structural audit without requiring labels",
    )
    audit_parser.add_argument("--manifest", type=Path)
    audit_parser.add_argument("--report", type=Path)
    audit_parser.set_defaults(handler=_audit_dataset)

    extract_parser = subparsers.add_parser(
        "extract-backbones", help="extract labelled backbones with GuroBack under WSL"
    )
    extract_parser.add_argument("dataset", type=Path)
    extract_parser.add_argument(
        "--partitions", nargs="+", default=["train", "validation"], metavar="PARTITION"
    )
    extract_parser.add_argument("--threads", type=int, default=4)
    extract_parser.add_argument("--distribution", default="Ubuntu")
    extract_parser.add_argument("--binary", type=Path)
    extract_parser.add_argument(
        "--smoke", action="store_true", help="extract only 8 train and 2 validation instances"
    )
    extract_parser.set_defaults(handler=_extract_backbones)

    prepare_parser = subparsers.add_parser(
        "prepare-backpas", help="build upstream BackPaS graphs and fixed partitions"
    )
    prepare_parser.add_argument("dataset", type=Path)
    prepare_parser.add_argument("--work-dir", type=Path, required=True)
    prepare_parser.add_argument(
        "--graph-type", choices=["literals", "variables"], default="literals"
    )
    prepare_parser.add_argument("--distribution", default="Ubuntu")
    prepare_parser.add_argument(
        "--smoke", action="store_true", help="select only 8 train and 2 validation instances"
    )
    prepare_parser.set_defaults(handler=_prepare_backpas)

    train_parser = subparsers.add_parser(
        "train-backpas", help="run the seeded upstream BackPaS trainer under WSL"
    )
    train_parser.add_argument("dataset", type=Path)
    train_parser.add_argument("--work-dir", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--threads", type=int, default=4)
    train_parser.add_argument("--distribution", default="Ubuntu")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.set_defaults(handler=_train_backpas)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (BackPaSError, DatasetAuditError, DatasetFetchError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _inspect(args: argparse.Namespace) -> int:
    instance = read_opb(args.input)
    payload = {
        "path": str(args.input),
        "problem_type": "PBO" if instance.objective is not None else "PBS",
        "variables": len(instance.variables),
        "constraints": len(instance.constraints),
        "equalities": sum(c.operator == "=" for c in instance.constraints),
        "objective_offset": instance.objective.constant if instance.objective else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _invert(args: argparse.Namespace) -> int:
    instance = read_opb(args.input)
    selected = None if args.variables is None else frozenset(args.variables)
    transformed = invert_polarity(instance, selected)

    if not args.no_provenance:
        scope = "all" if selected is None else ",".join(sorted(selected, key=variable_sort_key))
        transformed = replace(
            transformed,
            comments=(
                *transformed.comments,
                f"backbone-pbo augmentation: polarity; variables={scope}",
            ),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_opb(transformed, args.output)
    return 0


def _fetch_dataset(args: argparse.Namespace) -> int:
    payload = fetch_dmiplib_mvc(args.output, revision=args.revision, force=args.force)
    summary = {
        key: payload[key]
        for key in ("repository", "configuration", "revision", "license", "status")
    }
    summary["files"] = len(payload["files"])
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _audit_dataset(args: argparse.Namespace) -> int:
    dataset = args.dataset.resolve()
    manifest = args.manifest or Path("data/manifests") / f"{dataset.name}.json"
    report = args.report or Path("results/audits") / f"{dataset.name}-audit.md"
    result = audit_dataset(
        dataset,
        partitions_path=args.partitions,
        required_backbone_partitions=(
            () if args.allow_missing_backbones else tuple(args.require_backbones)
        ),
    )
    write_audit_outputs(result, manifest, report)
    output = {
        "dataset": result.dataset,
        "valid": result.valid,
        "manifest": manifest.as_posix(),
        "report": report.as_posix(),
        "summary": result.summary,
        "issues": [issue.__dict__ for issue in result.issues],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.valid else 1


def _extract_backbones(args: argparse.Namespace) -> int:
    limits = {"train": 8, "validation": 2} if args.smoke else None
    payload = extract_backbones(
        args.dataset,
        partitions=tuple(args.partitions),
        threads=args.threads,
        distribution=args.distribution,
        binary_path=args.binary,
        limits=limits,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _prepare_backpas(args: argparse.Namespace) -> int:
    limits = {"train": 8, "validation": 2} if args.smoke else None
    payload = prepare_backpas(
        args.dataset,
        args.work_dir,
        graph_type=args.graph_type,
        distribution=args.distribution,
        limits=limits,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _train_backpas(args: argparse.Namespace) -> int:
    payload = train_backpas(
        args.dataset,
        args.work_dir,
        epochs=args.epochs,
        seed=args.seed,
        threads=args.threads,
        resume=args.resume,
        distribution=args.distribution,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
