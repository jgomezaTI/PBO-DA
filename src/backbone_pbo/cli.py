"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from backbone_pbo.augmentation.polarity import invert_polarity
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


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


if __name__ == "__main__":
    raise SystemExit(main())
