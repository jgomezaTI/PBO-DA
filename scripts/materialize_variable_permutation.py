"""Create a variable-renamed augmentation without changing the source dataset."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

VARIABLE = re.compile(r"\b(?:x|v)\d+\b")


def permute_text(content: str, mapping: dict[str, str]) -> str:
    return VARIABLE.sub(lambda match: mapping.get(match.group(0), match.group(0)), content)


def variables_in(content: str) -> list[str]:
    return sorted(set(VARIABLE.findall(content)), key=lambda name: (name[0], int(name[1:])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    partitions = json.loads((source / "partitions.json").read_text(encoding="utf-8"))["partitions"]
    for directory in ("instance", "backbone"):
        (output / directory).mkdir(parents=True)
    derived: list[str] = []
    rng = random.Random(args.seed)
    for partition, names in partitions.items():
        for name in names:
            instance = source / "instance" / name
            backbone = source / "backbone" / f"{name}.backbone"
            if not instance.is_file() or not backbone.is_file():
                raise FileNotFoundError(name)
            shutil.copy2(instance, output / "instance" / name)
            shutil.copy2(backbone, output / "backbone" / f"{name}.backbone")
            if partition != "train":
                continue
            content = instance.read_text(encoding="ascii")
            names_in_instance = variables_in(content)
            shuffled = names_in_instance[:]
            rng.shuffle(shuffled)
            mapping = dict(zip(names_in_instance, shuffled, strict=True))
            target = f"perm_{name}"
            (output / "instance" / target).write_text(permute_text(content, mapping), encoding="ascii")
            (output / "backbone" / f"{target}.backbone").write_text(
                permute_text(backbone.read_text(encoding="ascii"), mapping), encoding="ascii"
            )
            derived.append(target)
    output_partitions = {
        "train": list(partitions.get("train", [])) + derived,
        "validation": list(partitions.get("validation", [])),
        "test": list(partitions.get("test", [])),
    }
    (output / "partitions.json").write_text(json.dumps(output_partitions, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"derived": len(derived), "train": len(output_partitions["train"]), "validation": len(output_partitions["validation"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
