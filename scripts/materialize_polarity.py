"""Materialize a polarity treatment without touching the source dataset."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from backbone_pbo.backbone import dumps_backbone, flip_backbone_label, loads_backbone
from backbone_pbo.io.lp import invert_lp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    partitions = json.loads((source / "partitions.json").read_text(encoding="utf-8"))["partitions"]
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    (output / "instance").mkdir(parents=True)
    (output / "backbone").mkdir()
    derived: list[str] = []
    for partition, names in partitions.items():
        for name in names:
            source_instance = source / "instance" / name
            source_backbone = source / "backbone" / f"{name}.backbone"
            if not source_instance.exists() or not source_backbone.exists():
                raise FileNotFoundError(name)
            if partition == "train":
                target_name = f"polarity_{name}"
                (output / "instance" / target_name).write_text(
                    invert_lp(source_instance.read_text(encoding="ascii")),
                    encoding="ascii",
                )
                source_text = source_instance.read_text(encoding="ascii")
                variables = set(re.findall(r"\b(?:v|x)\d+\b", source_text))
                labels = loads_backbone(
                    source_backbone.read_text(encoding="ascii"), variables
                )
                transformed = {key: flip_backbone_label(value) for key, value in labels.items()}
                (output / "backbone" / f"{target_name}.backbone").write_text(
                    dumps_backbone(transformed), encoding="ascii"
                )
                derived.append(target_name)
            shutil.copy2(source_instance, output / "instance" / name)
            shutil.copy2(source_backbone, output / "backbone" / f"{name}.backbone")
    output_partitions = {
        "train": partitions["train"] + derived,
        "validation": list(partitions.get("validation", [])),
        "test": list(partitions.get("test", [])),
    }
    (output / "partitions.json").write_text(
        json.dumps(output_partitions, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
