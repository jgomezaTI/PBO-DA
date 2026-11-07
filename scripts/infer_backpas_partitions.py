"""Infer BackPaS partitions from instance/backbone filename prefixes.

Only instance files with a matching ``.backbone`` file are included. This is
intended for imported archives whose partition manifest was not provided.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def partition_for(name: str) -> str | None:
    if re.match(r"^(?:train|valid)(?:_|\.)", name):
        return name.split("_", 1)[0]
    if re.match(r"^backpas_(?:train|valid)_", name):
        return name.split("_", 2)[1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    instance_dir = args.dataset / "instance"
    backbone_dir = args.dataset / "backbone"
    instances = {path.name for path in instance_dir.iterdir() if path.is_file()}
    partitions: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for backbone in sorted(backbone_dir.iterdir(), key=lambda path: path.name):
        if not backbone.is_file() or not backbone.name.endswith(".backbone"):
            continue
        instance_name = backbone.name.removesuffix(".backbone")
        if instance_name not in instances:
            continue
        # BackPaS only emits a target when the backbone has its completion
        # marker.  Exclude incomplete solver outputs from the inferred split.
        try:
            if backbone.read_text(encoding="utf-8").splitlines()[-1].strip() != "b 0":
                continue
        except (OSError, IndexError):
            continue
        partition = partition_for(instance_name)
        if partition == "train":
            partitions["train"].append(instance_name)
        elif partition == "valid":
            partitions["validation"].append(instance_name)
        elif partition == "test":
            partitions["test"].append(instance_name)
    payload = {"partitions": partitions}
    (args.dataset / "partitions.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: len(value) for key, value in partitions.items()}))


if __name__ == "__main__":
    main()
