"""Dataset loader, partitioner, and augmentation pipeline for PBO instances."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from torch_geometric.data import Data

from backbone_pbo.augmentation.polarity import invert_polarity
from backbone_pbo.backbone import BackboneLabel, read_backbone, transform_backbone
from backbone_pbo.graph.converter import instance_to_bipartite_graph
from backbone_pbo.io.opb import read_opb
from backbone_pbo.model import PBOInstance


@dataclass
class DatasetSplits:
    train_graphs: list[Data]
    val_graphs: list[Data]
    test_graphs: list[Data]
    stats: dict[str, int | float | dict]


def load_raw_dataset(
    dataset_dir: str | Path,
) -> list[tuple[PBOInstance, dict[str, BackboneLabel], str]]:
    """Load a dataset that follows the canonical BackPaS directory layout."""
    path = Path(dataset_dir)
    inst_dir = path / "instance"
    bb_dir = path / "backbone"

    if not inst_dir.exists() or not bb_dir.exists():
        raise FileNotFoundError(
            f"Expected BackPaS directories {inst_dir} and {bb_dir}. "
            "Use the singular names 'instance' and 'backbone'."
        )

    items = []
    for opb_file in sorted(inst_dir.glob("*.opb")):
        bb_file = bb_dir / f"{opb_file.name}.backbone"
        if not bb_file.exists():
            continue
        instance = read_opb(opb_file)
        labels = read_backbone(bb_file, instance.variables)
        items.append((instance, labels, opb_file.name))

    return items


def _parse_filename_meta(filename: str) -> tuple[str, str]:
    # e.g. vc_S_0001_n24.opb
    parts = filename.split("_")
    if len(parts) >= 2 and parts[1] in ("S", "M", "L"):
        return parts[1], parts[0]
    return "M", "pbo"


def create_dataset_splits(
    items: Sequence[tuple[PBOInstance, dict[str, BackboneLabel], str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    augment_polarity: bool = False,
) -> DatasetSplits:
    """Partition instances and optionally apply polarity inversion ONLY to the train split."""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_items = shuffled[:n_train]
    val_items = shuffled[n_train : n_train + n_val]
    test_items = shuffled[n_train + n_val :]

    train_graphs: list[Data] = []
    val_graphs: list[Data] = []
    test_graphs: list[Data] = []

    train_counts_orig = {0: 0, 1: 0, 2: 0}
    train_counts_final = {0: 0, 1: 0, 2: 0}

    # 1. Process Train split
    for instance, labels, fname in train_items:
        scale, fam = _parse_filename_meta(fname)
        for lbl in labels.values():
            train_counts_orig[int(lbl)] += 1
            train_counts_final[int(lbl)] += 1
        g_orig = instance_to_bipartite_graph(instance, labels, scale=scale, family=fam)
        train_graphs.append(g_orig)

        if augment_polarity:
            aug_inst = invert_polarity(instance)
            aug_labels = transform_backbone(labels)
            for lbl in aug_labels.values():
                train_counts_final[int(lbl)] += 1
            g_aug = instance_to_bipartite_graph(aug_inst, aug_labels, scale=scale, family=fam)
            train_graphs.append(g_aug)

    # 2. Process Validation split (strictly original, no augmentation)
    for instance, labels, fname in val_items:
        scale, fam = _parse_filename_meta(fname)
        val_graphs.append(instance_to_bipartite_graph(instance, labels, scale=scale, family=fam))

    # 3. Process Test split (strictly original, no augmentation)
    for instance, labels, fname in test_items:
        scale, fam = _parse_filename_meta(fname)
        test_graphs.append(instance_to_bipartite_graph(instance, labels, scale=scale, family=fam))

    stats = {
        "num_train_instances": len(train_graphs),
        "num_val_instances": len(val_graphs),
        "num_test_instances": len(test_graphs),
        "train_class_distribution_original": train_counts_orig,
        "train_class_distribution_final": train_counts_final,
        "augmented": augment_polarity,
    }

    return DatasetSplits(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        test_graphs=test_graphs,
        stats=stats,
    )
