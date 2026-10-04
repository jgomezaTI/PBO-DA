import json
import os
import shutil

import pytest

from backbone_pbo.backbone import read_backbone
from backbone_pbo.dataset.audit import inspect_instance
from backbone_pbo.dataset.backpas import extract_backbones


@pytest.mark.gurobi
def test_guroback_extracts_a_complete_backbone_under_wsl(repo_tmp_path):
    if os.environ.get("RUN_GUROBI_TESTS") != "1":
        pytest.skip("Set RUN_GUROBI_TESTS=1 after activating a Gurobi WSL license")

    dataset = repo_tmp_path / "guroback-fixture"
    (dataset / "instance").mkdir(parents=True)
    (dataset / "backbone").mkdir()
    instance = dataset / "instance/train_tiny.opb"
    shutil.copy2("tests/fixtures/opb/tiny.opb", instance)
    (dataset / "partitions.json").write_text(
        json.dumps({"partitions": {"train": [instance.name], "validation": []}}),
        encoding="utf-8",
    )

    stats = extract_backbones(dataset, partitions=("train",))
    labels = read_backbone(
        dataset / f"backbone/{instance.name}.backbone", inspect_instance(instance).variables
    )

    assert stats["extracted"] == 1
    assert set(labels) == {"x1", "x2", "x3"}
