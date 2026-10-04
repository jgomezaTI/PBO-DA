import json

import pytest

from backbone_pbo.dataset import dmiplib
from backbone_pbo.dataset.dmiplib import DatasetFetchError, fetch_dmiplib_mvc


def test_fetch_mvc_exports_deterministic_names_and_provenance(repo_tmp_path, monkeypatch):
    monkeypatch.setattr(dmiplib, "MVC_SPLITS", {"train": 2, "validation": 1})

    def loader(_repository, _configuration, *, split, revision, streaming):
        assert revision == "revision-1"
        assert streaming is True
        count = 2 if split == "train" else 1
        return [
            {"Unnamed: 0": index, "MILP": b"Minimize\n obj: x1\nBinary\n x1\nEnd\n"}
            for index in range(count)
        ]

    output = repo_tmp_path / "MVC"
    source = fetch_dmiplib_mvc(
        output,
        dataset_loader=loader,
        revision_resolver=lambda _repository: "revision-1",
    )

    assert source["revision"] == "revision-1"
    assert sorted(path.name for path in (output / "instance").iterdir()) == [
        "train_0000.lp",
        "train_0001.lp",
        "valid_0000.lp",
    ]
    partitions = json.loads((output / "partitions.json").read_text(encoding="utf-8"))
    assert partitions["partitions"]["validation"] == ["valid_0000.lp"]


def test_fetch_refuses_to_overwrite_different_instance(repo_tmp_path, monkeypatch):
    monkeypatch.setattr(dmiplib, "MVC_SPLITS", {"train": 1})

    def loader(_repository, _configuration, **_kwargs):
        return [{"MILP": "Minimize\n obj: x1\nBinary\n x1\nEnd\n"}]

    output = repo_tmp_path / "MVC"
    fetch_dmiplib_mvc(
        output,
        dataset_loader=loader,
        revision_resolver=lambda _repository: "revision-1",
    )
    (output / "instance/train_0000.lp").write_text("different", encoding="utf-8")

    with pytest.raises(DatasetFetchError, match="Refusing to overwrite"):
        fetch_dmiplib_mvc(
            output,
            dataset_loader=loader,
            revision_resolver=lambda _repository: "revision-1",
        )
