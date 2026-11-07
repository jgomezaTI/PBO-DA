import pytest

from backbone_pbo.cli import main
from backbone_pbo.dataset.provenance import (
    DatasetProvenance,
    PartitionRecord,
    TransformationRecord,
)


def _record(**overrides):
    values = {
        "dataset_id": "mvc-original",
        "problem_family": "MVC",
        "source": {"authors_or_organization": "D-MIPLIB", "publication_or_url": "https://example.test"},
        "license_or_terms": {"status": "citation-required", "reference": "dataset card"},
        "version": {"kind": "content-digest", "value": "sha256:abc"},
        "partitions": {
            "train": PartitionRecord("train", ("a.opb", "b.opb"), "train.json"),
            "validation": PartitionRecord("validation", ("c.opb",), "validation.json"),
            "test": PartitionRecord("test", (), None),
        },
        "labels": {
            "train": {"B0": 2, "B1": 2, "NB": 0},
            "validation": {"B0": 1, "B1": 1, "NB": 0},
            "test": {"B0": 0, "B1": 0, "NB": 0},
        },
        "transformations": (),
        "audit": {"status": "passed", "report": "results/audits/MVC.json"},
    }
    values.update(overrides)
    return DatasetProvenance(**values)


def test_provenance_round_trip_is_deterministic():
    record = _record()

    assert DatasetProvenance.from_dict(record.to_dict()) == record
    assert record.to_dict()["partitions"]["test"] == {
        "count": 0,
        "manifest": None,
        "instances": [],
    }


def test_required_fields_and_states_are_validated():
    with pytest.raises(ValueError, match="dataset_id"):
        _record(dataset_id="")
    with pytest.raises(ValueError, match="problem_family"):
        _record(problem_family="TSP")
    with pytest.raises(ValueError, match="license"):
        _record(license_or_terms={"status": "invented", "reference": None})
    with pytest.raises(ValueError, match="audit"):
        _record(audit={"status": "approved", "report": None})


def test_partition_count_is_derived_from_instances():
    partition = PartitionRecord("train", ("a.opb", "b.opb"), "train.json")

    assert partition.count == 2


def test_duplicate_across_partitions_reports_identifier():
    partitions = {
        "train": PartitionRecord("train", ("same.opb",), "train.json"),
        "validation": PartitionRecord("validation", ("same.opb",), "validation.json"),
        "test": PartitionRecord("test", (), None),
    }

    with pytest.raises(ValueError, match=r"same\.opb"):
        _record(partitions=partitions)


def test_derived_instances_must_remain_in_train():
    transformation = TransformationRecord(
        name="polarity_global",
        applies_to=("validation",),
        parent_mapping="parent-map.json",
    )

    with pytest.raises(ValueError, match="validation"):
        _record(transformations=(transformation,))


def test_cli_validates_serialized_provenance(repo_tmp_path, capsys):
    path = repo_tmp_path / "provenance.json"
    path.write_text(__import__("json").dumps(_record().to_dict()), encoding="utf-8")

    assert main(["validate-provenance", str(path)]) == 0
    assert __import__("json").loads(capsys.readouterr().out)["valid"] is True
