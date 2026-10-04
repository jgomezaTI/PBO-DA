import json
import shutil

from backbone_pbo.cli import main
from backbone_pbo.dataset.audit import audit_dataset, write_audit_outputs


def _make_dataset(root, *, with_backbone=True):
    dataset = root / "MVC"
    instance_dir = dataset / "instance"
    backbone_dir = dataset / "backbone"
    instance_dir.mkdir(parents=True)
    backbone_dir.mkdir()
    shutil.copy2("tests/fixtures/opb/tiny.opb", instance_dir / "train_0000.opb")
    if with_backbone:
        shutil.copy2(
            "tests/fixtures/backbone/tiny.opb.backbone",
            backbone_dir / "train_0000.opb.backbone",
        )
    (dataset / "partitions.json").write_text(
        json.dumps({"schema_version": 1, "partitions": {"train": ["train_0000.opb"]}}),
        encoding="utf-8",
    )
    (dataset / "source.json").write_text(
        json.dumps({"repository": "fixture", "revision": "abc123"}), encoding="utf-8"
    )
    return dataset


def test_valid_dataset_has_deterministic_manifest(repo_tmp_path):
    dataset = _make_dataset(repo_tmp_path)
    first = audit_dataset(dataset, required_backbone_partitions=("train",))
    second = audit_dataset(dataset, required_backbone_partitions=("train",))

    assert first.valid
    assert first.to_dict() == second.to_dict()
    assert first.summary["by_partition"]["train"]["labels"] == {
        "B0": 1,
        "B1": 1,
        "NB": 1,
    }

    manifest = repo_tmp_path / "manifest.json"
    report = repo_tmp_path / "report.md"
    write_audit_outputs(first, manifest, report)
    original = manifest.read_bytes()
    write_audit_outputs(second, manifest, report)
    assert manifest.read_bytes() == original


def test_missing_required_backbone_fails(repo_tmp_path):
    dataset = _make_dataset(repo_tmp_path, with_backbone=False)
    result = audit_dataset(dataset, required_backbone_partitions=("train",))

    assert not result.valid
    assert any(issue.code == "missing-backbone" for issue in result.issues)


def test_partition_overlap_and_duplicate_instances_fail(repo_tmp_path):
    dataset = _make_dataset(repo_tmp_path)
    shutil.copy2(dataset / "instance/train_0000.opb", dataset / "instance/valid_0000.opb")
    shutil.copy2(
        dataset / "backbone/train_0000.opb.backbone",
        dataset / "backbone/valid_0000.opb.backbone",
    )
    (dataset / "partitions.json").write_text(
        json.dumps(
            {
                "partitions": {
                    "train": ["train_0000.opb", "valid_0000.opb"],
                    "validation": ["valid_0000.opb"],
                }
            }
        ),
        encoding="utf-8",
    )

    result = audit_dataset(dataset, required_backbone_partitions=("train", "validation"))
    codes = {issue.code for issue in result.issues}
    assert "partition-overlap" in codes
    assert "duplicate-instance" in codes


def test_unknown_backbone_variable_fails(repo_tmp_path):
    dataset = _make_dataset(repo_tmp_path)
    (dataset / "backbone/train_0000.opb.backbone").write_text("b x999\nb 0\n", encoding="ascii")

    result = audit_dataset(dataset, required_backbone_partitions=("train",))
    assert not result.valid
    assert any(issue.code == "invalid-backbone" for issue in result.issues)


def test_audit_cli_returns_one_for_invalid_dataset(repo_tmp_path, capsys):
    dataset = _make_dataset(repo_tmp_path, with_backbone=False)
    exit_code = main(
        [
            "audit-dataset",
            str(dataset),
            "--manifest",
            str(repo_tmp_path / "manifest.json"),
            "--report",
            str(repo_tmp_path / "report.md"),
            "--require-backbones",
            "train",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["valid"] is False


def test_lp_instance_is_audited_with_pyscipopt(repo_tmp_path):
    dataset = repo_tmp_path / "LP"
    (dataset / "instance").mkdir(parents=True)
    (dataset / "backbone").mkdir()
    (dataset / "instance/train_0000.lp").write_text(
        "Minimize\n obj: + 1 x1 + 2 x2\nSubject To\n c1: x1 + x2 >= 1\nBinary\n x1 x2\nEnd\n",
        encoding="utf-8",
    )
    (dataset / "backbone/train_0000.lp.backbone").write_text("b -x1\nb 0\n", encoding="ascii")
    (dataset / "partitions.json").write_text(
        json.dumps({"partitions": {"train": ["train_0000.lp"]}}), encoding="utf-8"
    )
    (dataset / "source.json").write_text("{}", encoding="utf-8")

    result = audit_dataset(dataset, required_backbone_partitions=("train",))

    assert result.valid
    assert result.files[0]["binary_variables"] == 2
    assert result.files[0]["constraints"] == 1
