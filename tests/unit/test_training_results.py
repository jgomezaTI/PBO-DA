import pytest

from backbone_pbo.cli import main
from backbone_pbo.training.results import (
    CheckpointRecord,
    ExperimentRecord,
    ResultRecord,
    compare_experiments,
)


def _experiment(**overrides):
    values = {
        "experiment_id": "MVC/baseline/seed-0",
        "problem_family": "MVC",
        "scenario": "baseline",
        "seed": 0,
        "dataset_manifest": "results/MVC/baseline/seed-0/partitions.json",
        "configuration": {
            "partition_digest": "abc",
            "representation": "graph_with_literals",
            "architecture": {"layer": "GTR", "layers": 8},
            "hyperparameters": {"learning_rate": 0.001, "batch_size": 32},
            "epochs": 200,
        },
        "software_environment": {
            "project_commit": "project",
            "upstream_commit": "upstream",
            "runtime_versions": "runtime-versions.json",
        },
        "checkpoint_selection": "minimum_validation_loss",
        "run_status": "planned",
        "artifacts": {"training_log": "training_log.csv"},
    }
    values.update(overrides)
    return ExperimentRecord(**values)


def test_experiment_round_trip_uses_run_status():
    experiment = _experiment()
    payload = experiment.to_dict()

    assert "status" not in payload
    assert payload["run_status"] == "planned"
    assert ExperimentRecord.from_dict(payload) == experiment


def test_invalid_transition_is_rejected():
    experiment = _experiment()

    with pytest.raises(ValueError, match=r"planned.*running"):
        experiment.transition("running")


def test_interrupted_experiment_can_resume():
    experiment = _experiment(run_status="interrupted")

    assert experiment.transition("running").run_status == "running"


def test_result_marks_absent_class_as_not_applicable():
    result = ResultRecord.from_metrics(
        experiment_id="MVC/baseline/seed-0",
        evidence_level="preliminary-validation",
        best_epoch=117,
        metrics={"B0": {"f1": 0.9}, "B1": {"f1": 0.8}, "NB": {"f1": 0.0}},
        class_support={"B0": 10, "B1": 12, "NB": 0},
        generated_from=("training_log.csv",),
        generated_at="2026-10-20T00:00:00Z",
    )

    assert result.metrics["NB"] == "N/A"


def test_checkpoint_requires_completed_epoch_and_validation_loss():
    with pytest.raises(ValueError, match="epoch"):
        CheckpointRecord(epoch=0, validation_loss=0.2, partition_metrics={}, model_path="x.pth")
    with pytest.raises(ValueError, match="validation_loss"):
        CheckpointRecord(epoch=1, validation_loss=None, partition_metrics={}, model_path="x.pth")


def test_comparison_allows_only_scenario_and_training_augmentation_difference():
    baseline = _experiment()
    polarity = _experiment(
        experiment_id="MVC/polarity/seed-0",
        scenario="polarity",
        configuration={**baseline.configuration, "training_augmentation": "polarity"},
    )

    comparison = compare_experiments(baseline, polarity)
    assert comparison.comparable
    assert comparison.differences == ("scenario", "training_augmentation")


def test_comparison_rejects_architecture_difference():
    baseline = _experiment()
    changed = {
        **baseline.configuration,
        "architecture": {"layer": "GTR", "layers": 4},
        "training_augmentation": "polarity",
    }
    polarity = _experiment(
        experiment_id="MVC/polarity/seed-0", scenario="polarity", configuration=changed
    )

    comparison = compare_experiments(baseline, polarity)
    assert not comparison.comparable
    assert "configuration.architecture" in comparison.differences


def test_cli_validates_serialized_experiment(repo_tmp_path, capsys):
    path = repo_tmp_path / "experiment.json"
    path.write_text(__import__("json").dumps(_experiment().to_dict()), encoding="utf-8")

    assert main(["validate-experiment", str(path)]) == 0
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload == {
        "experiment_id": "MVC/baseline/seed-0",
        "run_status": "planned",
        "valid": True,
    }
