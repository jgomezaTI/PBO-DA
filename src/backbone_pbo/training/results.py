"""Experiment, checkpoint, and result contracts for controlled comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

RUN_STATES = {
    "planned",
    "prepared",
    "smoke-passed",
    "running",
    "interrupted",
    "failed",
    "completed-validation",
    "completed-test",
}
EVIDENCE_LEVELS = {"preliminary-validation", "completed-validation", "completed-test"}
TRANSITIONS = {
    "planned": {"prepared", "failed"},
    "prepared": {"smoke-passed", "failed"},
    "smoke-passed": {"running", "failed"},
    "running": {"interrupted", "failed", "completed-validation"},
    "interrupted": {"running", "failed"},
    "completed-validation": {"completed-test"},
    "failed": set(),
    "completed-test": set(),
}


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    problem_family: str
    scenario: str
    seed: int
    dataset_manifest: str
    configuration: dict[str, Any]
    software_environment: dict[str, Any]
    checkpoint_selection: str
    run_status: str
    artifacts: dict[str, str]

    def __post_init__(self) -> None:
        if not self.experiment_id or not self.dataset_manifest:
            raise ValueError("experiment_id and dataset_manifest are required")
        if self.run_status not in RUN_STATES:
            raise ValueError(f"Invalid run_status: {self.run_status!r}")
        if self.checkpoint_selection != "minimum_validation_loss":
            raise ValueError("checkpoint_selection must be minimum_validation_loss")
        required_environment = {"project_commit", "upstream_commit", "runtime_versions"}
        missing = required_environment.difference(self.software_environment)
        if missing:
            raise ValueError(f"Missing software environment fields: {sorted(missing)}")

    def transition(self, target: str) -> ExperimentRecord:
        if target not in RUN_STATES:
            raise ValueError(f"Invalid run_status: {target!r}")
        if target not in TRANSITIONS[self.run_status]:
            raise ValueError(f"Invalid transition {self.run_status!r} -> {target!r}")
        return replace(self, run_status=target)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentRecord:
        return cls(**payload)


@dataclass(frozen=True)
class CheckpointRecord:
    epoch: int
    validation_loss: float | None
    partition_metrics: dict[str, Any]
    model_path: str
    is_selected: bool = False

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise ValueError("epoch must identify a completed epoch (>= 1)")
        if self.validation_loss is None:
            raise ValueError("validation_loss is required")
        if not self.model_path:
            raise ValueError("model_path is required")


@dataclass(frozen=True)
class ResultRecord:
    experiment_id: str
    evidence_level: str
    best_epoch: int
    metrics: dict[str, Any]
    class_support: dict[str, int]
    generated_from: tuple[str, ...]
    generated_at: str

    def __post_init__(self) -> None:
        if self.evidence_level not in EVIDENCE_LEVELS:
            raise ValueError(f"Invalid evidence_level: {self.evidence_level!r}")
        if self.best_epoch < 1:
            raise ValueError("best_epoch must identify a completed epoch")

    @classmethod
    def from_metrics(
        cls,
        *,
        experiment_id: str,
        evidence_level: str,
        best_epoch: int,
        metrics: dict[str, Any],
        class_support: dict[str, int],
        generated_from: tuple[str, ...],
        generated_at: str,
    ) -> ResultRecord:
        normalized = {
            label: ("N/A" if class_support.get(label, 0) == 0 else value)
            for label, value in metrics.items()
        }
        return cls(
            experiment_id=experiment_id,
            evidence_level=evidence_level,
            best_epoch=best_epoch,
            metrics=normalized,
            class_support=dict(class_support),
            generated_from=tuple(generated_from),
            generated_at=generated_at,
        )


@dataclass(frozen=True)
class ComparisonResult:
    comparable: bool
    differences: tuple[str, ...]


def compare_experiments(
    baseline: ExperimentRecord, treatment: ExperimentRecord
) -> ComparisonResult:
    """Return whether runs differ only by scenario and training augmentation."""

    differences: list[str] = []
    if baseline.scenario != treatment.scenario:
        differences.append("scenario")
    if baseline.problem_family != treatment.problem_family:
        differences.append("problem_family")
    if baseline.seed != treatment.seed:
        differences.append("seed")
    if baseline.checkpoint_selection != treatment.checkpoint_selection:
        differences.append("checkpoint_selection")

    keys = {
        "partition_digest",
        "representation",
        "architecture",
        "hyperparameters",
        "epochs",
    }
    for key in sorted(keys):
        if baseline.configuration.get(key) != treatment.configuration.get(key):
            differences.append(f"configuration.{key}")
    if baseline.configuration.get("training_augmentation") != treatment.configuration.get(
        "training_augmentation"
    ):
        differences.append("training_augmentation")
    for key in ("project_commit", "upstream_commit"):
        if baseline.software_environment.get(key) != treatment.software_environment.get(key):
            differences.append(f"software_environment.{key}")

    allowed = {"scenario", "training_augmentation"}
    return ComparisonResult(
        comparable=not set(differences).difference(allowed), differences=tuple(differences)
    )
