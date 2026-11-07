"""Validated, serializable records for dataset provenance and partition integrity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PARTITIONS = ("train", "validation", "test")
PROBLEM_FAMILIES = {"MVC", "MIS", "CA", "approved-other"}
LICENSE_STATES = {"known", "permission-granted", "citation-required", "unknown"}
AUDIT_STATES = {"unverified", "partial", "passed", "failed"}


@dataclass(frozen=True)
class PartitionRecord:
    """Manifest-backed membership of one immutable dataset partition."""

    name: str
    instances: tuple[str, ...]
    manifest: str | None

    def __post_init__(self) -> None:
        if self.name not in PARTITIONS:
            raise ValueError(f"Unsupported partition: {self.name!r}")
        normalized = tuple(str(item) for item in self.instances)
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"Duplicate instance inside partition {self.name!r}")
        if self.name != "test" and not self.manifest:
            raise ValueError(f"Partition {self.name!r} requires a manifest")
        object.__setattr__(self, "instances", normalized)

    @property
    def count(self) -> int:
        return len(self.instances)

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "manifest": self.manifest, "instances": list(self.instances)}

    @classmethod
    def from_dict(cls, name: str, payload: dict[str, Any]) -> PartitionRecord:
        instances = tuple(payload.get("instances", ()))
        declared = payload.get("count", len(instances))
        if declared != len(instances):
            raise ValueError(
                f"Partition {name!r} declares {declared} instances but lists {len(instances)}"
            )
        return cls(name=name, instances=instances, manifest=payload.get("manifest"))


@dataclass(frozen=True)
class TransformationRecord:
    """A derived-instance transformation and its parent mapping."""

    name: str
    applies_to: tuple[str, ...]
    parent_mapping: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("transformation name is required")
        if not self.parent_mapping:
            raise ValueError("parent_mapping is required")
        invalid = set(self.applies_to).difference(PARTITIONS)
        if invalid:
            raise ValueError(f"Unsupported transformation partitions: {sorted(invalid)}")


@dataclass(frozen=True)
class DatasetProvenance:
    """Complete provenance record required before reportable training."""

    dataset_id: str
    problem_family: str
    source: dict[str, Any]
    license_or_terms: dict[str, Any]
    version: dict[str, Any]
    partitions: dict[str, PartitionRecord]
    labels: dict[str, dict[str, int]]
    transformations: tuple[TransformationRecord, ...]
    audit: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id is required")
        if self.problem_family not in PROBLEM_FAMILIES:
            raise ValueError(f"Unsupported problem_family: {self.problem_family!r}")
        if not self.source.get("authors_or_organization") or not self.source.get(
            "publication_or_url"
        ):
            raise ValueError("source authors_or_organization and publication_or_url are required")
        if self.license_or_terms.get("status") not in LICENSE_STATES:
            raise ValueError("license status is invalid")
        if not self.version.get("kind") or not self.version.get("value"):
            raise ValueError("version kind and value are required")
        missing = set(PARTITIONS).difference(self.partitions)
        if missing:
            raise ValueError(f"Missing partitions: {sorted(missing)}")
        if self.audit.get("status") not in AUDIT_STATES:
            raise ValueError("audit status is invalid")
        self._validate_partition_integrity()
        self._validate_transformations()
        self._validate_labels()

    def _validate_partition_integrity(self) -> None:
        owners: dict[str, str] = {}
        duplicates: list[str] = []
        for name in PARTITIONS:
            record = self.partitions[name]
            if record.name != name:
                raise ValueError(f"Partition key {name!r} contains record {record.name!r}")
            for instance_id in record.instances:
                if instance_id in owners:
                    duplicates.append(instance_id)
                else:
                    owners[instance_id] = name
        if duplicates:
            raise ValueError(f"Instances assigned to multiple partitions: {sorted(duplicates)}")

    def _validate_transformations(self) -> None:
        for transformation in self.transformations:
            invalid = set(transformation.applies_to).difference({"train"})
            if invalid:
                raise ValueError(
                    f"Transformation {transformation.name!r} applies outside train: "
                    f"{sorted(invalid)}"
                )

    def _validate_labels(self) -> None:
        for partition in PARTITIONS:
            counts = self.labels.get(partition)
            if counts is None:
                raise ValueError(f"Missing label counts for partition {partition!r}")
            for label in ("B0", "B1", "NB"):
                value = counts.get(label)
                if not isinstance(value, int) or value < 0:
                    raise ValueError(f"Invalid {label} count for partition {partition!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "problem_family": self.problem_family,
            "source": dict(self.source),
            "license_or_terms": dict(self.license_or_terms),
            "version": dict(self.version),
            "partitions": {name: self.partitions[name].to_dict() for name in PARTITIONS},
            "labels": {name: dict(self.labels[name]) for name in PARTITIONS},
            "transformations": [asdict(item) for item in self.transformations],
            "audit": dict(self.audit),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DatasetProvenance:
        partitions = {
            name: PartitionRecord.from_dict(name, record)
            for name, record in payload["partitions"].items()
        }
        transformations = tuple(
            TransformationRecord(
                name=item["name"],
                applies_to=tuple(item["applies_to"]),
                parent_mapping=item["parent_mapping"],
            )
            for item in payload.get("transformations", ())
        )
        return cls(
            dataset_id=payload["dataset_id"],
            problem_family=payload["problem_family"],
            source=dict(payload["source"]),
            license_or_terms=dict(payload["license_or_terms"]),
            version=dict(payload["version"]),
            partitions=partitions,
            labels={name: dict(counts) for name, counts in payload["labels"].items()},
            transformations=transformations,
            audit=dict(payload["audit"]),
        )
