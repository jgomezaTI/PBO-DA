"""Deterministic validation for datasets using the BackPaS directory contract."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backbone_pbo.backbone import BackboneFormatError, read_backbone
from backbone_pbo.io.opb import read_opb
from backbone_pbo.model import variable_sort_key

SUPPORTED_INSTANCE_SUFFIXES = {".lp", ".mps", ".opb"}


class DatasetAuditError(RuntimeError):
    """Raised when an audit cannot be executed rather than when data is invalid."""


@dataclass(frozen=True)
class InstanceShape:
    variables: tuple[str, ...]
    binary_variables: int
    constraints: int


@dataclass(frozen=True)
class AuditIssue:
    code: str
    severity: str
    path: str
    message: str


@dataclass
class AuditResult:
    dataset: str
    root: str
    files: list[dict[str, Any]] = field(default_factory=list)
    partitions: dict[str, list[str]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    issues: list[AuditIssue] = field(default_factory=list)
    source: dict[str, Any] | None = None
    schema_version: int = 1

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "root": self.root,
            "valid": self.valid,
            "source": self.source,
            "partitions": self.partitions,
            "summary": self.summary,
            "files": self.files,
            "issues": [asdict(issue) for issue in self.issues],
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_instance(path: Path) -> InstanceShape:
    """Read an optimization instance without solving it."""

    suffix = path.suffix.lower()
    if suffix == ".opb":
        instance = read_opb(path)
        variables = tuple(sorted(instance.variables, key=variable_sort_key))
        return InstanceShape(variables, len(variables), len(instance.constraints))
    if suffix not in {".lp", ".mps"}:
        raise DatasetAuditError(f"Unsupported instance format: {path.suffix}")

    try:
        from pyscipopt import Model
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise DatasetAuditError(
            "LP/MPS auditing requires the data extra: install with `uv sync --extra data`."
        ) from exc

    model = Model()
    model.hideOutput(True)
    try:
        model.readProblem(str(path))
        model_variables = list(model.getVars())
        variables = tuple(sorted((var.name for var in model_variables), key=variable_sort_key))
        binary_variables = sum(var.vtype() == "BINARY" for var in model_variables)
        return InstanceShape(variables, binary_variables, int(model.getNConss()))
    except Exception as exc:
        raise DatasetAuditError(f"Unable to parse {path.name}: {exc}") from exc
    finally:
        with suppress(Exception):
            model.freeProb()


def load_partitions(path: Path) -> dict[str, list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetAuditError(f"Unable to read partition file {path}: {exc}") from exc
    raw = payload.get("partitions", payload)
    if not isinstance(raw, dict):
        raise DatasetAuditError("The partition document must contain a 'partitions' object")
    partitions: dict[str, list[str]] = {}
    for name, entries in raw.items():
        if (
            not isinstance(name, str)
            or not isinstance(entries, list)
            or not all(isinstance(entry, str) for entry in entries)
        ):
            raise DatasetAuditError(f"Invalid partition entry: {name!r}")
        partitions[name] = sorted(entries)
    return dict(sorted(partitions.items()))


def audit_dataset(
    dataset_dir: str | Path,
    *,
    partitions_path: str | Path | None = None,
    required_backbone_partitions: tuple[str, ...] = ("train", "validation"),
) -> AuditResult:
    root = Path(dataset_dir).resolve()
    instance_dir = root / "instance"
    backbone_dir = root / "backbone"
    partition_file = Path(partitions_path) if partitions_path else root / "partitions.json"
    result = AuditResult(dataset=root.name, root=root.as_posix())

    for directory, code in (
        (instance_dir, "missing-instance-directory"),
        (backbone_dir, "missing-backbone-directory"),
    ):
        if not directory.is_dir():
            result.issues.append(
                AuditIssue(code, "error", directory.as_posix(), "Required directory is missing")
            )
    if not instance_dir.is_dir():
        result.summary = _summarize(result)
        return result
    try:
        result.partitions = load_partitions(partition_file)
    except DatasetAuditError as exc:
        result.issues.append(
            AuditIssue("invalid-partitions", "error", partition_file.as_posix(), str(exc))
        )
        result.summary = _summarize(result)
        return result

    source_path = root / "source.json"
    if source_path.exists():
        try:
            result.source = json.loads(source_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            result.issues.append(
                AuditIssue("invalid-source", "error", "source.json", f"Invalid JSON: {exc}")
            )
    else:
        result.issues.append(
            AuditIssue("missing-source", "warning", "source.json", "Dataset provenance is missing")
        )

    instance_paths = sorted(
        path
        for path in instance_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_INSTANCE_SUFFIXES
    )
    instance_names = {path.name for path in instance_paths}
    assignments: dict[str, list[str]] = defaultdict(list)
    for partition, names in result.partitions.items():
        for name in names:
            assignments[name].append(partition)
            if name not in instance_names:
                result.issues.append(
                    AuditIssue(
                        "partition-file-missing",
                        "error",
                        name,
                        f"Listed in partition {partition!r} but no instance exists",
                    )
                )
    for name, assigned in sorted(assignments.items()):
        if len(assigned) > 1:
            result.issues.append(
                AuditIssue(
                    "partition-overlap",
                    "error",
                    name,
                    f"Assigned to multiple partitions: {', '.join(sorted(assigned))}",
                )
            )
    for name in sorted(instance_names.difference(assignments)):
        result.issues.append(
            AuditIssue(
                "unassigned-instance", "error", name, "Instance is not assigned to a partition"
            )
        )

    checksum_names: dict[str, list[str]] = defaultdict(list)
    for instance_path in instance_paths:
        relative = instance_path.relative_to(root).as_posix()
        partition_names = assignments.get(instance_path.name, [])
        partition = partition_names[0] if len(partition_names) == 1 else None
        checksum = sha256_file(instance_path)
        checksum_names[checksum].append(instance_path.name)
        record: dict[str, Any] = {
            "path": relative,
            "name": instance_path.name,
            "format": instance_path.suffix.lower().lstrip("."),
            "partition": partition,
            "sha256": checksum,
        }
        try:
            shape = inspect_instance(instance_path)
            record.update(
                {
                    "variables": len(shape.variables),
                    "binary_variables": shape.binary_variables,
                    "constraints": shape.constraints,
                }
            )
            if shape.binary_variables != len(shape.variables):
                result.issues.append(
                    AuditIssue(
                        "non-binary-variables",
                        "error",
                        relative,
                        f"{len(shape.variables) - shape.binary_variables} variables are not binary",
                    )
                )
        except (DatasetAuditError, ValueError) as exc:
            result.issues.append(AuditIssue("invalid-instance", "error", relative, str(exc)))
            result.files.append(record)
            continue

        backbone_path = backbone_dir / f"{instance_path.name}.backbone"
        required = partition in required_backbone_partitions
        record["backbone_required"] = required
        record["backbone_path"] = backbone_path.relative_to(root).as_posix()
        if backbone_path.exists():
            record["backbone_sha256"] = sha256_file(backbone_path)
            try:
                labels = read_backbone(backbone_path, shape.variables)
                counts = Counter(label.name for label in labels.values())
                record["labels"] = {name: counts.get(name, 0) for name in ("B0", "B1", "NB")}
                record["backbone_complete"] = True
            except (BackboneFormatError, UnicodeError) as exc:
                record["backbone_complete"] = False
                result.issues.append(
                    AuditIssue("invalid-backbone", "error", record["backbone_path"], str(exc))
                )
        else:
            record["backbone_complete"] = False
            if required:
                result.issues.append(
                    AuditIssue(
                        "missing-backbone",
                        "error",
                        record["backbone_path"],
                        f"Backbone required for partition {partition!r}",
                    )
                )
        result.files.append(record)

    for checksum, names in sorted(checksum_names.items()):
        if len(names) > 1:
            result.issues.append(
                AuditIssue(
                    "duplicate-instance",
                    "error",
                    ",".join(sorted(names)),
                    f"Files have the same SHA-256: {checksum}",
                )
            )

    expected_backbones = {f"{name}.backbone" for name in instance_names}
    for backbone_path in sorted(backbone_dir.glob("*.backbone")) if backbone_dir.is_dir() else []:
        if backbone_path.name not in expected_backbones:
            result.issues.append(
                AuditIssue(
                    "orphan-backbone",
                    "error",
                    backbone_path.relative_to(root).as_posix(),
                    "No matching instance exists",
                )
            )

    result.issues.sort(key=lambda issue: (issue.severity, issue.code, issue.path, issue.message))
    result.summary = _summarize(result)
    return result


def _summarize(result: AuditResult) -> dict[str, Any]:
    by_partition: dict[str, dict[str, Any]] = {}
    for partition in result.partitions:
        records = [record for record in result.files if record.get("partition") == partition]
        label_counts = Counter()
        for record in records:
            label_counts.update(record.get("labels", {}))
        by_partition[partition] = {
            "instances": len(records),
            "complete_backbones": sum(bool(record.get("backbone_complete")) for record in records),
            "variables": sum(int(record.get("variables", 0)) for record in records),
            "binary_variables": sum(int(record.get("binary_variables", 0)) for record in records),
            "constraints": sum(int(record.get("constraints", 0)) for record in records),
            "labels": {name: label_counts.get(name, 0) for name in ("B0", "B1", "NB")},
        }
    return {
        "instances": len(result.files),
        "complete_backbones": sum(bool(record.get("backbone_complete")) for record in result.files),
        "errors": sum(issue.severity == "error" for issue in result.issues),
        "warnings": sum(issue.severity == "warning" for issue in result.issues),
        "by_partition": by_partition,
    }


def write_audit_outputs(result: AuditResult, manifest_path: Path, report_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_text = (
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    _atomic_write(manifest_path, manifest_text)

    lines = [
        f"# {result.dataset} dataset audit",
        "",
        f"Status: **{'PASS' if result.valid else 'FAIL'}**",
        "",
        "## Summary",
        "",
        f"- Instances: {result.summary.get('instances', 0)}",
        f"- Complete backbones: {result.summary.get('complete_backbones', 0)}",
        f"- Errors: {result.summary.get('errors', 0)}",
        f"- Warnings: {result.summary.get('warnings', 0)}",
        "",
        "| Partition | Instances | Backbones | B0 | B1 | NB |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in result.summary.get("by_partition", {}).items():
        labels = summary["labels"]
        lines.append(
            f"| {name} | {summary['instances']} | {summary['complete_backbones']} | "
            f"{labels['B0']} | {labels['B1']} | {labels['NB']} |"
        )
    lines.extend(["", "## Issues", ""])
    if result.issues:
        lines.extend(
            f"- `{issue.severity.upper()}` `{issue.code}` `{issue.path}`: {issue.message}"
            for issue in result.issues
        )
    else:
        lines.append("No issues found.")
    _atomic_write(report_path, "\n".join(lines) + "\n")


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)
