"""WSL orchestration for the unmodified GuroBack and BackPaS checkouts."""

from __future__ import annotations

import csv
import json
import os
import pickle
import platform
import shutil
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backbone_pbo.backbone import BackboneFormatError, dumps_backbone, loads_backbone
from backbone_pbo.dataset.audit import inspect_instance, load_partitions, sha256_file


class BackPaSError(RuntimeError):
    """An upstream extraction, graph conversion, or training command failed."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_runtime() -> str:
    """Use WSL from Windows and execute tools directly on Linux."""

    return "wsl" if platform.system() == "Windows" else "native"


def ensure_free_space(path: Path, minimum_gib: int = 20) -> None:
    free = shutil.disk_usage(path.resolve()).free
    required = minimum_gib * 1024**3
    if free < required:
        raise BackPaSError(
            f"At least {minimum_gib} GiB must be free; only {free / 1024**3:.1f} GiB remain."
        )


def windows_to_wsl(path: Path, *, distribution: str = "Ubuntu") -> str:
    portable_path = str(path.resolve()).replace("\\", "/")
    completed = subprocess.run(
        ["wsl", "--distribution", distribution, "--", "wslpath", "-a", portable_path],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise BackPaSError(f"Unable to translate path for WSL: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _selected_instances(
    dataset_dir: Path,
    partition_names: Iterable[str],
    limits: dict[str, int] | None = None,
) -> list[tuple[str, str]]:
    partitions = load_partitions(dataset_dir / "partitions.json")
    selected: list[tuple[str, str]] = []
    for partition in partition_names:
        if partition not in partitions:
            raise BackPaSError(f"Unknown partition {partition!r}")
        names = partitions[partition]
        if limits and partition in limits:
            names = names[: limits[partition]]
        selected.extend((partition, name) for name in names)
    return selected


def normalize_guroback_output(content: str, variables: Iterable[str]) -> str:
    """Map GuroBack labels back to exact source variable names."""

    variable_set = set(variables)
    normalized: list[str] = []
    seen: set[str] = set()
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines or lines[-1] != "b 0":
        raise BackboneFormatError("Missing final 'b 0' marker")
    for line_number, line in enumerate(lines[:-1], start=1):
        parts = line.split()
        if len(parts) != 2 or parts[0] != "b":
            raise BackboneFormatError(f"Invalid line {line_number}: {line!r}")
        raw = parts[1]
        negative = raw.startswith("-")
        name = raw[1:] if negative else raw
        # Upstream GuroBack unconditionally removes the first character of the
        # solver variable name. Reconstruct it from the authoritative instance
        # instead of assuming that every dataset uses the ``x`` prefix.
        candidates = {
            candidate for candidate in variable_set if candidate == name or candidate[1:] == name
        }
        if len(candidates) != 1:
            raise BackboneFormatError(
                f"GuroBack variable {name!r} has {len(candidates)} matches in the instance"
            )
        mapped = candidates.pop()
        if mapped in seen:
            raise BackboneFormatError(f"Duplicate GuroBack label for {mapped!r}")
        seen.add(mapped)
        normalized.append(f"b {'-' if negative else ''}{mapped}")
    normalized.append("b 0")
    canonical = "\n".join(normalized) + "\n"
    labels = loads_backbone(canonical, variable_set)
    return dumps_backbone(labels)


def extract_backbones(
    dataset_dir: str | Path,
    *,
    partitions: tuple[str, ...] = ("train", "validation"),
    threads: int = 4,
    distribution: str = "Ubuntu",
    runtime: str | None = None,
    binary_path: str | Path | None = None,
    limits: dict[str, int] | None = None,
) -> dict[str, int]:
    root = Path(dataset_dir).resolve()
    runtime = runtime or default_runtime()
    ensure_free_space(root)
    runtime_directory = "wsl" if runtime == "wsl" else "linux"
    binary = (
        Path(binary_path)
        if binary_path
        else repository_root() / f".runtime/{runtime_directory}/bin/guroback"
    )
    if not binary.exists():
        raise BackPaSError(
            f"GuroBack binary not found at {binary}. Run scripts/setup_wsl_runtime.ps1 first."
        )
    backbone_dir = root / "backbone"
    log_dir = root / "backbone_extraction_log"
    backbone_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    executable = _runtime_path(binary, runtime=runtime, distribution=distribution)
    stats = {"selected": 0, "extracted": 0, "skipped": 0, "failed": 0}

    for partition, name in _selected_instances(root, partitions, limits):
        stats["selected"] += 1
        instance = root / "instance" / name
        destination = backbone_dir / f"{name}.backbone"
        shape = inspect_instance(instance)
        if destination.exists():
            try:
                loads_backbone(destination.read_text(encoding="ascii"), shape.variables)
                stats["skipped"] += 1
                continue
            except (BackboneFormatError, UnicodeError):
                pass

        partial = destination.with_name(f".{destination.name}.partial")
        partial.unlink(missing_ok=True)
        tool_arguments = [
            executable,
            f"Threads={threads}",
            "FeasibilityTol=1e-9",
            "OptimalityTol=1e-9",
            "IntFeasTol=1e-9",
            _runtime_path(instance, runtime=runtime, distribution=distribution),
            _runtime_path(partial, runtime=runtime, distribution=distribution),
        ]
        command = _runtime_command(tool_arguments, runtime=runtime, distribution=distribution)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=_runtime_environment(runtime),
        )
        log_payload = {
            "instance": name,
            "partition": partition,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        (log_dir / f"{name}.json").write_text(
            json.dumps(log_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            if completed.returncode != 0:
                raise BackPaSError(completed.stderr.strip() or completed.stdout.strip())
            if not partial.exists():
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise BackPaSError(detail or "GuroBack produced no backbone file")
            canonical = normalize_guroback_output(
                partial.read_text(encoding="ascii"), shape.variables
            )
            partial.write_text(canonical, encoding="ascii", newline="\n")
            partial.replace(destination)
            stats["extracted"] += 1
        except (OSError, BackboneFormatError, BackPaSError) as exc:
            stats["failed"] += 1
            partial.unlink(missing_ok=True)
            raise BackPaSError(f"Backbone extraction failed for {name}: {exc}") from exc
    return stats


def prepare_backpas(
    dataset_dir: str | Path,
    work_dir: str | Path,
    *,
    graph_type: str = "literals",
    distribution: str = "Ubuntu",
    runtime: str | None = None,
    limits: dict[str, int] | None = None,
) -> dict[str, int]:
    runtime = runtime or default_runtime()
    if graph_type not in {"literals", "variables"}:
        raise BackPaSError(f"Unsupported graph type: {graph_type}")
    root = Path(dataset_dir).resolve()
    work = Path(work_dir).resolve()
    ensure_free_space(root)
    selected = _selected_instances(root, ("train", "validation"), limits)
    selected_names = {name for _, name in selected}
    missing_backbones = [
        name
        for name in sorted(selected_names)
        if not (root / "backbone" / f"{name}.backbone").exists()
    ]
    if missing_backbones:
        raise BackPaSError(f"Missing {len(missing_backbones)} selected backbones")

    backpas = repository_root() / "external/backpas"
    if not backpas.is_dir():
        raise BackPaSError("external/backpas is missing; run scripts/setup_upstreams.ps1")
    script = backpas / "src/1_create_ml_dataset.py"
    script_runtime = _runtime_path(script, runtime=runtime, distribution=distribution)
    root_runtime = _runtime_path(root, runtime=runtime, distribution=distribution)
    arguments = (
        f"python {shell_quote(script_runtime)} "
        f"--dataset_path {shell_quote(root_runtime)} "
        f"--graph_type {shell_quote(graph_type)}"
    )
    command = _environment_command(arguments, runtime=runtime, distribution=distribution)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise BackPaSError(completed.stderr.strip() or completed.stdout.strip())

    graph_dir = root / f"ml_dataset_{graph_type}"
    missing_graphs = [
        name for name in sorted(selected_names) if not (graph_dir / f"{name}.pkl").exists()
    ]
    if missing_graphs:
        message = (
            f"BackPaS silently omitted {len(missing_graphs)} selected graphs; "
            f"first: {missing_graphs[0]}"
        )
        raise BackPaSError(message)
    work.mkdir(parents=True, exist_ok=True)
    ml_partitions: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    for partition, name in selected:
        ml_partitions["valid" if partition == "validation" else partition].append(name)
    partition_payload = {
        "ml_partitions": ml_partitions,
        "trust_regions_partitions": {"valid": [], "test": []},
    }
    with (work / "partitions.pkl").open("wb") as stream:
        pickle.dump(partition_payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (work / "partitions.json").write_text(
        json.dumps(partition_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "train": len(ml_partitions["train"]),
        "validation": len(ml_partitions["valid"]),
        "graphs": len(selected_names),
    }


def train_backpas(
    dataset_dir: str | Path,
    work_dir: str | Path,
    *,
    epochs: int = 200,
    seed: int = 0,
    threads: int = 4,
    resume: bool = False,
    distribution: str = "Ubuntu",
    runtime: str | None = None,
    use_cuda: bool = False,
    batch_size: int = 8,
    batch_accumulation_size: int = 32,
) -> dict[str, Any]:
    runtime = runtime or default_runtime()
    root = Path(dataset_dir).resolve()
    work = Path(work_dir).resolve()
    ensure_free_space(root)
    runner = repository_root() / "scripts/run_backpas_seeded.py"
    backpas_train = repository_root() / "external/backpas/src/3_train.py"
    graph_dir = root / "ml_dataset_literals"
    for required in (runner, backpas_train, graph_dir, work / "partitions.pkl"):
        if not required.exists():
            raise BackPaSError(f"Required baseline input is missing: {required}")

    runner_runtime = _runtime_path(runner, runtime=runtime, distribution=distribution)
    trainer_runtime = _runtime_path(backpas_train, runtime=runtime, distribution=distribution)
    graph_runtime = _runtime_path(graph_dir, runtime=runtime, distribution=distribution)
    work_runtime = _runtime_path(work, runtime=runtime, distribution=distribution)
    arguments = (
        f"python {shell_quote(runner_runtime)} "
        f"--script {shell_quote(trainer_runtime)} "
        f"--seed {seed} --threads {threads} -- "
        f"--ml_dataset_path {shell_quote(graph_runtime)} "
        f"--dataset_wkdir_path {shell_quote(work_runtime)} "
        f"--graph_type literals --layer_type GTR --num_layers 8 --epochs {epochs} "
        f"--learning_rate 0.001 --batch_size {batch_size} "
        f"--batch_accumulation_size {batch_accumulation_size}"
    )
    if resume:
        arguments += " --continue_training_from_last_epoch"
    if use_cuda:
        arguments += " --use_cuda"
    command = _environment_command(arguments, runtime=runtime, distribution=distribution)
    started = datetime.now(UTC).isoformat()
    process_log = work / "training-process.log"
    with process_log.open("a" if resume else "w", encoding="utf-8", newline="\n") as stream:
        completed = subprocess.run(
            command,
            check=False,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        detail = _tail_text(process_log, maximum_characters=4000)
        raise BackPaSError(f"BackPaS training failed; see {process_log}\n{detail}")
    return collect_training_results(
        root,
        work,
        seed=seed,
        epochs=epochs,
        threads=threads,
        started=started,
        runtime=runtime,
        device="cuda" if use_cuda else "cpu",
        batch_size=batch_size,
        batch_accumulation_size=batch_accumulation_size,
    )


def collect_training_results(
    dataset_dir: Path,
    work_dir: Path,
    *,
    seed: int,
    epochs: int,
    threads: int,
    started: str,
    runtime: str = "unknown",
    device: str = "cpu",
    batch_size: int = 8,
    batch_accumulation_size: int = 32,
) -> dict[str, Any]:
    nested = work_dir / "ml_training/graph_with_literals_8_GTR"
    source_log = nested / "training_log.csv"
    if not source_log.exists():
        raise BackPaSError("BackPaS did not create training_log.csv")
    for source_name, destination_name in (
        ("training_log.csv", "training-log.csv"),
        ("best_model.pth", "best-model.pth"),
        ("last_model.pth", "last-model.pth"),
        ("last_optimizer.pth", "last-optimizer.pth"),
    ):
        source = nested / source_name
        if source.exists():
            shutil.copy2(source, work_dir / destination_name)

    with source_log.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    validation = [row for row in rows if row.get("partition") == "valid"]
    if not validation:
        raise BackPaSError("Training log contains no validation rows")
    best = min(validation, key=lambda row: float(row["loss"]))
    class_names = ("B0", "B1", "NB")
    metrics: dict[str, Any] = {
        "selected_by": "minimum validation cross-entropy",
        "epoch": int(best["epoch"]),
        "loss": float(best["loss"]),
        "accuracy": float(best["accuracy_micro"]),
        "balanced_accuracy": float(best["accuracy_macro"]),
        "macro_f1": float(best["f1_score_macro"]),
        "class_metrics": {},
    }
    for class_name in class_names:
        metrics["class_metrics"][class_name] = {
            metric: float(best[f"multiclass_{metric}_{class_name}"])
            for metric in ("precision", "recall", "f1_score")
        }
    confusion = [
        [float(best[f"confusion_matrix_cm_{row}_{column}"]) for column in class_names]
        for row in class_names
    ]
    metrics["confusion_matrix_normalized"] = confusion
    _write_json(work_dir / "metrics.json", metrics)
    with (work_dir / "confusion-matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual/predicted", *class_names])
        for name, values in zip(class_names, confusion, strict=True):
            writer.writerow([name, *values])

    problem = next((name for name in ("MVC", "MIS", "CA")
                    if dataset_dir.name == name or dataset_dir.name.startswith(f"{name}-")),
                   dataset_dir.name)
    technique = work_dir.parent.name
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "started_at": started,
        "completed_at": datetime.now(UTC).isoformat(),
        "dataset": problem,
        "technique": technique,
        "dataset_path": dataset_dir.as_posix(),
        "partitions_sha256": sha256_file(dataset_dir / "partitions.json"),
        "seed": seed,
        "epochs": epochs,
        "cpu_threads": threads,
        "runtime": runtime,
        "device": device,
        "configuration": {
            "graph_type": "literals",
            "layer_type": "GTR",
            "num_layers": 8,
            "learning_rate": 0.001,
            "batch_size": batch_size,
            "batch_accumulation_size": batch_accumulation_size,
            "literal_messages": False,
        },
        "upstream_commits": {
            "backpas": _git_commit(repository_root() / "external/backpas"),
            "guroback": _git_commit(repository_root() / "external/guroback"),
        },
    }
    _write_json(work_dir / "run-manifest.json", manifest)
    report = (
        f"# {problem} {technique} training\n\n"
        f"- Selected epoch: {metrics['epoch']}\n"
        f"- Validation cross-entropy: {metrics['loss']:.8f}\n"
        f"- Validation accuracy: {metrics['accuracy']:.8f}\n"
        f"- Validation balanced accuracy: {metrics['balanced_accuracy']:.8f}\n"
        f"- Validation macro-F1: {metrics['macro_f1']:.8f}\n"
    )
    report_name = "baseline-report.md" if technique in {"baseline", "full"} else "experiment-report.md"
    (work_dir / report_name).write_text(report, encoding="utf-8", newline="\n")
    return metrics


def _micromamba_command(command: str) -> str:
    return (
        'export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"; '
        '"$HOME/.local/bin/micromamba" run -n pbo_backbones ' + command
    )


def _runtime_path(path: Path, *, runtime: str, distribution: str) -> str:
    if runtime == "wsl":
        return windows_to_wsl(path, distribution=distribution)
    if runtime in {"native", "colab"}:
        return path.resolve().as_posix()
    raise BackPaSError(f"Unsupported runtime: {runtime}")


def _runtime_command(arguments: list[str], *, runtime: str, distribution: str) -> list[str]:
    if runtime == "wsl":
        return ["wsl", "--distribution", distribution, "--", *arguments]
    if runtime in {"native", "colab"}:
        return arguments
    raise BackPaSError(f"Unsupported runtime: {runtime}")


def _runtime_environment(runtime: str) -> dict[str, str]:
    """Forward a Windows Gurobi license file path into WSL when extracting."""

    environment = os.environ.copy()
    license_path = environment.get("GRB_LICENSE_FILE")
    if runtime != "wsl" or not license_path:
        return environment

    entries = [
        entry
        for entry in environment.get("WSLENV", "").split(":")
        if entry and entry.split("/", 1)[0] != "GRB_LICENSE_FILE"
    ]
    entries.append("GRB_LICENSE_FILE/p")
    environment["WSLENV"] = ":".join(entries)
    return environment


def _environment_command(command: str, *, runtime: str, distribution: str) -> list[str]:
    if runtime == "colab":
        return ["bash", "-lc", command]
    shell_command = ["bash", "-lc", _micromamba_command(command)]
    return _runtime_command(shell_command, runtime=runtime, distribution=distribution)


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _tail_text(path: Path, *, maximum_characters: int) -> str:
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - maximum_characters))
        return stream.read().decode("utf-8", errors="replace")


def _git_commit(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    return completed.stdout.strip() if completed.returncode == 0 else None
