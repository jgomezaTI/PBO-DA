#!/usr/bin/env python3
"""Run one reproducible BackPaS job from a persistent local/Drive workspace.

The workspace can be a Google Drive mount in Colab. Only the selected job is
executed, so multiple Colab sessions can run different matrix rows safely.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def read_jobs(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("defaults", {}), payload["jobs"]


def find_job(jobs: list[dict], job_id: str) -> dict:
    for job in jobs:
        candidate = f"{job['problem']}/{job['technique']}/seed-{job['seed']}"
        if candidate == job_id:
            return job
    raise SystemExit(f"Unknown job {job_id!r}; expected one of: " + ", ".join(
        f"{j['problem']}/{j['technique']}/seed-{j['seed']}" for j in jobs
    ))


def write_status(path: Path, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(fields, updated_at=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--matrix", type=Path, default=Path("configs/cloud_jobs.json"))
    parser.add_argument("--job", required=True, help="PROBLEM/TECHNIQUE/seed-N")
    parser.add_argument("--drive-root", type=Path, help="Persistent Drive directory")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-completed", action="store_true",
                        help="Allow rerunning a job whose work directory already has final metrics")
    parser.add_argument("--allow-existing", action="store_true",
                        help="Permit use of a pre-existing local results directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    matrix_path = (repo / args.matrix).resolve() if not args.matrix.is_absolute() else args.matrix
    defaults, jobs = read_jobs(matrix_path)
    job = find_job(jobs, args.job)
    epochs = args.epochs or defaults.get("epochs", 200)
    threads = args.threads or defaults.get("threads", 2)
    work = repo / "results" / job["problem"] / job["technique"] / f"seed-{job['seed']}"
    if args.drive_root:
        # Write checkpoints directly to Drive so an interrupted Colab runtime
        # still leaves the latest epoch available for --resume.
        work = args.drive_root / "runs" / job["problem"] / job["technique"] / f"seed-{job['seed']}"
    completed_markers = (work / "metrics.json", work / "experiment-status.json")
    if not args.allow_completed and any(marker.exists() for marker in completed_markers):
        raise SystemExit(
            f"Refusing to rerun completed job {args.job}; existing result found in {work}. "
            "Use --allow-completed only for an intentional reproduction."
        )
    status = work / "status.json"
    # A stale "training" marker can remain after a Colab runtime is reclaimed.
    # A live process is identified by the caller/runtime, not this Drive file;
    # callers must ensure only one session starts this job at a time.
    dataset = (repo / job["dataset"]).resolve()
    if args.dry_run:
        print(f"job={args.job}")
        print(f"dataset={dataset}")
        print(f"work={work}")
        print(f"generation={job.get('augmentation_kind', 'existing dataset')}")
        return 0
    work.mkdir(parents=True, exist_ok=True)
    write_status(status, job=args.job, state="generating" if job.get("augmentation_kind") else "preparing")
    if job.get("augmentation_kind"):
        if dataset.exists():
            manifest_path = dataset / "augmentation.json"
            if not manifest_path.is_file():
                raise SystemExit(f"Dataset exists without a complete augmentation manifest: {dataset}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {"kind": job["augmentation_kind"], "seed": job["seed"],
                        "replicas_per_train_instance": job.get("replicas", 1),
                        "attempts_per_replica": job.get("attempts", 20)}
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise SystemExit(f"Existing generated dataset recipe does not match this job: {dataset}")
            print(f"Reusing validated generated dataset: {dataset}", flush=True)
        else:
            source = (repo / job["source_dataset"]).resolve()
            generator = [sys.executable, "scripts/materialize_perturbation.py", str(source), str(dataset),
                         "--kind", job["augmentation_kind"], "--seed", str(job["seed"]),
                         "--replicas", str(job.get("replicas", 1)), "--attempts", str(job.get("attempts", 20))]
            print("Generating dataset:", " ".join(generator), flush=True)
            generated = subprocess.run(generator, cwd=repo, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            if generated.returncode != 0:
                write_status(status, job=args.job, state="failed", stage="generation",
                             return_code=generated.returncode)
                return generated.returncode
    if not dataset.is_dir():
        raise SystemExit(f"Dataset directory is missing: {dataset}")

    # BackPaS needs graph caches and a fixed partitions.pkl under each job's
    # work directory. Build them on demand once for newly materialized data.
    write_status(status, job=args.job, state="preparing")
    graph_cache = dataset / "ml_dataset_literals"
    if not (work / "partitions.pkl").exists() or not graph_cache.is_dir():
        prepare = [sys.executable, "-m", "backbone_pbo.cli", "prepare-backpas", str(dataset),
                   "--work-dir", str(work), "--runtime", "colab"]
        print("Preparing BackPaS inputs:", " ".join(prepare), flush=True)
        prepared = subprocess.run(prepare, cwd=repo, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        if prepared.returncode != 0:
            write_status(status, job=args.job, state="failed", stage="preparation",
                         return_code=prepared.returncode)
            return prepared.returncode

    command = [sys.executable, "-m", "backbone_pbo.cli", "train-backpas", str(dataset),
               "--work-dir", str(work), "--epochs", str(epochs), "--seed", str(job["seed"]),
               "--threads", str(threads), "--batch-size", str(defaults.get("batch_size", 8)),
               "--batch-accumulation-size", str(defaults.get("batch_accumulation_size", 32)),
               "--runtime", "colab"]
    if not args.cpu and defaults.get("cuda", True):
        command.append("--cuda")
    has_checkpoint = (work / "last-model.pth").exists() or (work / "ml_training/graph_with_literals_8_GTR/last_model.pth").exists()
    if args.resume or has_checkpoint:
        command.append("--resume")
    print(" ".join(command))
    write_status(status, job=args.job, state="training", stage="training", command=command)
    result = subprocess.run(command, cwd=repo, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    write_status(status, job=args.job, state="completed" if result.returncode == 0 else "failed",
                 return_code=result.returncode)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
