#!/usr/bin/env python3
"""Run one reproducible BackPaS job from a persistent local/Drive workspace.

The workspace can be a Google Drive mount in Colab. Only the selected job is
executed, so multiple Colab sessions can run different matrix rows safely.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    defaults, jobs = read_jobs((repo / args.matrix).resolve() if not args.matrix.is_absolute() else args.matrix)
    job = find_job(jobs, args.job)
    epochs = args.epochs or defaults.get("epochs", 200)
    threads = args.threads or defaults.get("threads", 2)
    work = repo / "results" / job["problem"] / job["technique"] / f"seed-{job['seed']}"
    if args.drive_root:
        # Write checkpoints directly to Drive so an interrupted Colab runtime
        # still leaves the latest epoch available for --resume.
        work = args.drive_root / "runs" / job["problem"] / job["technique"] / f"seed-{job['seed']}"
        work.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    completed_markers = (work / "metrics.json", work / "experiment-status.json")
    if not args.allow_completed and any(marker.exists() for marker in completed_markers):
        raise SystemExit(
            f"Refusing to rerun completed job {args.job}; existing result found in {work}. "
            "Use --allow-completed only for an intentional reproduction."
        )
    status = work / "status.json"
    command = [sys.executable, "-m", "backbone_pbo.cli", "train-backpas", job["dataset"],
               "--work-dir", str(work), "--epochs", str(epochs), "--seed", str(job["seed"]),
               "--threads", str(threads), "--batch-size", str(defaults.get("batch_size", 8)),
               "--batch-accumulation-size", str(defaults.get("batch_accumulation_size", 32)),
               "--runtime", "native"]
    if not args.cpu and defaults.get("cuda", True):
        command.append("--cuda")
    if args.resume:
        command.append("--resume")
    print(" ".join(command))
    if args.dry_run:
        return 0
    write_status(status, job=args.job, state="running", command=command)
    result = subprocess.run(command, cwd=repo, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    write_status(status, job=args.job, state="completed" if result.returncode == 0 else "failed",
                 return_code=result.returncode)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
