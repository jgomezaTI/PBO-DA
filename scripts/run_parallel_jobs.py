"""Launch independent training jobs concurrently.

Each job gets its own work directory. This is intended for separate Colab
sessions or a multi-GPU machine; do not use it to run multiple CUDA jobs on a
single small GPU unless that contention is intentional.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_one(repo: Path, job: str, drive_root: str | None, cpu: bool, resume: bool,
            matrix: Path) -> tuple[str, int]:
    command = [sys.executable, "scripts/cloud_run.py", "--repo", str(repo),
               "--matrix", str(matrix), "--job", job]
    if drive_root:
        command += ["--drive-root", drive_root]
    if cpu:
        command.append("--cpu")
    if resume:
        command.append("--resume")
    completed = subprocess.run(command, cwd=repo, check=False)
    return job, completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", nargs="*", help="PROBLEM/TECHNIQUE/seed-N identifiers")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--drive-root")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Request continuation; checkpoints are also detected automatically")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--matrix", type=Path, default=Path("configs/cloud_jobs.json"))
    parser.add_argument("--one-per-technique", action="store_true",
                        help="Select one job for each of the four mapped techniques")
    parser.add_argument("--problems", nargs="+", choices=("MVC", "MIS", "CA"),
                        help="Problem per technique in matrix order")
    args = parser.parse_args()
    repo = args.repo.resolve()
    matrix_path = args.matrix.resolve() if args.matrix.is_absolute() else (repo / args.matrix).resolve()
    if args.one_per_technique and args.matrix == Path("configs/cloud_jobs.json"):
        pending_matrix = repo / "configs/pending_augmentation_jobs.json"
        if pending_matrix.is_file():
            matrix_path = pending_matrix
    jobs = list(args.jobs)
    if args.one_per_technique:
        if jobs:
            parser.error("pass explicit jobs or --one-per-technique, not both")
        if not args.problems or len(args.problems) != 4:
            parser.error("--one-per-technique requires exactly four --problems values")
        matrix_data = json.loads(matrix_path.read_text(encoding="utf-8"))
        techniques = list(dict.fromkeys(row["technique"] for row in matrix_data["jobs"]))
        if len(techniques) != 4:
            parser.error(f"expected four techniques in matrix, found {techniques}")
        jobs = []
        for technique, problem in zip(techniques, args.problems):
            row = next((row for row in matrix_data["jobs"]
                        if row["technique"] == technique and row["problem"] == problem), None)
            if row is None:
                parser.error(f"matrix has no job for {problem}/{technique}")
            jobs.append(f"{problem}/{technique}/seed-{row['seed']}")
        if len(set(jobs)) != 4:
            parser.error("the four technique sessions must use four distinct jobs")
    elif not jobs:
        parser.error("provide explicit job IDs or use --one-per-technique")
    workers = args.workers or len(jobs)
    if workers < len(jobs):
        parser.error(f"workers={workers} would serialize jobs; use at least {len(jobs)} workers")
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, repo, job, args.drive_root, args.cpu,
                               args.resume, matrix_path)
                   for job in jobs]
        for future in as_completed(futures):
            job, code = future.result()
            print(f"{job}: {'completed' if code == 0 else f'failed ({code})'}", flush=True)
            if code != 0:
                failures.append(job)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
