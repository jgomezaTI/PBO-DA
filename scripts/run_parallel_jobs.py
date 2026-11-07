"""Launch independent training jobs concurrently.

Each job gets its own work directory. This is intended for separate Colab
sessions or a multi-GPU machine; do not use it to run multiple CUDA jobs on a
single small GPU unless that contention is intentional.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_one(repo: Path, job: str, drive_root: str | None, cpu: bool, resume: bool) -> tuple[str, int]:
    command = [sys.executable, "scripts/cloud_run.py", "--repo", str(repo), "--job", job]
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
    parser.add_argument("jobs", nargs="+", help="PROBLEM/TECHNIQUE/seed-N identifiers")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--drive-root")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    workers = args.workers or len(args.jobs)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, args.repo.resolve(), job, args.drive_root, args.cpu, args.resume)
                   for job in args.jobs]
        for future in as_completed(futures):
            job, code = future.result()
            print(f"{job}: {'completed' if code == 0 else f'failed ({code})'}", flush=True)
            if code != 0:
                failures.append(job)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
