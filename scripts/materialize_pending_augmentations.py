"""Materialize all mapped but unrun augmentation families for each benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from materialize_perturbation import materialize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=20)
    args = parser.parse_args()
    repo = args.repo.resolve()
    jobs = []
    for problem in ("MVC", "MIS", "CA"):
        source = repo / "data" / "backpas" / problem
        for kind, technique in (("selective-polarity", "selective-polarity"),
                                ("constraint", "constraint-perturbation"),
                                ("objective", "objective-perturbation"),
                                ("synthetic", "synthetic-conditioned")):
            output = repo / "data" / "backpas" / f"{problem}-{technique}-v1"
            # Dataset creation is deferred to the selected Colab job. The
            # matrix stores the generator recipe alongside its training input.
            jobs.append({"problem": problem, "technique": technique, "seed": args.seed,
                         "dataset": str(output.relative_to(repo)).replace("\\", "/"),
                         "source_dataset": str(source.relative_to(repo)).replace("\\", "/"),
                         "augmentation_kind": kind, "replicas": args.replicas,
                         "attempts": args.attempts})
    source_matrix = json.loads((repo / "configs" / "cloud_jobs.json").read_text(encoding="utf-8"))
    pending_matrix = {"version": source_matrix["version"], "repository": source_matrix.get("repository"),
                      "defaults": source_matrix["defaults"], "jobs": jobs}
    path = repo / "configs" / "pending_augmentation_jobs.json"
    path.write_text(json.dumps(pending_matrix, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"jobs": len(jobs), "matrix": str(path), "generation": "on-demand in each Colab session"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
