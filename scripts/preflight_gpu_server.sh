#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
dataset="${1:-$repo_root/data/backpas/MVC}"
work_dir="${2:-$repo_root/results/MVC/baseline/seed-0}"
micromamba="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || fail "This preflight must run on native Linux."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable."
[[ -x "$micromamba" ]] || fail "Micromamba is missing; run scripts/setup_gpu_server.sh."
[[ -d "$repo_root/external/backpas/.git" ]] || fail "The BackPaS checkout is missing."
[[ -f "$dataset/partitions.json" ]] || fail "Dataset partitions are missing: $dataset"
[[ -d "$dataset/ml_dataset_literals" ]] || fail "Literal graphs are missing: $dataset/ml_dataset_literals"
[[ -f "$work_dir/partitions.pkl" ]] || fail "BackPaS partitions are missing: $work_dir/partitions.pkl"

free_kib="$(df -Pk "$repo_root" | awk 'NR==2 {print $4}')"
(( free_kib >= 20 * 1024 * 1024 )) || fail "At least 20 GiB of free disk space is required."

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$micromamba" run -n pbo_backbones python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the pbo_backbones environment")
print(f"gpu={torch.cuda.get_device_name(0)}")
tensor = torch.ones(1, device="cuda")
print(f"cuda_smoke={tensor.item()}")
PY

"$micromamba" run -n pbo_orchestrator backbone-pbo audit-dataset "$dataset"
"$micromamba" run -n pbo_orchestrator python - "$dataset" "$work_dir" <<'PY'
import json
import pickle
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
work = Path(sys.argv[2])
source = json.loads((dataset / "partitions.json").read_text(encoding="utf-8"))["partitions"]
with (work / "partitions.pkl").open("rb") as stream:
    prepared = pickle.load(stream)["ml_partitions"]
expected = source["train"] + source["validation"]
missing = [name for name in expected if not (dataset / "ml_dataset_literals" / f"{name}.pkl").is_file()]
if missing:
    raise SystemExit(f"Missing {len(missing)} literal graphs; first: {missing[0]}")
if prepared["train"] != source["train"] or prepared["valid"] != source["validation"]:
    raise SystemExit("Prepared BackPaS partitions do not match authoritative partitions.json")
print(f"training_instances={len(prepared['train'])}")
print(f"validation_instances={len(prepared['valid'])}")
print(f"literal_graphs={len(expected)}")
PY

if [[ -f "$HOME/gurobi.lic" ]]; then
  echo "gurobi_license=$HOME/gurobi.lic"
else
  echo "gurobi_license=not-present (not required when backbones are already generated)"
fi

echo "GPU server preflight passed."
