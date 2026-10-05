#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset="$repo_root/data/backpas/MVC"
work_dir="$repo_root/results/MVC/baseline/seed-0"
epochs=200
seed=0
threads=4
resume=0
device="cuda"
detach=0

usage() {
  echo "Usage: $0 [--dataset PATH] [--work-dir PATH] [--epochs N] [--seed N] [--threads N] [--resume] [--cpu] [--detach]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) dataset="$2"; shift 2 ;;
    --work-dir) work_dir="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --threads) threads="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    --cpu) device="cpu"; shift ;;
    --detach) detach=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

dataset="$(realpath "$dataset")"
mkdir -p "$work_dir"
work_dir="$(realpath "$work_dir")"

if [[ "$detach" -eq 1 ]]; then
  launch_log="$work_dir/server-launch.log"
  forwarded=(--dataset "$dataset" --work-dir "$work_dir" --epochs "$epochs" --seed "$seed" --threads "$threads")
  [[ "$resume" -eq 1 ]] && forwarded+=(--resume)
  [[ "$device" == "cpu" ]] && forwarded+=(--cpu)
  nohup "$0" "${forwarded[@]}" >"$launch_log" 2>&1 &
  echo $! >"$work_dir/training.pid"
  echo "Training started as PID $(cat "$work_dir/training.pid"). Log: $launch_log"
  exit 0
fi

"$repo_root/scripts/preflight_gpu_server.sh" "$dataset" "$work_dir"

micromamba="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
arguments=(
  train-backpas "$dataset"
  --work-dir "$work_dir"
  --epochs "$epochs"
  --seed "$seed"
  --threads "$threads"
  --runtime native
)
[[ "$resume" -eq 1 ]] && arguments+=(--resume)
[[ "$device" == "cuda" ]] && arguments+=(--cuda)

"$micromamba" run -n pbo_orchestrator backbone-pbo "${arguments[@]}"

