#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="${1:-$repo_root/results/MVC/baseline/seed-0}"
training_dir="$work_dir/ml_training/graph_with_literals_8_GTR"
log="$training_dir/training_log.csv"

if [[ -f "$work_dir/training.pid" ]]; then
  pid="$(cat "$work_dir/training.pid")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "process=running pid=$pid"
  else
    echo "process=stopped pid=$pid"
  fi
else
  echo "process=unknown (no PID file)"
fi

if [[ -f "$log" ]]; then
  total_rows="$(( $(wc -l < "$log") - 1 ))"
  train_rows="$(awk -F, 'NR > 1 && $3 == "train" {count++} END {print count+0}' "$log")"
  valid_rows="$(awk -F, 'NR > 1 && $3 == "valid" {count++} END {print count+0}' "$log")"
  echo "records=$total_rows train_epochs=$train_rows validation_epochs=$valid_rows"
  tail -n 4 "$log"
else
  echo "training_log=missing"
fi

[[ -f "$work_dir/server-launch.log" ]] && tail -n 20 "$work_dir/server-launch.log"

