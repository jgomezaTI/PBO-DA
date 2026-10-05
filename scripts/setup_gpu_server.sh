#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="$repo_root/.runtime/linux"
micromamba="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"

backpas_commit="b5c633579fa73d673f3c4e65e177c89bcddea3d7"
guroback_commit="428f7d112aa82a3b6b951793e1b9e6b712ef1cea"

for tool in git curl tar; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required system command: $tool" >&2
    echo "On Ubuntu, ask the administrator to run: sudo apt-get install git curl bzip2 build-essential" >&2
    exit 1
  fi
done

mkdir -p "$runtime_root/bin" "$HOME/.local/bin" "$repo_root/external"

if [[ ! -x "$micromamba" ]]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C "$runtime_root" bin/micromamba
  install -m 0755 "$runtime_root/bin/micromamba" "$micromamba"
fi

clone_pinned() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local destination="$repo_root/external/$name"

  if [[ ! -d "$destination/.git" ]]; then
    git clone "$url" "$destination"
  fi
  git -C "$destination" fetch --quiet origin "$commit"
  git -C "$destination" checkout --quiet --detach "$commit"
  test "$(git -C "$destination" rev-parse HEAD)" = "$commit"
}

clone_pinned backpas https://github.com/bryan-alvarado-ulloa/backpas.git "$backpas_commit"
clone_pinned guroback https://github.com/bryan-alvarado-ulloa/guroback.git "$guroback_commit"

if [[ ! -d "$MAMBA_ROOT_PREFIX/envs/pbo_orchestrator" ]]; then
  "$micromamba" create --yes -n pbo_orchestrator python=3.11 pip
fi
"$micromamba" run -n pbo_orchestrator python -m pip install -e "$repo_root[data,dev]"

if [[ ! -d "$MAMBA_ROOT_PREFIX/envs/pbo_backbones" ]]; then
  "$micromamba" create --yes -n pbo_backbones -f "$repo_root/external/backpas/environment.yml"
fi

echo "Server environments and pinned upstream repositories are ready."
echo "Next: copy the ignored training payload, then run scripts/preflight_gpu_server.sh."
