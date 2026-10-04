#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: setup_wsl_runtime.sh <repository-root>" >&2
  exit 2
fi

repo_root="$1"
runtime_root="$repo_root/.runtime/wsl"
gurobi_version="11.0.3"
gurobi_archive="$runtime_root/gurobi${gurobi_version}_linux64.tar.gz"
gurobi_root="$runtime_root/gurobi1103/linux64"

missing_system_tools=0
for tool in g++ make curl wget bzip2; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    missing_system_tools=1
  fi
done
if [[ "$missing_system_tools" -ne 0 ]]; then
  echo "Missing WSL build tools. Run scripts/setup_wsl_runtime.ps1 first." >&2
  exit 1
fi

mkdir -p "$runtime_root/bin" "$HOME/.local/bin"

if [[ ! -x "$HOME/.local/bin/micromamba" ]]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C "$runtime_root" bin/micromamba
  install -m 0755 "$runtime_root/bin/micromamba" "$HOME/.local/bin/micromamba"
fi

if [[ ! -d "$gurobi_root" ]]; then
  wget -O "$gurobi_archive" \
    "https://packages.gurobi.com/11.0/gurobi${gurobi_version}_linux64.tar.gz"
  tar -xzf "$gurobi_archive" -C "$runtime_root"
fi

g++ -std=c++11 -m64 -O2 -Wall -Wextra -Wpedantic -Werror \
  "$repo_root/external/guroback/guroback.cpp" \
  -I"$gurobi_root/include" -L"$gurobi_root/lib" \
  -Wl,-rpath,"$gurobi_root/lib" \
  -lgurobi_c++ -lgurobi110 -lm \
  -o "$runtime_root/bin/guroback"

export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
if ! "$HOME/.local/bin/micromamba" env list | grep -q '^pbo_backbones '; then
  "$HOME/.local/bin/micromamba" create --yes \
    -n pbo_backbones -f "$repo_root/external/backpas/environment.yml"
fi

echo "WSL runtime installed."
echo "Activate a Gurobi 11 academic license inside WSL before extraction."
