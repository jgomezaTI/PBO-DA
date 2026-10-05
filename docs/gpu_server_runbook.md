# Native Linux GPU training runbook

This runbook prepares a temporary university GPU machine without committing data,
credentials, Gurobi licenses, graphs, checkpoints, or experiment results.

## What must be transferred

Cloning the repository is not enough because experimental artifacts are ignored by
Git. On the Windows workstation, create the payload:

```powershell
.\scripts\export_training_payload.ps1
```

Copy `artifacts/mvc-training-payload.tar.gz` to the server through the institution's
approved transfer mechanism together with its `.sha256` file. Verify and extract it
at the root of the cloned repository:

```bash
(cd /path/to && sha256sum -c mvc-training-payload.tar.gz.sha256)
tar -xzf /path/to/mvc-training-payload.tar.gz
```

The payload contains the authoritative dataset partitions, instances, backbones,
literal graphs, and the baseline work directory. It may also contain resumable
checkpoints if they exist locally. Treat it as research data rather than source code.

## One-time server setup

The setup requires Linux x86-64, Git, curl, tar, an NVIDIA driver visible through
`nvidia-smi`, internet access, and enough local disk space. It installs Micromamba
under the current user's home, checks out the pinned BackPaS/GuroBack revisions, and
creates the historical BackPaS environment without modifying either upstream.

```bash
bash scripts/setup_gpu_server.sh
bash scripts/preflight_gpu_server.sh
```

The historical environment is intentionally preserved. If its PyTorch 1.10.2 and
CUDA 11.3 stack cannot execute on the server GPU, stop and record the incompatibility
before creating a modernized environment; do not silently change the baseline.

Gurobi is not required to train from the transferred graphs and backbones. A Gurobi
license is needed only if labels must be regenerated on the server. Never copy a
license into the repository.

## Smoke test

Use a separate work directory prepared with the smoke partitions before reserving a
multi-day window. The current full payload can be checked without training by the
preflight command. When a smoke partition exists:

```bash
bash scripts/run_backpas_training.sh \
  --dataset data/backpas/MVC \
  --work-dir results/MVC/smoke/seed-0 \
  --epochs 1
```

## Full run and resume

Start the 200-epoch CUDA baseline detached from the SSH session:

```bash
bash scripts/run_backpas_training.sh --detach
```

Inspect progress:

```bash
bash scripts/show_backpas_training_status.sh
```

Resume from the last fully completed epoch after an interruption:

```bash
bash scripts/run_backpas_training.sh --resume --detach
```

The training log and last model/optimizer checkpoints are updated after every epoch.
Copy `results/` back to the workstation before relinquishing the server.
