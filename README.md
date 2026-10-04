# backbonePBO

TT1 project on data augmentation techniques to mitigate backbone class imbalance
in GNNs for Pseudo-Boolean Optimization (PBO).

The repository focuses on augmentation methods as the main contribution. BackPaS,
GuroBack, and related prior work may be used as infrastructure or baselines, but
they are not the project's primary scope.

## Experimental scope

The reference BackPaS study evaluates three binary PBO benchmarks:

| ID | Problem | Status in this project |
| --- | --- | --- |
| `MIS` | Maximum Independent Set | In scope |
| `MVC` | Minimum Vertex Cover | In scope |
| `CA` | Combinatorial Auctions | In scope |

The related ConPaS study also uses Item Placement (`IP`). BackPaS deliberately
excludes it because it contains non-binary variables, so it is not silently treated
as a fourth PBO benchmark here. A fourth problem family will be added only after its
identity and data source are confirmed.

The public BackPaS repository contains the processing pipeline, but not the original
experiment instances and backbones. Those artifacts must be obtained from the
research source and placed locally using the layout documented in
[`data/README.md`](data/README.md).

## Current status

The first implemented technique is **polarity inversion** for linear OPB instances.
For a selected set of variables, the transformation applies

```text
x_i = 1 - y_i
```

The transformation is a bijection between original and augmented assignments,
preserves feasibility and optimality, and transforms backbone labels as follows:

```text
B0 <-> B1
NB  -> NB
```

Current support is deliberately limited to the restricted linear OPB format
(`min:`, `>=` or `=` constraints, and variables `x1` through `xN`). Nonlinear
products and WBO remain outside the core package.

The first real MVC vertical slice is now operational. The repository can fetch and
audit the pinned `D-MIPLIB/MVC-easy` reconstruction, invoke GuroBack through WSL,
normalize and validate its labels, build BackPaS literal graphs, and launch the
seeded reference trainer without modifying either upstream checkout.

The audited local MVC dataset contains 800 training and 100 validation instances,
with 1,200 binary variables and 5,975 constraints per instance. All 900 backbones
and literal graphs are complete. The observed labels contain 418,707 `B0`, 661,293
`B1`, and no `NB` variables. The 200-epoch CPU baseline for seed 0 is in progress;
its outputs remain preliminary until the run and validation gates are complete.

## Development installation

```bash
uv sync --extra dev --extra data --extra ml
```

On Linux/macOS, use `.venv/bin/python` instead.

## Usage

Invert all variables:

```bash
backbone-pbo invert input.opb output.opb
```

Invert only selected variables:

```bash
backbone-pbo invert input.opb output.opb --variables x1 x3 x8
```

Inspect an instance:

```bash
backbone-pbo inspect input.opb
```

Run the local checks:

```bash
ruff check .
ruff format --check .
pytest
```

## MVC baseline pipeline

The first reproducible baseline uses the published `D-MIPLIB/MVC-easy`
distribution. Dataset files, labels, graphs, models, and reports stay inside this
working tree but are ignored by Git.

Download only the 800 training and 100 validation instances:

```powershell
.venv\Scripts\backbone-pbo fetch-dataset MVC --output data\backpas\MVC
```

Audit the instances before backbone extraction:

```powershell
.venv\Scripts\backbone-pbo audit-dataset data\backpas\MVC --allow-missing-backbones
```

Install the pinned Linux runtime in Ubuntu WSL 2:

```powershell
.\scripts\setup_wsl_runtime.ps1
```

From the Gurobi license portal, copy the `grbgetkey ...` command for the academic
license and run it inside Ubuntu WSL. Accept the default path
`/home/javier/gurobi.lic`; never store the key or license file in this repository.
Then run an extraction over eight training and two validation instances, followed
by smoke training:

```powershell
.venv\Scripts\backbone-pbo extract-backbones data\backpas\MVC --smoke
.venv\Scripts\backbone-pbo prepare-backpas data\backpas\MVC `
  --work-dir results\MVC\smoke\seed-0 --smoke
.venv\Scripts\backbone-pbo train-backpas data\backpas\MVC `
  --work-dir results\MVC\smoke\seed-0 --epochs 1
```

Remove `--smoke`, prepare `results\MVC\baseline\seed-0`, and use the default 200
epochs only after the complete labelled dataset passes the audit. The baseline
wrapper fixes the random seed and CPU thread count without modifying BackPaS.

The local reconstruction passed that gate with 900/900 complete backbones and
900/900 literal graphs. Its full baseline command is:

```powershell
.venv\Scripts\backbone-pbo train-backpas data\backpas\MVC `
  --work-dir results\MVC\baseline\seed-0 --epochs 200
```

The public BackPaS `environment.yml` pins Python 3.8.13, PyTorch 1.10.2, PyG
2.0.4, and PySCIPOpt 4.2.0. Bryan Alvarado-Ulloa's thesis reports a newer
PyTorch 2.7.1/PyG 2.6.1 GPU environment. This project currently follows the
published repository environment and records that discrepancy instead of silently
mixing the two software stacks.

Long runs can be started or resumed for any compatible dataset without embedding
MVC-specific paths in the orchestration code:

```powershell
.\scripts\run_backpas_training.ps1 `
  -Dataset data\backpas\MVC `
  -WorkDir results\MVC\baseline\seed-0

.\scripts\run_backpas_training.ps1 `
  -Dataset data\backpas\MVC `
  -WorkDir results\MVC\baseline\seed-0 `
  -Resume

.\scripts\show_backpas_training_status.ps1 `
  -WorkDir results\MVC\baseline\seed-0
```

`-Resume` requires the existing training log, last model, and optimizer checkpoint.
It continues from the next fully completed epoch; an interrupted partial epoch is
recomputed.

Fixtures that write files use `.tmp/pytest-local/` inside the repository. This
directory is ignored by Git, so tests do not depend on the global Windows temporary
directory.

## Initial experimental design

1. Obtain Bryan Alvarado's original instances, backbones, and partitions.
2. Characterize B0, B1, and NB by instance, family, and partition.
3. Keep validation and test splits free of augmentation.
4. Compare the same training setup using `train original` versus
   `train original + train inverted`.
5. Use the same seeds and hyperparameters, and report macro and per-class metrics.

The mathematical details and verified properties are documented in
[`docs/polarity_inversion.md`](docs/polarity_inversion.md).

The repository boundaries and upstream integration are documented in
[`docs/repository_architecture.md`](docs/repository_architecture.md).

The ordered dataset, testing, training, and reporting plan is maintained in
[`docs/roadmap.md`](docs/roadmap.md).

## Repository structure

```text
src/backbone_pbo/       model, OPB parser, augmentation, and CLI
tests/unit/              deterministic tests
tests/property/          mathematical properties tested with Hypothesis
tests/integration/       future tests requiring GuroBack/Gurobi
configs/benchmarks/      benchmark scope and compatibility registry
configs/                 experiment configurations
data/README.md           data contract and provenance requirements
external/README.md       local upstream checkout contract
docs/                    mathematical and methodological decisions
scripts/                 environment and thesis helper scripts
```
