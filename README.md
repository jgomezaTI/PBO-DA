# backbonePBO

TT1 project on data augmentation techniques to mitigate backbone class imbalance
in GNNs for Pseudo-Boolean Optimization (PBO).

The repository focuses on augmentation methods as the main contribution. BackPaS,
GuroBack, and related prior work may be used as infrastructure or baselines, but
they are not the project's primary scope.

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
products, WBO, and GuroBack integration are not yet part of the core package.

## Development installation

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
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

Fixtures that write files use `.tmp/pytest-local/` inside the repository. This
directory is ignored by Git, so tests do not depend on the global Windows temporary
directory.

## Initial experimental design

1. Obtain the original instances, backbones, and partitions.
2. Characterize B0, B1, and NB by instance, family, and partition.
3. Keep validation and test splits free of augmentation.
4. Compare the same training setup using `train original` versus
   `train original + train inverted`.
5. Use the same seeds and hyperparameters, and report macro and per-class metrics.

The mathematical details and verified properties are documented in
[`docs/polarity_inversion.md`](docs/polarity_inversion.md).

For safe integration with an existing Overleaf template, see
[`docs/overleaf.md`](docs/overleaf.md).

## Repository structure

```text
src/backbone_pbo/       model, OPB parser, augmentation, and CLI
tests/unit/              deterministic tests
tests/property/          mathematical properties tested with Hypothesis
tests/integration/       future tests requiring GuroBack/Gurobi
configs/                 initial experiment configurations
data/README.md           data contract and provenance requirements
docs/                    mathematical and methodological decisions
```
