# Repository architecture

## Purpose

This repository owns the data-augmentation contribution and its controlled
evaluation. It does not vendor or rewrite BackPaS or GuroBack. Keeping these roles
separate makes it possible to update either upstream project without mixing its code
with the thesis implementation.

## Boundaries

```text
external/backpas/         local checkout of the reference ML and search pipeline
external/guroback/        local checkout of the reference backbone extractor
data/backpas/<BENCHMARK>/ local instances, backbones, and derived graph datasets
src/backbone_pbo/         augmentation, validation, adapters, and experiment code
configs/                  tracked experiment and benchmark definitions
results/                  generated outputs; never committed
thesis/                   local LaTeX source; never committed
```

The two directories below `external/` are ignored by Git. Their URLs and roles are
documented in `external/README.md`, and `scripts/setup_upstreams.ps1` creates the
checkouts reproducibly.

## Data flow

1. Place an original benchmark in `data/backpas/<BENCHMARK>/instance/`.
2. Place the corresponding labels in `backbone/`, or extract them with GuroBack and
   a valid Gurobi installation and licence.
3. Preserve Bryan Alvarado's original partitions. Do not repartition before the
   baseline is reproduced.
4. Build the BackPaS graph representation from the original instances.
5. Apply an augmentation only to the training partition.
6. Run baseline and augmented training with identical partitions, seeds,
   architecture, and hyperparameters.
7. Save generated models, logs, figures, and tables below `results/`.

## Benchmark scope

The BackPaS thesis and repository support three binary PBO domains: `MIS`, `MVC`,
and `CA`. ConPaS also reports Item Placement (`IP`), but BackPaS excludes it because
its formulation contains non-binary variables. Since the current parser, backbone
labels, and polarity transformation are binary-only, `IP` cannot be added by merely
creating a folder.

The request to report four problem types is therefore recorded as an unresolved
scope decision. The fourth domain must be confirmed with the thesis supervisor and
must have a documented data source compatible with binary PBO before experiments
are reported for it.

## Reproducibility rule

Every reported run must identify the upstream commits, dataset provenance,
partition file or naming rule, augmentation parameters, random seed, environment,
and output directory. Unit tests use only small in-memory fixtures; the repository
does not maintain a separate synthetic benchmark dataset.

For MVC, the local source is a pinned export of `D-MIPLIB/MVC-easy`, not a copy of
the unpublished BackPaS experiment directory. Matching published dimensions are
evidence of compatibility, not proof of bit-for-bit identity. Comparisons must also
distinguish the public repository environment (PyTorch 1.10.2/PyG 2.0.4) from the
newer GPU environment reported in the final thesis (PyTorch 2.7.1/PyG 2.6.1).
