# Experimental roadmap

## Objective

Build a reproducible evaluation of data augmentation for backbone prediction using
only the original BackPaS research datasets. Instance preparation means importing,
validating, and documenting Bryan Alvarado's artifacts; it does not mean replacing
them with newly generated synthetic instances.

The execution order is `MVC -> MIS -> CA`. MVC is the first vertical slice because
it was explicitly discussed with the thesis supervisor. A fourth problem remains a
scope decision and must not block work on the three confirmed PBO benchmarks.

## Non-negotiable experimental rules

1. Preserve the original BackPaS instances, backbones, and partitions.
2. Audit all data before creating graphs or starting training.
3. Reproduce the unaugmented baseline before evaluating an augmentation.
4. Apply augmentation only to the training partition.
5. Keep validation and test instances original in every comparison.
6. Use identical partitions, architecture, hyperparameters, and seeds for paired
   baseline/augmentation runs.
7. Record dataset provenance, upstream commits, configuration, seed, environment,
   metrics, and generated files for every run.
8. Do not include a result in the thesis until its validation gate is complete.

## Milestones

### M0 - Scope and upstream setup

**Status:** complete for repository structure; data access remains open.

**Inputs**

- Local BackPaS and GuroBack checkouts.
- Confirmed domains: MVC, MIS, and CA.
- Supervisor decision pending for the fourth problem.

**Exit criteria**

- Upstream URLs and commits are recorded.
- Repository boundaries and local data layout are documented.
- Synthetic benchmark artifacts are absent from the repository.

### M1 - Dataset intake and audit command

**Status:** complete.

Implement `backbone-pbo audit-dataset <dataset-directory>` with machine-readable
and human-readable output.

The audit must verify:

- canonical `instance/` and `backbone/` directories;
- one matching suffix-preserving `.backbone` file per labelled instance;
- valid LP, MPS, or restricted OPB syntax and binary variable identifiers;
- complete backbone marker and labels that belong to the instance;
- original partition assignment without overlap or unexpected files;
- instance, variable, constraint, and label counts by partition;
- B0, B1, and NB distribution by partition;
- SHA-256 checksums and dataset provenance;
- unsupported instances, without silently skipping them.

**Tests**

- unit tests for missing, duplicate, malformed, and mismatched files;
- a valid miniature BackPaS-layout fixture;
- a test that detects train/validation/test overlap;
- deterministic manifest generation.

**Outputs**

```text
data/manifests/<BENCHMARK>.json
results/audits/<BENCHMARK>-audit.md
```

Both output trees remain local and ignored by Git. A sanitized example manifest may
be versioned later if it contains no research data.

**Exit criteria**

- All tests pass.
- The command returns a non-zero exit code for an invalid dataset.
- The manifest is reproducible from the same input files.

### M2 - Obtain and prepare MVC

**Status:** complete for the reproducible `MVC-easy` reconstruction. The 800/100
instances, 900 GuroBack labels, and 900 literal graphs pass their validation gates.
The WLS academic license remains outside the repository.

The final audit found 418,707 `B0`, 661,293 `B1`, and zero `NB` variables. This is
an observed property of the reconstruction, not a class distribution reported in the
BackPaS thesis. Because the original experiment checksums are unavailable, the data
must be described as a reproducible reconstruction rather than a bit-for-bit copy.

**Required artifacts**

- original MVC instances;
- matching backbone files for training and validation labels;
- original BackPaS partition definition or naming rule;
- source, licence, acquisition date, and upstream configuration.

**Procedure**

1. Export the D-MIPLIB rows under `data/backpas/MVC/` with deterministic names and
   retain their original split, row identifier, source revision, and checksum.
2. Run the dataset audit.
3. Resolve every audit error; do not skip invalid files silently.
4. Generate variable/constraint graphs with the BackPaS-compatible representation.
5. Record class distributions and dataset dimensions.

**Exit criteria**

- MVC passes the complete audit.
- Partition counts agree with the source documentation.
- Every training label can be traced to its original backbone file.

### M3 - Reproduce the MVC baseline

**Status:** in progress. The smoke run passed, and the 200-epoch seed-0 CPU baseline
is running over 800 training and 100 validation graphs. The first completed epoch
created all three expected checkpoints. No augmentation is present in this run.

The current environment follows the public BackPaS `environment.yml` (Python
3.8.13, PyTorch 1.10.2, PyG 2.0.4). The thesis reports PyTorch 2.7.1 and PyG 2.6.1
on an NVIDIA A40. This software and hardware deviation must accompany every
comparison with the thesis. CPU training takes roughly twenty minutes per epoch;
cluster or GPU access should be investigated before expanding the experiment matrix.

Run two levels of training:

1. **Smoke training:** one seed and a small epoch budget to validate loading,
   batching, forward/backward passes, checkpoints, and metric files.
2. **Baseline training:** the confirmed BackPaS configuration over the full MVC
   partitions and the agreed seed set.

Record at minimum:

- loss curves and selected checkpoint;
- accuracy and balanced accuracy;
- precision, recall, and F1 for B0, B1, and NB;
- macro-F1 and confusion matrix;
- runtime, hardware, seed, and configuration;
- comparison with the corresponding BackPaS reference result when the same metric
  is available.

**Exit criteria**

- Repeated execution with the same seed reproduces the result within the documented
  deterministic tolerance.
- The baseline discrepancy relative to BackPaS is explained or resolved.
- No augmentation is present in any partition.

### M4 - Evaluate polarity inversion on MVC

1. Freeze the successful MVC baseline configuration.
2. Apply polarity inversion only after partitioning and only to training instances.
3. Verify transformed OPB syntax, backbone labels, feasibility correspondence,
   objective offset, and involution.
4. Train baseline and polarity scenarios with paired seeds.
5. Compare per-class metrics and uncertainty across seeds.

**Exit criteria**

- Validation and test file checksums are identical in both scenarios.
- Baseline and augmentation runs differ only in the declared augmentation fields.
- Results include means, standard deviations, per-seed values, and confusion
  matrices.
- The written interpretation distinguishes mathematical correctness from empirical
  usefulness.

### M5 - Repeat the complete slice for MIS

Repeat M2 through M4 for MIS. Do not copy MVC conclusions: audit the MIS backbone
distribution and report its results independently.

**Exit criteria**

- MIS audit, baseline reproduction, and polarity comparison are complete.
- MVC and MIS results use the same reporting schema.

### M6 - Repeat the complete slice for CA

Repeat M2 through M4 for CA, accounting for its item/bid formulation and its own
instance scale.

**Exit criteria**

- CA audit, baseline reproduction, and polarity comparison are complete.
- MVC, MIS, and CA can be compared without pooling their instances or labels.

### M7 - Decide the fourth problem

Ask the supervisor whether the requested fourth problem is:

- Item Placement from ConPaS, which requires extending the project beyond binary
  PBO; or
- another binary PBO benchmark with artifacts compatible with this thesis.

**Exit criteria**

- The problem name, formulation, data source, licence, and compatibility decision
  are recorded.
- If incompatible, the exclusion and its methodological reason are documented.

### M8 - Additional augmentation techniques

Only after the polarity experiment is stable on at least MVC:

1. implement one augmentation at a time;
2. add mathematical and property-based tests;
3. audit its effect on class distributions;
4. run the same paired baseline protocol on MVC, MIS, and CA;
5. evaluate combinations only after the individual techniques are understood.

Candidate techniques are tracked separately in `docs/propuestas_aumentacion.md`.

### M9 - End-to-end Predict-and-Search evaluation

Integrate the selected models with BackPaS trust-region construction and compare
against the original solver flow. Report predictive metrics separately from search
metrics such as primal integral, solution quality over time, valid fixations, and
runtime overhead.

**Exit criteria**

- The search experiment is reproducible from a recorded model and configuration.
- Prediction improvements are not presented as solver improvements unless the
  end-to-end measurements support that conclusion.

### M10 - Thesis reporting

For every completed benchmark, produce:

- dataset and partition characterization;
- B0/B1/NB distribution table;
- baseline reproduction table;
- augmentation comparison table;
- per-class metrics and confusion matrix;
- statistical summary across paired seeds;
- limitations and deviations from BackPaS.

The thesis status must continue to distinguish implemented, tested, experimentally
evaluated, and pending work.

## Test ladder

| Level | Purpose | Must pass before |
| --- | --- | --- |
| Unit | Parser, labels, transformations, metrics, manifests | Dataset audit |
| Property | Bijection, involution, feasibility and optimum correspondence | Augmented training |
| Dataset contract | Files, checksums, partitions, counts, class distribution | Any training |
| Integration | GuroBack/BackPaS formats and graph conversion | Baseline reproduction |
| Training smoke | One short run, checkpoint and metrics | Full training |
| Reproducibility | Same seed/configuration produces equivalent output | Reporting results |
| End-to-end | Model guidance inside Predict-and-Search | Solver claims |

## Standard run layout

Generated artifacts use a stable hierarchy and remain ignored by Git:

```text
results/
  <BENCHMARK>/
    <EXPERIMENT>/
      seed-<SEED>/
        run-manifest.json
        metrics.json
        training-log.csv
        confusion-matrix.csv
        best-model.pth
```

`<EXPERIMENT>` begins with `baseline` or `polarity-global`; later augmentations use
their explicit technique and parameter names.

## Experiment matrix

| Benchmark | Data audit | Baseline | Polarity | Other techniques | Search evaluation |
| --- | --- | --- | --- | --- | --- |
| MVC | Passed: 900/900 | Running: seed 0 | Blocked by baseline | Not started | Not started |
| MIS | Pending data | Blocked by audit | Blocked by baseline | Not started | Not started |
| CA | Pending data | Blocked by audit | Blocked by baseline | Not started | Not started |
| Fourth problem | Pending decision | Not applicable | Not applicable | Not applicable | Not applicable |

## Immediate work queue

1. Complete the 200-epoch MVC seed-0 baseline and collect its validation report.
2. Record the environment and hardware deviation from the BackPaS thesis.
3. Ask the supervisor about university cluster or GPU access before augmented runs.
4. Freeze the successful baseline configuration and generate polarity-augmented
   training data without changing validation.
5. Confirm the fourth benchmark and obtain the MIS and CA source artifacts.
