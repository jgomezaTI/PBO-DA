# Experimental roadmap

## Active thesis-workplan initiative

The local Spec Kit initiative `001-thesis-workplan` coordinates five independently
verifiable increments: literature grounding (US1), dataset and methodology traceability
(US2), the controlled baseline/polarity comparison (US3), reproducible reporting (US4),
and thesis synchronization (US5). Its execution order is setup → shared contracts → US1
and US2 → US3 → US4 → US5 → final quality gates.

As of 2026-10-20, the MVC baseline is still active and must not be interrupted. The
implementation may advance documentation and deterministic tests while training runs;
dataset generation and full polarity training remain gated by provenance, mathematical
correctness, smoke testing, and configuration comparability.

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

### Supervisor decision - label generation protocol (pending)

**Question for the supervisor:** Should the thesis use GuroBack as the single
procedure for generating ground-truth backbone labels for the original datasets,
and for any augmented instances that require new labels, instead of mixing it
with the repository's direct solver-based labeler?

Options to discuss:

1. **Unify on GuroBack:** use it for original instances and every transformed
   instance that needs fresh labels. Algebraically valid transformations (for
   example, polarity inversion) may derive labels from their source labels. First
   confirm GuroBack supports each instance format and that its extraction protocol
   matches the thesis definition.
2. **Keep both procedures:** permit the direct solver implementation for formats
   or transformations not supported by GuroBack, but require a documented
   equivalence check and record the labeler/version for every dataset.

GuroBack itself uses Gurobi internally, so either option still needs the solver
and a valid license when labels are generated. No datasets are to be relabeled and
no past checkpoints are to be replaced until this decision is confirmed. After the
decision, update the protocol, audit label provenance for MVC/MIS/CA, and establish
whether existing baselines remain comparable or need to be rerun.

### Local training reuse and partition completion (current recommendation)

Do not restart all experiments. Preserve datasets, logs, metrics, and checkpoints.
First determine whether an existing model was trained on the complete official
partitions. Adding missing instances to a dataset folder does not update a trained
model: if instances are added to `train`, that benchmark's training must resume or
be rerun on the completed partition before it represents the complete dataset.
Validation and test membership must remain fixed and must never be moved to train.

Current local evidence:

- **MVC:** the local `MVC-easy` reconstruction has 800 train and 100 validation
  instances, 900 complete GuroBack-format labels, and a passing audit. Its baseline
  manifest records this dataset and the pinned GuroBack revision. Keep this baseline.
- **MIS:** the current source partition and the saved `full` training partition
  agree exactly at 761 train and 97 validation. This is not a case of missing files
  relative to the local source: that source itself does not contain the target
  800/100. The local audit reports 1,500 instances but rejects all 1,500 backbone
  files because their `nb x...` lines do not match the expected canonical format.
  Therefore, do not invent a train/validation difference or add files from another
  split. To reach the thesis target, obtain/restore the intended 800/100 MIS source,
  label and audit the full partitions, then train MIS on that dataset. Existing MIS
  runs remain historical results for their recorded 761/97 set.
- **CA:** the saved `full` training partition has 788 train and 100 validation;
  the current source manifest has 800/100. The exact 12 missing training instance
  names are present locally with corresponding backbone files. A previous GuroBack
  audit reports all 900 target labels complete, although that audit also flags 300
  extra unassigned files in the directory. First materialize a clean, isolated
  CA dataset containing the official 800/100 partitions and validate its 900 labels.
  Then train/continue CA with all 800 training instances. Keep the existing 788/100
  run as a historical comparison; it does not become an 800/100 result by adding
  files after training.

This limits new work to the incomplete MIS and CA baselines. It does not require
repeating MVC or unrelated experiments. Never append validation instances to train;
only the 12 verified CA training instances may fill the CA training gap.

### M9a - Optional large-instance inference study (supervisor decision pending)

This branch reuses a trained checkpoint on larger, unlabeled instances. It does
not recompute B0/B1/NB labels. It can measure inference cost, trust-region
construction, solver runtime, solution quality, primal integral, and whether a
valid trust region is produced. It cannot report classification accuracy, F1,
recall, or backbone-selection quality without ground-truth labels.

Candidate sources are the public Distributional MIPLIB splits and the thesis'
GeCo MIS generalization collections. Preserve source splits and record the
exact distribution, instance sizes, citation, license, checkpoint, and solver
configuration. The generated local scalability files are excluded until their
formulation and distribution are independently validated against the chosen
source.

If the supervisor approves:

1. Select unlabeled large-instance distributions for approved benchmarks and
   sizes, without relabeling them.
2. Run the existing checkpoint for inference and trust-region construction.
3. Run the original solver and BackPaS on the same instances and hardware limits.
4. Record per-instance graph-construction and prediction times, trust-region
   success/failure, solver runtime, best objective, primal integral, and
   feasibility/optimality outcomes.
5. Report this as solver-level generalization, not supervised classification
   performance. Backbone quality remains unverified without ground truth.

Keep this branch optional until the supervisor decides; it is not part of the
confirmed experiment matrix.

**Runtime planning estimate:** existing epoch logs show validation passes over
100 instances taking roughly 30–80 seconds, or about 0.3–0.8 seconds per
instance on the logged local CPU environment. This includes loss/metric
calculation and may reuse graph caches; it is not an inference-only benchmark.
For six collections of 100 instances, a preliminary CPU planning range is about
5–15 minutes for model-pass time, plus graph construction/loading,
trust-region construction, solver runs, and startup. Large graphs may increase
cost with variable and constraint counts. Measure per-instance timings before
making a precise runtime claim. GPU can reduce neural-network time, but not
graph conversion or solver work.

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

1. Allow the active 200-epoch MVC seed-0 baseline to complete without interruption and
   then collect its validation report.
2. Implement the shared provenance and result contracts and their tests.
3. Complete the literature protocol, comparison matrix, and state-of-the-art chapter.
4. Record the effective environment and hardware deviation from the BackPaS thesis.
5. Ask the supervisor about university cluster or GPU access before augmented runs.
6. Freeze the successful baseline configuration and generate polarity-augmented
   training data without changing validation.
7. Confirm the fourth benchmark and obtain the MIS and CA source artifacts.
