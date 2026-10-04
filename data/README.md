# Data

Experimental data is kept inside this repository working tree, but is not versioned
by Git. The canonical local layout mirrors BackPaS:

```text
data/
  backpas/
    MIS/
      instance/
      backbone/
      backbone_extraction_log/     # optional
      ml_dataset_variables/        # generated
      ml_dataset_literals/         # generated
    MVC/
      ...
    CA/
      ...
  manifests/                       # generated local provenance records
```

Use the singular directory names `instance/` and `backbone/`; these are the names
expected by BackPaS. An instance named `train_example.opb` must have a matching
backbone named `train_example.opb.backbone`. The same suffix-preserving rule applies
to LP and MPS files, for example `train_0000.lp.backbone`.

Before running experiments, record at least:

- source and license;
- version or acquisition date;
- families and number of instances;
- checksums of the original files;
- backbone format;
- train/validation/test partitions and their seed;
- the exact generation or transformation procedure.

Apply augmentation **after** creating the partitions and only to the training split
to avoid leakage between an instance and its inverted version.

Do not replace missing research data with generated examples. Synthetic instances
may be used for unit and smoke tests, but their results are not thesis evidence and
must not be mixed with the BackPaS benchmark results.

## BackPaS repository status

The public BackPaS repository contains processing code and a `dataset/dataset.txt`
placeholder, but not the experiment instances or `.backbone` files. Its README says
that instances must be placed manually in `dataset/DATASET_NAME/instance/`;
backbones can be placed in `dataset/DATASET_NAME/backbone/` or generated with
GuroBack. Then run `1_create_ml_dataset.py` and `2_create_partitions.py`.

Bryan Alvarado's thesis documents three PBO benchmarks: MIS, MVC, and CA. It also
explains that Item Placement, present in the related ConPaS evaluation, was excluded
because it has non-binary variables. Therefore `IP` is not a valid fourth dataset for
the current binary-only pipeline.

The backbone format observed in `1_create_ml_dataset.py` is:

```text
b -x1   # B0
b x2    # B1
b 0     # complete extraction; must be the last line
```

Extraction requires GuroBack, Gurobi, and a valid license. Unit tests use a small
fixture for this format. The real local MVC reconstruction and all derived labels
remain ignored by Git and are not redistributed.

## MVC-easy intake

The local MVC baseline is reconstructed from the published
`weiminhu/D-MIPLIB`, configuration `MVC-easy`. The source contains rows rather
than original filenames, so the importer creates deterministic names and records
the exact row-to-file mapping in `source.json`:

```text
train_0000.lp ... train_0799.lp
valid_0000.lp ... valid_0099.lp
```

`partitions.json` is authoritative. The importer does not download or use the
D-MIPLIB test split for this baseline because it is not the separate 6000-node
solver-evaluation set described in the BackPaS thesis.

The completed local audit records 800 training and 100 validation instances, 900
complete backbones, and 900 literal graphs. Its label distribution is 418,707 `B0`,
661,293 `B1`, and zero `NB`. The BackPaS thesis does not publish an equivalent
per-class count, so this observation must not be attributed to its original private
artifacts without their checksums.
