# Data

Datasets are not versioned directly in Git. Before running experiments, record at
least:

- source and license;
- version or acquisition date;
- families and number of instances;
- checksums of the original files;
- backbone format;
- train/validation/test partitions and their seed;
- the exact generation or transformation procedure.

Apply augmentation **after** creating the partitions and only to the training split
to avoid leakage between an instance and its inverted version.

## BackPaS repository status

The public BackPaS repository contains processing code and a `dataset/dataset.txt`
file, but not the experiment instances or `.backbone` files. Its README says that
instances must be placed manually in `dataset/DATASET_NAME/instance/`; backbones can
be placed in `dataset/DATASET_NAME/backbone/` or generated with GuroBack. Then run
`1_create_ml_dataset.py` and `2_create_partitions.py`.

The backbone format observed in `1_create_ml_dataset.py` is:

```text
b -x1   # B0
b x2    # B1
b 0     # complete extraction; must be the last line
```

Extraction requires GuroBack, Gurobi, and a valid license. For now, this repository
includes a small fixture to test the parser for this format, but does not invent or
redistribute experimental datasets.
