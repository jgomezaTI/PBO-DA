# Comparative validation report

Generated from validation rows only (`partition=valid`). The selected checkpoint is the `best_model.pth` associated with the minimum validation cross-entropy.

## Summary

| Experiment | Scenario | Epochs | Best epoch | Loss | Accuracy | Macro-F1 | Macro precision | Macro recall | Checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| MVC | polarity | 50 | 41 | 0.124604 | 0.9443 | 0.9412 | 0.9413 | 0.6274 | `results/MVC/polarity/seed-0/ml_training/graph_with_literals_8_GTR/best_model.pth` |
| MIS | baseline | 65 | 62 | 0.374969 | 0.8318 | 0.8266 | 0.8284 | 0.8394 | `results/MIS/full/seed-0/ml_training/graph_with_literals_8_GTR/best_model.pth` |
| CA | baseline | 65 | 56 | 0.233952 | 0.9015 | 0.7437 | 0.8111 | 0.4720 | `results/CA/full/seed-0/ml_training/graph_with_literals_8_GTR/best_model.pth` |

## Per-class metrics

See [`class-metrics.csv`](class-metrics.csv) for the complete table.

## Confusion matrices

See [`confusion-matrices.csv`](confusion-matrices.csv); class order is `B0`, `B1`, `NB`.

## Training curves

- [Validation loss](loss.svg)
- [Validation accuracy](accuracy_micro.svg)
- [Validation macro-F1](f1_score_macro.svg)

## Reproducibility

- [Manifest](manifest.json)
- [Summary CSV](summary.csv)
- All metrics are selected from validation rows only.
