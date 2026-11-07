"""Generate a reproducible validation report for the three BackPaS runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CLASSES = ("B0", "B1", "NB")
EXPERIMENTS = (
    ("MVC", "polarity", "results/MVC/polarity/seed-0", "data/backpas/MVC-polarity-run"),
    ("MIS", "baseline", "results/MIS/full/seed-0", "data/backpas/MIS"),
    ("CA", "baseline", "results/CA/full/seed-0", "data/backpas/CA"),
)
SUMMARY_FIELDS = (
    "experiment",
    "scenario",
    "seed",
    "epochs_recorded",
    "best_epoch",
    "best_loss",
    "accuracy_micro",
    "accuracy_macro",
    "f1_score_micro",
    "f1_score_macro",
    "precision_macro",
    "precision_micro",
    "recall_macro",
    "recall_micro",
    "checkpoint",
)


def read_validation_log(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    validation = [row for row in rows if row.get("partition") == "valid"]
    if not validation:
        raise ValueError(f"No validation rows found in {path}")
    return validation


def best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda row: float(row["loss"]))


def metric(row: dict[str, str], name: str) -> float:
    """Read a global metric, deriving recall when BackPaS omits it."""
    if name in row and row[name] not in (None, ""):
        return float(row[name])
    if name in ("recall_macro", "recall_micro"):
        values = [float(row[f"multiclass_recall_{class_name}"]) for class_name in CLASSES]
        return sum(values) / len(values)
    raise KeyError(f"Metric {name!r} is not available in the training log")


def svg_line_chart(path: Path, series: dict[str, list[tuple[int, float]]], title: str) -> None:
    width, height = 900, 500
    left, right, top, bottom = 70, 25, 55, 55
    values = [value for points in series.values() for _, value in points]
    epochs = [epoch for points in series.values() for epoch, _ in points]
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1.0
    xlo, xhi = min(epochs), max(epochs)
    if xhi == xlo:
        xhi = xlo + 1

    def x(epoch: int) -> float:
        return left + (epoch - xlo) / (xhi - xlo) * (width - left - right)

    def y(value: float) -> float:
        return height - bottom - (value - lo) / (hi - lo) * (height - top - bottom)

    colors = ("#2563eb", "#dc2626", "#059669")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#444"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#444"/>',
        f'<text x="18" y="{height / 2:.0f}" transform="rotate(-90 18 {height / 2:.0f})" text-anchor="middle" font-family="sans-serif">metric</text>',
        f'<text x="{width / 2:.0f}" y="{height-12}" text-anchor="middle" font-family="sans-serif">epoch</text>',
    ]
    for index, (name, points) in enumerate(series.items()):
        color = colors[index % len(colors)]
        polyline = " ".join(f"{x(epoch):.1f},{y(value):.1f}" for epoch, value in points)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{polyline}"/>')
        lx = width - 150
        ly = 55 + index * 22
        lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+20}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{lx+27}" y="{ly+5}" font-family="sans-serif" font-size="12">{name}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/comparative"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    curves = {"loss": {}, "accuracy_micro": {}, "f1_score_macro": {}}
    class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []

    for experiment, scenario, work, dataset in EXPERIMENTS:
        work_path = root / work
        log_path = work_path / "ml_training/graph_with_literals_8_GTR/training_log.csv"
        checkpoint = work_path / "ml_training/graph_with_literals_8_GTR/best_model.pth"
        rows = read_validation_log(log_path)
        best = best_row(rows)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing best checkpoint: {checkpoint}")
        key = f"{experiment} ({scenario})"
        for metric_name in curves:
            curves[metric_name][key] = [(int(row["epoch"]), float(row[metric_name])) for row in rows]
        records.append({
            "experiment": experiment,
            "scenario": scenario,
            "seed": 0,
            "epochs_recorded": max(int(row["epoch"]) for row in rows) + 1,
            "best_epoch": int(best["epoch"]),
            "best_loss": float(best["loss"]),
            **{metric_name: metric(best, metric_name) for metric_name in (
                "accuracy_micro", "accuracy_macro", "f1_score_micro", "f1_score_macro",
                "precision_macro", "precision_micro", "recall_macro", "recall_micro",
            )},
            "checkpoint": str(checkpoint.relative_to(root)).replace("\\", "/"),
        })
        for class_name in CLASSES:
            class_rows.append({
                "experiment": experiment,
                "scenario": scenario,
                "class": class_name,
                "precision": float(best[f"multiclass_precision_{class_name}"]),
                "recall": float(best[f"multiclass_recall_{class_name}"]),
                "f1_score": float(best[f"multiclass_f1_score_{class_name}"]),
                "accuracy": float(best[f"multiclass_accuracy_{class_name}"]),
            })
        for actual in CLASSES:
            for predicted in CLASSES:
                confusion_rows.append({
                    "experiment": experiment,
                    "scenario": scenario,
                    "actual": actual,
                    "predicted": predicted,
                    "value": float(best[f"confusion_matrix_cm_{actual}_{predicted}"]),
                })

    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    for name, rows in (("class-metrics.csv", class_rows), ("confusion-matrices.csv", confusion_rows)):
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    for metric_name, series in curves.items():
        svg_line_chart(output / f"{metric_name}.svg", series, f"Validation {metric_name}")

    manifest = {
        "report_type": "comparative-validation",
        "selection": "minimum validation cross-entropy",
        "validation_only": True,
        "class_order": list(CLASSES),
        "experiments": [
            {"experiment": e, "scenario": s, "seed": 0, "dataset": d, "work_dir": w}
            for e, s, w, d in EXPERIMENTS
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Comparative validation report",
        "",
        "Generated from validation rows only (`partition=valid`). The selected checkpoint is the `best_model.pth` associated with the minimum validation cross-entropy.",
        "",
        "## Summary",
        "",
        "| Experiment | Scenario | Epochs | Best epoch | Loss | Accuracy | Macro-F1 | Macro precision | Macro recall | Checkpoint |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in records:
        lines.append("| {experiment} | {scenario} | {epochs_recorded} | {best_epoch} | {best_loss:.6f} | {accuracy_micro:.4f} | {f1_score_macro:.4f} | {precision_macro:.4f} | {recall_macro:.4f} | `{checkpoint}` |".format(**row))
    lines += ["", "## Per-class metrics", "", "See [`class-metrics.csv`](class-metrics.csv) for the complete table.", "", "## Confusion matrices", "", "See [`confusion-matrices.csv`](confusion-matrices.csv); class order is `B0`, `B1`, `NB`.", "", "## Training curves", "", "- [Validation loss](loss.svg)", "- [Validation accuracy](accuracy_micro.svg)", "- [Validation macro-F1](f1_score_macro.svg)", "", "## Reproducibility", "", "- [Manifest](manifest.json)", "- [Summary CSV](summary.csv)", "- All metrics are selected from validation rows only."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Generated comparative report in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
