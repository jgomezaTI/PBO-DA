"""Trainer and evaluation metrics for Backbone GNN with scale stratification."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch_geometric.loader import DataLoader

from backbone_pbo.training.gnn import PBOBackboneGNN


def _compute_metrics_dict(
    y_true: np.ndarray, y_pred: np.ndarray, loss: float = 0.0
) -> dict[str, Any]:
    if len(y_true) == 0:
        return {}

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    pred_bb_mask = (y_pred == 0) | (y_pred == 1)
    if pred_bb_mask.sum() > 0:
        correct_bb = ((y_pred == y_true) & pred_bb_mask).sum()
        fixation_precision = float(correct_bb / pred_bb_mask.sum())
    else:
        fixation_precision = 0.0

    return {
        "loss": loss,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "fixation_precision": fixation_precision,
        "class_metrics": {
            "B0": {
                "precision": float(p_per_class[0]),
                "recall": float(r_per_class[0]),
                "f1": float(f1_per_class[0]),
                "support": int(support[0]),
            },
            "B1": {
                "precision": float(p_per_class[1]),
                "recall": float(r_per_class[1]),
                "f1": float(f1_per_class[1]),
                "support": int(support[1]),
            },
            "NB": {
                "precision": float(p_per_class[2]),
                "recall": float(r_per_class[2]),
                "f1": float(f1_per_class[2]),
                "support": int(support[2]),
            },
        },
    }


def evaluate_model(
    model: PBOBackboneGNN,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate GNN model on variable nodes and compute multi-class metrics + breakdown by scale."""
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_scales: list[np.ndarray] = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            mask = batch.var_mask & (batch.y >= 0)

            if not mask.any():
                continue

            loss = F.cross_entropy(logits[mask], batch.y[mask])
            total_loss += loss.item()
            num_batches += 1

            preds = logits[mask].argmax(dim=-1).cpu().numpy()
            targets = batch.y[mask].cpu().numpy()

            # Repeat scale_id across variable nodes in each graph
            # Batch has graph-level scale_id
            node_scale_ids = []
            ptr = (
                batch.ptr.cpu().numpy()
                if hasattr(batch, "ptr") and batch.ptr is not None
                else [0, len(batch.x)]
            )
            scale_arr = (
                batch.scale_id.cpu().numpy()
                if hasattr(batch, "scale_id") and batch.scale_id is not None
                else [1] * (len(ptr) - 1)
            )

            for g_idx in range(len(ptr) - 1):
                g_start, g_end = ptr[g_idx], ptr[g_idx + 1]
                g_mask = mask[g_start:g_end].cpu().numpy()
                g_scale = scale_arr[g_idx] if np.ndim(scale_arr) > 0 else int(scale_arr)
                node_scale_ids.extend([g_scale] * g_mask.sum())

            all_preds.append(preds)
            all_targets.append(targets)
            all_scales.append(np.array(node_scale_ids))

    if not all_preds:
        return {}

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    scales = np.concatenate(all_scales) if all_scales else np.zeros_like(y_true)

    avg_loss = total_loss / max(1, num_batches)
    global_metrics = _compute_metrics_dict(y_true, y_pred, loss=avg_loss)
    global_metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    # Breakdown by scale (0: S, 1: M, 2: L)
    scale_names = {0: "Small (S)", 1: "Medium (M)", 2: "Large (L)"}
    by_scale_metrics = {}
    for sc_id, sc_name in scale_names.items():
        sc_mask = scales == sc_id
        if sc_mask.sum() > 0:
            by_scale_metrics[sc_name] = _compute_metrics_dict(y_true[sc_mask], y_pred[sc_mask])

    global_metrics["by_scale"] = by_scale_metrics
    return global_metrics


def train_model(
    model: PBOBackboneGNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
) -> tuple[PBOBackboneGNN, dict[str, list[float]]]:
    """Train GNN model with early tracking of validation Macro-F1."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
    }

    best_val_macro_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())

    for _epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch.x, batch.edge_index)
            mask = batch.var_mask & (batch.y >= 0)

            if not mask.any():
                continue

            loss = F.cross_entropy(logits[mask], batch.y[mask])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_train_loss = epoch_loss / max(1, batches)
        val_metrics = evaluate_model(model, val_loader, device)

        val_loss = val_metrics.get("loss", 0.0)
        val_f1 = val_metrics.get("macro_f1", 0.0)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model, history
