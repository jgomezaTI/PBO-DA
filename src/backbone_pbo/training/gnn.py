"""Graph Neural Network architectures for PBO Backbone Prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


class PBOBackboneGNN(nn.Module):
    """Bipartite Graph Neural Network for node-level backbone classification.

    Predicts 3 classes for variable nodes: B0 (0), B1 (1), NB (2).
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 64,
        num_classes: int = 3,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for all nodes."""
        h = self.node_encoder(x)

        for i in range(self.num_layers):
            h_prev = h
            h = self.convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_prev  # Residual connection

        logits = self.classifier(h)
        return logits
