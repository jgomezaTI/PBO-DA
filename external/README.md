# External research software

This directory contains local, ignored checkouts of the research software used by
the project:

| Directory | Upstream | Role |
| --- | --- | --- |
| `external/backpas/` | <https://github.com/bryan-alvarado-ulloa/backpas> | Reference ML and Predict-and-Search pipeline |
| `external/guroback/` | <https://github.com/bryan-alvarado-ulloa/guroback> | Backbone extraction with Gurobi |

Create both checkouts from the repository root:

```powershell
./scripts/setup_upstreams.ps1
```

The checkouts are dependencies, not project source. Do not edit them to implement
augmentation techniques. Record the exact commit hashes used by each experiment.
