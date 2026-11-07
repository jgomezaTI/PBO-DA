# Auditoría de entrenamiento local

Esta auditoría evita repetir en Colab experimentos que ya tienen evidencia
local. La carpeta `results/` es la fuente de verdad para este control.

## Experimentos ya realizados

| Problema | Tratamientos con evidencia local |
|---|---|
| MVC | baseline, polarity, permutation |
| MIS | full (baseline), polarity, permutation |
| CA | full (baseline), polarity, permutation |

`full` es el nombre histórico usado por los entrenamientos de MIS y CA y debe
tratarse como el baseline correspondiente, no como un experimento adicional.

El runner `scripts/cloud_run.py` rechaza por defecto un trabajo cuyo directorio
en Drive ya contiene `metrics.json` o `experiment-status.json`. Para reproducir
intencionalmente un resultado existente hay que usar `--allow-completed`.

## Trabajo pendiente

Las propuestas de perturbación controlada de restricciones, perturbación de la
función objetivo y generación sintética condicionada están documentadas, pero
no tienen todavía una definición ejecutable ni datasets validados. No deben
agregarse a la matriz de entrenamiento hasta fijar sus parámetros, validación
con el solver exacto y criterio de descarte.
