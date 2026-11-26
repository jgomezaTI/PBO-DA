# Propuestas de aumentación de datos

Estas técnicas están definidas operacionalmente para generar tratamientos
reproducibles. No se consideran resultados experimentales hasta entrenar y
comparar cada tratamiento con sus métricas.

## 1. Inversión de polaridad selectiva

**Justificación** = Permite aplicar la transformación solo a variables o
instancias seleccionadas, para controlar con mayor precisión el balance entre
`B0` y `B1`.

**Qué hacer** = Definir el criterio de selección, generar las instancias solo
en `train`, transformar sus etiquetas y comparar contra la inversión global.

## 2. Perturbación controlada de restricciones

**Justificación** = Cambios pequeños en restricciones pueden producir instancias
con varias soluciones óptimas y aumentar la presencia de variables `NB`.

**Definición ejecutable general** = Para cada instancia de `train`, elegir con
semilla una restricción y cambiar su RHS en `-1` o `+1`, usando el solver exacto
para rechazar instancias infactibles y conservar solo cambios distintos del
original. Validación y test se copian sin modificación.

**Protocolo específico para MIS** = Como las restricciones de MIS tienen la
forma `x_u + x_v <= 1`, modificar el RHS a `0` o `2` no representa una
perturbación natural del grafo. Por eso, en MIS se añade o elimina exactamente
una restricción de arista `x_u + x_v <= 1`. La operación se elige con semilla,
se conserva la misma función objetivo y el mismo conjunto de variables, y se
rechazan candidatos sin factibilidad, sin cambio real o cuyo backbone no pueda
recalcularse exactamente. Cada candidato registra la operación, la arista
afectada, el número de aristas antes/después y la semilla. Solo se transforma
`train`; `validation` y `test` permanecen intactos.

## 3. Perturbación controlada de la función objetivo

**Justificación** = Modificar de forma acotada los coeficientes puede generar
instancias cercanas con distinta estructura de soluciones óptimas.

**Definición ejecutable** = Para cada instancia de `train`, elegir con semilla
un coeficiente objetivo no nulo y cambiarlo en `-1` o `+1`. Rechazar objetivos
vacíos, instancias inviables y cambios idénticos; recalcular etiquetas óptimas
exactas. Validación y test se copian sin modificación.

## 4. Aumentación por simetrías

**Justificación** = Permutar variables y restricciones conserva la estructura
matemática, pero entrega representaciones distintas del mismo problema.

**Qué hacer** = Aplicar permutaciones consistentes en OPB, etiquetas y grafo, y
comprobar que la predicción sea invariante ante dichas permutaciones.

## 5. Generación sintética condicionada

**Justificación** = Permite construir instancias con tamaños, familias y
distribuciones objetivo de `B0`, `B1` y `NB`.

**Definición ejecutable** = Generar candidatos cercanos a cada instancia de
`train` mediante una perturbación acotada de RHS y otra de coeficiente objetivo.
Resolver cada candidato exactamente, recalcular su backbone y conservarlo solo
si es factible y distinto de su origen. El número de candidatos por instancia es
configurable; no se leen instancias de validación/test como fuentes. Las
particiones originales se mantienen intactas.

La implementación vive en `scripts/materialize_perturbation.py`; el comando
`scripts/materialize_pending_augmentations.py` genera una matriz de doce trabajos
(cuatro técnicas pendientes por MVC, MIS y CA) sin crear datasets ni ejecutar
entrenamientos localmente. Cada sesión Colab crea y valida solo el dataset que
le corresponde antes de entrenar. La inversión selectiva invierte
determinísticamente la mitad de las variables de cada instancia de train.

## Orden sugerido de evaluación

Probar las técnicas en trabajos independientes y simultáneos, con la misma
semilla experimental por problema. Los tratamientos ya ejecutados localmente
no deben relanzarse. Las pérdidas
ponderadas y `Focal Loss` deben tratarse como estrategias de entrenamiento
complementarias, no como aumentaciones de datos.
