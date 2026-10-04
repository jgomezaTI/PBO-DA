# Propuestas de aumentación de datos

Estas son líneas de experimentación posteriores a la primera prueba de
inversión global de polaridad. Son propuestas, no resultados obtenidos.

## 1. Inversión de polaridad selectiva

**Justificación** = Permite aplicar la transformación solo a variables o
instancias seleccionadas, para controlar con mayor precisión el balance entre
`B0` y `B1`.

**Qué hacer** = Definir el criterio de selección, generar las instancias solo
en `train`, transformar sus etiquetas y comparar contra la inversión global.

## 2. Perturbación controlada de restricciones

**Justificación** = Cambios pequeños en restricciones pueden producir instancias
con varias soluciones óptimas y aumentar la presencia de variables `NB`.

**Qué hacer** = Definir perturbaciones acotadas, verificar factibilidad y
optimalidad con el solver exacto, recalcular el backbone y medir la distribución
de `NB`.

## 3. Perturbación controlada de la función objetivo

**Justificación** = Modificar de forma acotada los coeficientes puede generar
instancias cercanas con distinta estructura de soluciones óptimas.

**Qué hacer** = Establecer un rango de perturbación, generar nuevas instancias,
recalcular sus backbones y descartar cambios que no cumplan el criterio
experimental definido.

## 4. Aumentación por simetrías

**Justificación** = Permutar variables y restricciones conserva la estructura
matemática, pero entrega representaciones distintas del mismo problema.

**Qué hacer** = Aplicar permutaciones consistentes en OPB, etiquetas y grafo, y
comprobar que la predicción sea invariante ante dichas permutaciones.

## 5. Generación sintética condicionada

**Justificación** = Permite construir instancias con tamaños, familias y
distribuciones objetivo de `B0`, `B1` y `NB`.

**Qué hacer** = Definir parámetros de generación, resolver cada instancia con
el solver exacto, conservar solo las instancias válidas y estratificar las
particiones sin leakage.

## Orden sugerido de evaluación

Probar primero cada técnica por separado con el mismo baseline y las mismas
semillas. Después comparar solo las dos mejores combinaciones. Las pérdidas
ponderadas y `Focal Loss` deben tratarse como estrategias de entrenamiento
complementarias, no como aumentaciones de datos.
