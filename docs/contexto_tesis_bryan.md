# Contexto de la tesis de Bryan Alvarado

## Alcance de este documento

Este archivo resume el contenido de `Tesis_Bryan_Alvarado.pdf`, cuya portada
indica el título **Backbone-based Predict and Search for Pseudo-Boolean
Optimization**, autoría de Bryan Jesús Alvarado Ulloa, Universidad Técnica
Federico Santa María, Departamento de Informática, con fecha 2 de marzo de 2026.

El resumen se mantiene dentro de lo que aparece en la tesis. No se presentan
como hechos resultados, datasets, técnicas o decisiones que no estén descritos
en ella. En el índice y los capítulos de la tesis no aparece un capítulo
dedicado a aumentación de datos.

## Identificación y propósito

- **Título:** Backbone-based Predict and Search for Pseudo-Boolean Optimization.
- **Autor:** Bryan Jesús Alvarado Ulloa.
- **Institución:** Universidad Técnica Federico Santa María, Departamento de
  Informática.
- **Profesor guía:** Roberto Asín A., Ph.D.
- **Profesor co-guía:** Ricardo Ñanculef A., Ph.D.
- **Problema general:** estudiar la integración de aprendizaje automático con
  problemas de optimización combinatoria formulados como Pseudo-Boolean
  Optimization (PBO).
- **Propuesta:** BackPaS, una adaptación de Predict-and-Search (PaS) para PBO
  basada en la predicción de backbones.

## Idea central

La tesis parte de que las técnicas de aprendizaje automático pueden aprender
patrones de instancias que se resuelven repetidamente y usar esos patrones para
guiar solucionadores clásicos. La dificultad es que una predicción incorrecta
puede producir soluciones infactibles, perder calidad o comprometer las
garantías de optimalidad.

PaS separa el proceso en dos etapas:

1. **Prediction:** un modelo predice información sobre la instancia.
2. **Search:** el solucionador usa esa información para restringir su espacio
   de búsqueda.

BackPaS reemplaza las etiquetas basadas en soluciones factibles de alta calidad
por etiquetas de backbone. La tesis define el backbone como las variables cuya
asignación es igual en todas las soluciones óptimas de una instancia.

La predicción se formula como clasificación de tres clases por variable:

- **B0:** la variable está fijada a 0 en todas las soluciones óptimas.
- **B1:** la variable está fijada a 1 en todas las soluciones óptimas.
- **NB:** la variable no pertenece al backbone.

Las predicciones se incorporan al solucionador mediante una región de confianza
(*trust region*) adaptativa.

## Hipótesis y objetivos de Bryan

### Hipótesis

La tesis plantea que predecir backbones en problemas PBO mediante una GNN,
integrada dentro de un marco PaS, mejorará la métrica **Primal Integral** en
comparación con enfoques PaS anteriores.

### Objetivo general

Desarrollar un enfoque para PBO que use Redes Neuronales de Grafos (GNN) para
predecir backbones dentro de PaS, con el objetivo de mejorar la Primal Integral.

### Objetivos específicos

1. Diseñar e implementar una herramienta de extracción de backbones para
   instancias PBO, generando etiquetas para entrenamiento y evaluación.
2. Desarrollar y entrenar una GNN capaz de predecir variables del backbone en
   instancias PBO.
3. Integrar las predicciones de la GNN en un marco PaS para guiar la búsqueda
   del solucionador.
4. Evaluar rigurosamente BackPaS frente a solucionadores y métodos de aprendizaje
   existentes.

## Fundamentos técnicos

### Programación por restricciones

La tesis presenta Constraint Programming como un paradigma declarativo: el
usuario describe las propiedades de una solución mediante variables, dominios y
restricciones, y un solucionador busca una asignación que las satisfaga.

Distingue:

- **CSP:** busca determinar si existe una asignación factible.
- **COP:** agrega una función objetivo para escoger la mejor asignación factible.

### SAT, PBS y PBO

SAT trabaja con variables binarias y cláusulas formadas por disyunciones de
literales. El modelo completo se expresa en forma normal conjuntiva.

Pseudo-Boolean Satisfaction (PBS) mantiene variables binarias, pero permite
operaciones aritméticas sobre ellas. Pseudo-Boolean Optimization (PBO) agrega
una función objetivo que se minimiza o maximiza.

La tesis distingue tres formulaciones:

- **Formulación literal general:** permite monomios y restricciones no lineales.
- **Formulación literal lineal:** restringe los monomios a literales
  individuales.
- **Formulación lineal basada en variables:** utiliza variables binarias y
  restricciones lineales.

El trabajo se concentra en PBO lineal. La tesis indica que una formulación PBO
no lineal puede transformarse en una PBO lineal en tiempo polinomial; desde ese
punto, el término PBO se utiliza para referirse a la formulación lineal salvo
que se indique lo contrario.

### Solucionadores y búsqueda

Los solucionadores exactos garantizan encontrar el óptimo global o demostrar que
no existe una solución factible. Los solucionadores incompletos priorizan
encontrar soluciones buenas rápidamente sin garantizar optimalidad.

Durante la ejecución, la mejor solución factible encontrada se denomina
*incumbent*. La evolución de la calidad del incumbent es el comportamiento
*anytime*.

La tesis menciona dos motores algorítmicos:

- **CDCL:** búsqueda con retroceso no cronológico y aprendizaje de cláusulas
  después de conflictos; se asocia principalmente a SAT y también puede usarse
  en PBO.
- **Branch-and-Cut:** extensión de Branch-and-Bound que agrega cortes para
  eliminar regiones fraccionarias sin eliminar soluciones enteras; se mencionan
  SCIP y Gurobi.

### Primal Integral

La Primal Integral mide cómo se acerca el incumbent a la mejor solución conocida
durante un límite temporal. Integra la brecha primal en el intervalo de tiempo.
Un valor menor indica que se encuentran soluciones de buena calidad más rápido.

En la evaluación de Bryan se utiliza un límite de **1000 segundos**. El mejor
valor conocido se define como el mejor resultado obtenido entre todos los
métodos probados dentro de ese límite.

### Backbone

Para una instancia PBO, sea `F` el conjunto factible, `f*` el valor óptimo y
`O` el conjunto de soluciones factibles con valor `f*`. Una variable pertenece al
backbone si toma el mismo valor binario en todas las soluciones de `O`.

La tesis señala que:

- el backbone identifica la parte fija del espacio de soluciones óptimas;
- calcularlo es co-NP-completo en general;
- su relación con la dificultad depende del dominio y del problema;
- puede usarse para reducir o guiar el espacio de búsqueda.

## Representaciones de grafos

La tesis utiliza grafos bipartitos con dos conjuntos de nodos: variables o
literales, y restricciones. Una arista indica que una variable o literal
participa en una restricción. La función objetivo se incorpora como un nodo de
restricción especializado.

Se comparan dos representaciones:

- **Variable-based:** un nodo por variable.
- **Literal-based:** dos nodos por variable, correspondientes al literal
  positivo y al negativo.

Para la representación literal, los coeficientes se normalizan para que sean no
negativos, transformando variables con coeficientes negativos mediante su literal
complementario y ajustando el término constante o el lado derecho.

## Revisión de literatura de Bryan

### Aprendizaje automático para optimización combinatoria

La tesis organiza los enfoques según su relación con el solucionador:

1. **End-to-end:** el modelo recibe la instancia y produce directamente una
   solución o una predicción global. Se menciona NeuroSAT. La limitación
   señalada es la ausencia de garantías formales de factibilidad y optimalidad.
2. **Online integration:** el modelo se invoca durante la búsqueda para guiar
   decisiones internas. Se describe una GNN que imita Strong Branching dentro de
   Branch-and-Bound. El costo de inferencia repetida puede ser alto.
3. **Offline integration:** el modelo se ejecuta antes del solucionador y sus
   predicciones se convierten en restricciones auxiliares. PaS pertenece a esta
   categoría.

### Predict-and-Search

El PaS original predice valores de variables binarias mediante una GNN sobre una
representación bipartita basada en variables. Se entrena con soluciones factibles
de alta calidad y usa una pérdida Cross-Entropy ponderada por la calidad de cada
solución.

En la etapa de búsqueda, selecciona las variables con las predicciones más altas
para 1 y más bajas para 0. Agrega variables indicadoras de desviación y limita la
cantidad total de desviaciones permitidas mediante los parámetros `k0`, `k1` y
`Delta`.

### Contrastive Predict-and-Search

ConPaS incorpora aprendizaje contrastivo para incluir señales positivas y
negativas. La tesis distingue:

- **ConPaS-Inf:** usa soluciones infactibles como ejemplos negativos.
- **ConPaS-LQ:** usa soluciones factibles de baja calidad como ejemplos
  negativos.

Bryan utiliza ConPaS-LQ como baseline principal. La arquitectura GNN, la
representación del grafo y la construcción de la región de confianza se mantienen
como en PaS; cambian principalmente la construcción de datos y la pérdida.

### Backbones en búsqueda y optimización

La tesis menciona trabajos que estiman backbones en SAT y usan esas predicciones
para guiar el solucionador. También describe CadiBack para SAT y GuroBack para
PBO.

GuroBack usa Gurobi para certificar si las asignaciones de variables pueden
cambiar manteniendo el valor óptimo. La tesis utiliza GuroBack para generar las
etiquetas de entrenamiento de BackPaS.

## BackPaS

BackPaS modifica PaS en cuatro aspectos:

1. Usa variables del backbone como señal de entrenamiento, en lugar de soluciones
   de alta o baja calidad.
2. Usa una representación bipartita basada en literales.
3. Usa una GNN más profunda con capas Graph Transformer.
4. Usa una parametrización adaptativa de la región de confianza basada en la
   confianza del modelo.

La diferencia conceptual destacada es que cada muestra de entrenamiento contiene
información sobre la estructura global del conjunto de soluciones óptimas, no solo
sobre la calidad de una asignación individual.

### Extracción de etiquetas con GuroBack

El procedimiento descrito en la tesis es iterativo:

1. GuroBack resuelve la instancia original con Gurobi y obtiene una solución
   óptima `v*`.
2. Se crea el conjunto candidato con todas las variables binarias.
3. Se selecciona un lote de candidatos.
4. Se resuelve una instancia modificada que conserva el objetivo óptimo y exige
   que al menos una variable del lote cambie respecto de `v*`.
5. Si la instancia modificada es infactible, las variables del lote se confirman
   como backbone y el tamaño del lote aumenta.
6. Si es factible, se eliminan de los candidatos las variables que cambiaron y el
   tamaño del lote vuelve a uno.
7. El proceso termina cuando no quedan candidatos.

El conjunto restante corresponde al backbone completo de la instancia PBO.

### GNN y características

La GNN opera sobre el grafo bipartito literal. Cada ciclo tiene dos pasos:

1. **Variable-to-Constraint:** actualiza nodos de restricción usando nodos
   literales.
2. **Constraint-to-Variable:** actualiza nodos literales usando los nodos de
   restricción actualizados.

Las capas son Bipartite Transformer con atención multi-cabeza, conexión residual y
normalización de capa. La arquitectura final utiliza **8 capas Bipartite
Transformer completas**, equivalentes a 16 pasos intercalados.

Las características iniciales descritas son:

- **Literales:** coeficiente normalizado en el objetivo, media, mínimo y máximo
  de coeficientes en restricciones, número de restricciones y número de
  restricciones de la variable correspondiente.
- **Restricciones:** promedio de coeficientes, grado del nodo, lado derecho y
  sentido de la restricción.
- **Aristas:** coeficiente del literal en la restricción.

La salida predice una distribución de probabilidad para B0, B1 y NB.

### Pérdida Mean Cross-Entropy

La tesis usa Mean Cross-Entropy (MCE) porque las instancias tienen tamaños
distintos. Primero calcula la pérdida promedio sobre las variables de cada grafo y
luego promedia los grafos. De esta forma, una instancia grande no domina la
pérdida total solamente por tener más variables.

### Región de confianza adaptativa

La región de confianza se controla mediante dos parámetros:

- `theta` en `[0, 1]`: confianza mínima para seleccionar una variable.
- `alpha` en `[-1, 1]`: ajuste de la tolerancia según la confianza del modelo y
  la distribución de instancias.

Se seleccionan las variables cuya mayor probabilidad entre B0 y B1 supera
`theta`. Cada variable seleccionada recibe el valor de su clase backbone más
probable. La tolerancia `Delta` se estima a partir de la incertidumbre acumulada.

La parametrización es independiente del tamaño de la instancia. La tesis compara
esta parametrización con los parámetros dependientes del tamaño `k0`, `k1` y
`Delta` usados por PaS.

## Variantes y baselines

### Variantes de BackPaS

- **BackPaS-V0:** solo cambia la tarea de predicción a B0/B1/NB; conserva la
  arquitectura, la representación y los parámetros de confianza originales.
- **BackPaS-Net:** añade la arquitectura especializada de 8 capas Graph
  Transformer sobre grafos literales.
- **BackPaS-Param:** añade la parametrización adaptativa de la región de
  confianza, pero conserva la arquitectura base.
- **BackPaS:** integra predicción de backbone, arquitectura especializada y
  parametrización adaptativa.

### Baselines

- **Gurobi:** resuelve las instancias originales sin modificaciones.
- **ConPaS:** baseline de Predict-and-Search basado en aprendizaje contrastivo,
  usando la variante ConPaS-LQ.

## Entorno experimental de la tesis

### Hardware

- Tareas intensivas en CPU: clúster con procesadores Intel Xeon E5-2670 v3 y
  64 GB de RAM.
- Tareas intensivas en GPU: NVIDIA A40 con 48 GB de VRAM.
- Sistema operativo: Rocky Linux 8.

### Software

- Gurobi v10.0.0rc2.
- PyTorch v2.7.1 con CUDA 12.8.
- PyTorch Geometric v2.6.1.
- GuroBack y BackPaS desde sus repositorios públicos.

### Asignación de recursos

Para la extracción de backbones con GuroBack y la resolución base con Gurobi se
procesaron hasta 12 instancias en paralelo, con un núcleo de CPU y 5 GB de RAM
por instancia. La extracción de soluciones para ConPaS también usó hasta 12
instancias en paralelo, sin límite de RAM explícito.

## Datasets y particiones

La tesis evalúa tres benchmarks PBO:

| Benchmark | Partición BackPaS | Características | Origen indicado |
| --- | --- | --- | --- |
| MIS | train/val-backpas | 1000 nodos, grado promedio 4 | D-MIPLIB |
| MVC | train/val-backpas | 1200 nodos, grado promedio 5 | D-MIPLIB |
| CA | train/val-backpas | 300 ítems, 600 bids | Generado |

También se describen particiones para ConPaS, validación y test:

- **train-backpas / val-backpas:** 800 y 100 instancias, respectivamente, con
  backbones precomputados. Se selecciona para inferencia el checkpoint con menor
  Cross-Entropy de validación.
- **train-conpas / val-conpas:** 400 y 100 instancias, respectivamente, con
  etiquetas positivas y negativas obtenidas mediante ConPaS.
- **val:** 100 instancias para ajustar los hiperparámetros de BackPaS.
- **test:** 100 instancias para la evaluación final de Primal Integral.

La tesis indica que las particiones D-MIPLIB se usan cuando están disponibles;
en otros casos utiliza instancias generadas con generadores de D-MIPLIB y la
biblioteca GeCo para MIS.

También indica que MIS y MVC se basan en grafos Barabási-Albert y que CA sigue la
distribución Arbitrary.

La extracción de backbones para `train-backpas` y `val-backpas` requirió,
según la tesis, aproximadamente:

- CA: 55 horas de CPU.
- MIS: 279 horas de CPU.
- MVC: 12 horas de CPU.

## Métricas y evaluación

La evaluación usa dos niveles:

1. **Pipeline completo:** Primal Integral con límite de 1000 segundos.
2. **Predicción de la GNN:** Mean Cross-Entropy sobre `val-backpas`.

El tiempo de inferencia de la GNN y de construcción de la región de confianza se
excluye de la Primal Integral para todos los métodos.

Los hiperparámetros de la región de confianza se ajustan con optimización
bayesiana basada en Tree-structured Parzen Estimators (TPE). El proceso comienza
con 30 puntos aleatorios y continúa con tres lotes secuenciales de 10 puntos.

## Resultados reportados por Bryan

### Primal Integral en test

La tabla reportada presenta media y desviación estándar; un valor menor es mejor.

| Método | CA | MIS | MVC |
| --- | ---: | ---: | ---: |
| Gurobi | 20.74 (6.08) | 44.09 (7.73) | 7.79 (1.15) |
| ConPaS | 19.50 (7.36) | 8.93 (1.94) | 6.83 (2.53) |
| BackPaS | 16.09 (6.79) | 0.82 (0.53) | 0.49 (0.50) |
| BackPaS-Param | 30.13 (9.54) | 4.09 (1.44) | 1.50 (0.45) |
| BackPaS-Net | 24.05 (10.22) | 1.13 (0.17) | 1.74 (0.49) |
| BackPaS-V0 | 17.11 (6.79) | 2.31 (0.61) | 2.20 (0.44) |

La tesis señala que BackPaS obtuvo el menor PI en los tres benchmarks y que las
diferencias fueron evaluadas con pruebas de Wilcoxon pareadas (`p < 0.05`).

También describe que:

- BackPaS-V0 ya supera a Gurobi y ConPaS en los tres benchmarks.
- En MIS, agregar la arquitectura reduce PI de 2.31 a 1.13 frente a BackPaS-V0,
  una reducción indicada como 51%.
- En MVC, el PI de BackPaS-Net es 1.74 frente a 2.20 de BackPaS-V0.
- En el modelo completo, MIS alcanza 0.82 y la tesis indica una mejora de 80%
  respecto de BackPaS-Param.
- Los parámetros adaptativos por sí solos pueden empeorar CA y MIS, pero
  muestran mejores resultados junto con la arquitectura especializada.

### Comparación de arquitecturas

Se comparan:

- capas GCN frente a GTR;
- profundidades de 2, 4, 6 y 8 capas;
- representación basada en variables frente a representación basada en
  literales.

La figura reportada muestra menores pérdidas para GTR y grafos basados en
literales. La configuración de 8 capas GTR con representación literal aparece
como la mejor entre las configuraciones comparadas.

### Generalización en MIS

La tesis estudia generalización en grafos Barabási-Albert de mayor tamaño y
densidad. El modelo se entrena en MIS-666, MIS-1000 y MIS-1333. Se evalúa en:

| Benchmark | Partición | Característica |
| --- | --- | --- |
| MIS-666 | train, val | 666 nodos, grado promedio 2 |
| MIS-1000 | train, val | 1000 nodos, grado promedio 3 |
| MIS-1333 | train, val | 1333 nodos, grado promedio 4 |
| MIS-1666 | val, test | 1666 nodos, grado promedio 5 |
| MIS-2000-6 | val, test | 2000 nodos, grado promedio 6 |
| MIS-2000-10 | val, test | 2000 nodos, grado promedio 10 |

Los resultados reportados son:

| Método | MIS-1666 | MIS-2000-6 | MIS-2000-10 |
| --- | ---: | ---: | ---: |
| Gurobi | 3.81 (1.58) | 14.94 (3.57) | 32.64 (6.56) |
| BackPaS | 0.04 (0.17) | 0.19 (0.25) | - |

La tesis indica que BackPaS supera a Gurobi en MIS-1666 y MIS-2000-6, pero no
generaliza a MIS-2000-10 porque el modelo no alcanzó suficiente confianza para
seleccionar un backbone.

## Conclusiones de Bryan

La tesis concluye que el modelo completo combina cuatro elementos:

1. supervisión basada en backbones;
2. capas Graph Transformer;
3. representación bipartita basada en literales;
4. región de confianza adaptativa.

Según la tesis, el estudio de ablación muestra que estos componentes son
complementarios. La hipótesis se declara validada porque el modelo completo
obtiene una mejora significativa de Primal Integral frente a enfoques PaS
anteriores.

Los objetivos se declaran cumplidos mediante:

- una herramienta de extracción de backbones;
- una GNN para predecir variables backbone;
- la integración de predicciones en una región de confianza adaptativa;
- la evaluación en MIS, CA y MVC.

La sección de trabajo futuro de Bryan menciona extender BackPaS a clases más
amplias de optimización combinatoria y desarrollar esquemas de ponderación para
variables backbone según su importancia para el solucionador.

## Fuente consultada

- Archivo local: `Tesis_Bryan_Alvarado.pdf`.
- Documento analizado: 57 páginas.
- Este Markdown es una síntesis de ese documento; no reemplaza la cita de la
  tesis en la memoria.
