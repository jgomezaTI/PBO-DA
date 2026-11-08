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

## Registro maestro de notebooks Colab

Este inventario es la referencia para saber qué técnica/notebook se preparó y
qué corrida tiene evidencia suficiente. `No verificado` significa que no se
debe contar como entrenamiento terminado hasta revisar el `status.json`, el
log de épocas y los checkpoints del directorio exacto del trabajo.

| Problema | Técnica | Notebook | URL | Estado/evidencia |
|---|---|---|---|---|
| MVC | selective-polarity | `notebooks/colab-techniques/PBO-selective-polarity.ipynb` | [Abrir en Colab](https://colab.research.google.com/drive/15rahSEma9nFzCUYICZUtTW_LZGxmrWDQ) | **Ejecución reportada como terminada; artefactos pendientes de verificar**. La salida de la celda dice `Completado(s): MVC/selective-polarity/seed-0`; el setup informa MVC bruto 900/900 y transformación 800 instancias (train=1600, validation=100), con GTR 8 capas, 65 épocas, batch 64 en GPU. Falta revisar `status.json`, log de épocas y checkpoints en la cuenta `javierivmedioa@gmail.com`. No relanzar hasta comprobarlos. |
| — | constraint-perturbation | `notebooks/colab-techniques/PBO-constraint-perturbation.ipynb` | Pendiente: no se confirmó copia/enlace de Colab | Preparado localmente; no hay ejecución verificada. |
| — | objective-perturbation | `notebooks/colab-techniques/PBO-objective-perturbation.ipynb` | Pendiente: no se confirmó copia/enlace de Colab | Preparado localmente; no hay ejecución verificada. |
| — | synthetic-conditioned | `notebooks/colab-techniques/PBO-synthetic-conditioned.ipynb` | Pendiente: no se confirmó copia/enlace de Colab | Preparado localmente; no hay ejecución verificada. |

Al actualizar una fila, registrar el enlace directo, el ID de notebook, el job
exacto (`<problema>/<técnica>/seed-<semilla>`), dataset fuente/derivado, modelo y
configuración, fecha, estado y evidencia (status, épocas y checkpoints). No
marcar `completado` por el mensaje del runner solamente. Mantener separadas las
cuentas de Drive y registrar cuál contiene el notebook y cuál los artefactos.
Las tres técnicas listadas con problema `—` abarcan MVC, MIS y CA según
`configs/pending_augmentation_jobs.json`; la técnica del notebook se ejecuta
para un problema elegido en su celda `JOB`.
El conector de Drive disponible para esta sesión está autenticado como
`jgomeza@institutoroi.com`, no como `javierivmedioa@gmail.com`; cualquier carpeta
vacía observada con ese conector no demuestra que falten archivos en el Drive
personal. El notebook local de selective-polarity quedó fijado en MVC y ahora
trata los estados `completed`, activos, `failed` y desconocidos como no
ejecutables automáticamente. La vista actual del notebook online aún renderiza
dos viñetas vacías en el encabezado; la copia local limpia está corregida, pero
no se sincronizó a Colab porque el conector de Colab devuelve `Transport closed`.
La celda de entrenamiento original solo leía `augmentation.json` antes de
llamar a `cloud_run.py`; por eso no podía crear el dataset si faltaba. La copia
local ahora ejecuta primero `scripts/materialize_perturbation.py --kind
selective-polarity`, valida el manifiesto y las carpetas `instance/` y
`backbone/`, y después inicia el trainer. Para MVC (`.lp`), el generador invierte
la polaridad de una selección de variables y deriva las etiquetas con B0↔B1 en
las variables seleccionadas y NB conservado.

## Trabajo pendiente

La matriz `configs/pending_augmentation_jobs.json` cubre cuatro técnicas aún no
entrenadas (polaridad selectiva, perturbación de restricciones, perturbación de
objetivo y generación condicionada) sobre MVC, MIS y CA. Los datasets se generan
por trabajo en Colab y los checkpoints se guardan en
`PBO-Tesis/runs/<problema>/<técnica>/seed-<semilla>`. Para selective-polarity se
partió de instancias y archivos `.backbone` ya etiquetados en el dataset fuente:
el proceso de Colab no ejecutó Gurobi ni GuroBack para volver a etiquetar. La
transformación conserva la procedencia de esas etiquetas mediante la regla
algebraica B0↔B1 para las variables invertidas y NB→NB. Otras perturbaciones
pueden necesitar resolver las instancias transformadas para obtener etiquetas
nuevas; el solver concreto depende del formato y está indicado en el manifiesto
de generación.

## Corrida CA / selective-polarity / seed-0

| Campo | Registro |
|---|---|
| Trabajo | `CA/selective-polarity/seed-0` |
| Dataset fuente configurado | `data/backpas/CA` |
| Dataset preparado para el trabajo | `data/backpas/CA-selective-polarity-v1` |
| Transformación | Una réplica de selective-polarity, semilla 0; aplicada solo a train |
| Etiquetado base | Archivos `.backbone` existentes en el dataset `CA` cargado en Colab; no se invocó Gurobi/GuroBack durante esta corrida para reetiquetar. |
| Etiquetas aumentadas | Se parte de las etiquetas base; B0/B1 se intercambian para las variables invertidas y NB se conserva. Validation no se transforma. |
| Ejecutor y modelo | BackPaS, GTR, 8 capas, 65 épocas solicitadas, GPU NVIDIA A100 80 GB |
| Parámetros registrados por el comando | `threads=4`, `batch_size=64`, `batch_accumulation_size=64`, `seed=0`, CUDA |
| Estado observado en Colab | `completed`, `return_code=0`; `training_log.csv` terminó con train y valid de épocas 63 y 64 (65 épocas, numeradas desde 0) |
| Checkpoints observados | `best_model.pth`, `last_model.pth` y `last_optimizer.pth` presentes en el directorio del trabajo de Drive |

El log de entrenamiento mostró que Colab **reutilizó** el dataset generado
`CA-selective-polarity-v1`; no entrenó directamente sobre `CA-polarity-run`,
`CA-polarity-run-v2` ni `CA-permutation-run-v1`. El notebook restaura el bundle
fuente del repositorio y los archivos `MVC`, `MIS` y `CA` desde Drive antes de
generar el trabajo. La configuración identifica `CA` como fuente, pero la
corrida no registró hashes de los archivos fuente o derivados, así que no queda
demostrada la identidad byte a byte entre esos archivos de Drive y las carpetas
que están actualmente en este PC.

La salida también informó `No test partition found`; por tanto, esta corrida no
deja evidencia de evaluación en test. La auditoría de clasificación mostró
recall y F1 de NB iguales a 0, así que el resultado no permite afirmar que el
modelo haya aprendido a detectar NB en una partición evaluable. El entrenamiento
terminó correctamente, pero esa limitación debe quedar presente al interpretar
la corrida.

La inversión selectiva es una variante nueva; no se confunde con polarity, que
ya tiene ejecución local. GuroBack/Gurobi pertenece al flujo de extracción de
las etiquetas base del dataset original; no fue el etiquetador de las instancias
selective-polarity de esta corrida.
