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
ya tiene ejecución local. En esta corrida se derivaron las etiquetas de las
instancias transformadas mediante la regla algebraica B0↔B1 y NB→NB; no se
reextrajeron con GuroBack porque la transformación es biyectiva y preserva el
backbone bajo ese mapeo.

## Auditoría MIS: posibilidad de completar aumentaciones existentes (2026-11-24)

La partición original local de MIS contiene 761 instancias de train y 97 de
validation; `rebaseline-800-100-v1/MIS-Full` contiene 800/100. Los 39 archivos
de train y 3 de validation que faltaban ya están en MIS-Full. Las instancias y
etiquetas `.backbone` de las 761/97 compartidas coinciden byte a byte entre las
dos fuentes.

Los datasets `MIS-polarity-run-v2` y `MIS-permutation-run-v1` tienen cada uno
1522 filas de train (761 fuentes y 761 derivadas) y 97 de validation. Para
completarlos sin reconstruir las filas existentes habría que añadir, por
técnica, 39 filas fuente de train y sus 39 transformaciones (78 filas de train
en total), además de copiar 3 filas de validation. No se necesita reetiquetado
con solver: polaridad deriva las etiquetas intercambiando B0/B1, y permutación
renombra variables de instancia y backbone.

La ampliación de polaridad es determinista. En permutación, las 761 filas
existentes reproducen exactamente el generador con semilla 0 sobre el orden
antiguo; sin embargo, solo 13 de esas 761 coinciden al reproducir semilla 0
sobre el nuevo orden completo de 800 fuentes. Por ello, conservarlas y añadir
solo las filas ausentes requiere registrar la ampliación como backfill
incremental con su orden/semilla, y no describirla como una regeneración limpia
de toda la matriz con la semilla 0.

Los materializadores actuales rechazan directorios de salida ya existentes y
no implementan backfill. Si se autoriza completarlo, también habrá que actualizar
las particiones y preparar los grafos/cachés correspondientes para entrenamiento.
Esta revisión fue solo de lectura; no cambió datasets ni inició corridas.

## Auditoría integral de Drive: datos, aumentaciones y corridas (2026-11-24)

### Alcance y capacidad

Se revisó el proyecto `PBO Tesis` de la cuenta personal de Drive: `data/backpas`,
`runs`, `results` y `results/audits`. La carpeta del proyecto contiene una copia
del repositorio. Drive mostraba 36,79 GB usados de 400 GB en total durante la
revisión. No se movieron, renombraron ni cargaron archivos; el propósito fue
inventariar la situación antes de elegir una fuente de verdad.

Las particiones se anotan como `train/validation/test`. Las carpetas fuente
actuales de los tres problemas tienen 800/100/0 según sus manifiestos. Los
resultados históricos no siempre se entrenaron con esa versión completa.

### Inventario de datasets de Drive

| Problema/técnica | Carpeta o variante revisada | Particiones | Hallazgo |
|---|---|---:|---|
| MVC | fuente `MVC` | 800/100/0 | Base completa. |
| MVC | `MVC-polarity-run` | 1600/100/0 | Train equivale a fuentes más una fila transformada por fuente. Su auditoría Drive pasa con 1700 instancias, pero señala que falta `source.json`; no se encontró manifiesto de aumentación. |
| MVC | `MVC-permutation-run-v1` | 1600/100/0 | Dataset aumentado presente. |
| MVC | `MVC-selective-polarity-v1` | 800/100/0 | `augmentation.json` registra 0 generadas y 16.000 rechazadas. La carpeta no contiene las filas aumentadas esperadas. |
| MVC | `MVC-polarity` | sin conteo verificable en el manifiesto visible | Variante histórica; no tratar como canónica sin reconciliarla. |
| MVC | `MVC-backbones-imported` | 1600 instancias, 900 backbones, 700 errores | Auditoría Drive falla por 700 instancias sin asignación a partición; preservar como artefacto importado, no mezclar con la base de 900. |
| MIS | fuente `MIS` | 800/100/0 | La fuente de Drive está reparada/completa. Hay también `partitions.json.before_gurobi_repair.json`, que es histórico. |
| MIS | `MIS-selective-polarity-v1` | 1522/97/0 | 761 originales de train y 761 transformadas; usa la partición histórica 761/97, no la fuente actual 800/100. |
| MIS | `MIS-polarity-run-v2` | 1522/97/0 | Misma escala de partición histórica 761/97. |
| MIS | `MIS-polarity-run` | sin manifiesto de particiones | Variante histórica sin conteo verificable desde su carpeta. |
| MIS | `MIS-permutation-run-v1` | 1522/97/0 | Misma escala de partición histórica 761/97. |
| MIS | `MIS-constraint-perturbation-v1` | 1157/100/0 | 357 réplicas aceptadas, 8.860 rechazadas; conserva 800 fuentes de train más las réplicas aceptadas. No es una réplica completa por fuente. |
| MIS | `MIS-constraint-perturbation-v1.invalid-pilot100-attempts5` | 100/0/0 | Artefacto de intento piloto; excluir de resultados de tesis. |
| CA | fuente `CA` | 800/100/0 | Base completa en Drive. |
| CA | `CA-selective-polarity-v1` | 1576/100/0 | 788 fuentes y 788 transformadas; deriva de la partición histórica 788/100. |
| CA | `CA-polarity-run-v2` | 1576/100/0 | También corresponde a 788 fuentes más sus transformaciones; `CA-polarity-run` antiguo carece de manifiesto de particiones. |
| CA | `CA-permutation-run-v1` | 1576/100/0 | Deriva de 788 fuentes, no de las 800 actuales. |
| CA | `CA-constraint-perturbation-v1` | 800/100/0 | `augmentation.json` registra 0 generadas y 16.000 rechazadas: la carpeta no contiene filas de constraint augmentation. |

En el inventario de `data/backpas` no aparecen carpetas de
`objective-perturbation` ni `synthetic-conditioned` para MVC, MIS o CA. Tampoco
aparecen corridas de entrenamiento de esas dos técnicas en `runs` o `results`.
Sí aparecen como trabajos pendientes en la configuración local; eso documenta
una intención de trabajo, no una ejecución en Drive.

### Informes de auditoría preexistentes en Drive

| Informe | Dictamen encontrado | Reconciliación |
|---|---|---|
| `MVC-audit.md` | PASS, 900/900 etiquetas, 800/100, 0 errores. | Concuerda con la fuente MVC local; sus SHA-256 de instancias también coinciden. |
| `MVC-polarity-run-audit.md` | PASS, 1600 train + 100 validation; warning por falta de `source.json`. | Dataset presente, procedencia incompleta. |
| `MVC-backbones-imported-audit.md` | FAIL, 1600 archivos, 900 backbones y 700 instancias sin partición. | Variante importada con extras; no usar como fuente del benchmark. |
| `MIS-audit.md` | FAIL, marca 1500/1500 etiquetas inválidas al rechazar `nb xN`. | Parser del informe es antiguo; no detecta que el parser actual acepta NB explícito. Además, 38 etiquetas Drive carecen del marcador final. El MIS-Full local ya se auditó con el parser actual. |
| `CA-audit.md` | FAIL, 1200 archivos, 888 asignados con backbone y 312 sin asignación válida. | Incluye 300 archivos adicionales fuera de las particiones y 12 train que no estaban asignados en la fuente local antigua. CA-Full actual excluye los extras, añade los 12 asignados y pasa 900/900. |

### Corridas y artefactos guardados

| Problema | Evidencia de entrenamiento en Drive | Lectura de auditoría |
|---|---|---|
| MVC baseline | `results/MVC/baseline/seed-0`: métricas, logs y checkpoints; `experiment-status.json` indica interrupción tras 189/200 épocas. Partición 800/100/0. | Corrida interrumpida, no marcar como corrida final completa. |
| MVC polarity | `results/MVC/polarity/seed-0`: métricas, logs y checkpoints; partición 1600/100/0. | Hay artefactos de entrenamiento, pero falta un `status.json`/manifiesto equivalente en esa carpeta para cerrar su procedencia. |
| MVC selective-polarity | `runs/MVC/selective-polarity/seed-0` reporta `completed`, con métricas/logs/checkpoints; el dataset asociado `MVC-selective-polarity-v1` tiene partición 800/100/0 y registra 0 generadas. | El proceso terminó, pero no entrenó con ejemplos aumentados según el manifiesto del dataset. |
| MVC permutation | `results/MVC/permutation/seed-0`: métricas, logs y checkpoints, partición 1600/100/0. El manifiesto lo describe como “preliminary unaugmented validation baseline”. | No demuestra una evaluación final entrenada con permutación; conciliar antes de reportar. |
| MIS baseline (`full`) | métricas, logs y checkpoints; partición 761/97/0. | Entrenamiento histórico sobre la partición antigua, no sobre la fuente actual 800/100. |
| MIS selective-polarity | `runs/MIS/selective-polarity/seed-0` reporta `completed`, con métricas/logs/checkpoints; partición 1522/97/0. | Completó con la derivación antigua de 761/97; no cubre la fuente actual completa. |
| MIS permutation | `results/MIS/permutation/seed-0` tiene métricas, logs y checkpoints, pero el manifiesto la llama baseline preliminar sin aumentación y señala `dataset: MVC` pese a la ruta MIS. | Metadatos contradictorios; no contar como entrenamiento de permutación reportable. |
| MIS constraint-perturbation | `runs/MIS/constraint-perturbation/seed-0/status.json` indica `failed` en preparación, código 1. | No hay entrenamiento completado confirmado. |
| CA baseline (`full`) | métricas, logs y checkpoints; partición 788/100/0. | Baseline histórico incompleto frente a la fuente actual 800/100. |
| CA selective-polarity | `runs/CA/selective-polarity/seed-0` reporta `completed`, con métricas/logs/checkpoints; partición 1576/100/0. | Entrenó usando las 788 fuentes de la versión histórica, no la fuente actual completa. El resultado guardado además tiene F1/recall NB igual a 0. |
| CA permutation | `results/CA/permutation/seed-0` contiene logs de cola/proceso y particiones 1576/100/0, pero no `metrics.json` ni checkpoints. | No hay entrenamiento finalizado confirmado. |
| CA constraint-perturbation | `runs/CA/constraint-perturbation/seed-0` reporta `completed`, con métricas/logs/checkpoints; la partición es 800/100/0. | El entrenamiento terminó sobre las filas fuente, pero `generated=0`; no es evidencia de una aumentación de restricciones efectiva. NB aparece con cero soporte en las métricas. |

Los directorios `runs` y `results` no forman todavía un registro perfectamente
conciliado: algunos resultados no tienen estado o manifiesto, otros tienen
metadatos que contradicen su carpeta, y las carpetas `baseline` pueden estar
vacías mientras la corrida histórica está guardada como `full`. Por eso, la
presencia de un modelo o de `metrics.json` por sí sola no identifica de manera
segura la técnica ni la partición exacta.

### Estado de las técnicas y decisión operativa

En Drive hay artefactos para baseline, polarity, selective-polarity,
permutation y constraint-perturbation, con calidad de procedencia desigual.
Objective-perturbation y synthetic-conditioned no tienen datasets ni
entrenamientos encontrados. Entre las carpetas con nombre de aumentación,
`MVC-selective-polarity-v1` y `CA-constraint-perturbation-v1` reportan cero
instancias generadas; por tanto, sus entrenamientos no miden el efecto de esas
transformaciones. La constraint augmentation de MIS sí contiene 357 ejemplos,
pero su entrenamiento correspondiente falló.

Drive tiene espacio suficiente según la cuota observada, pero todavía no es una
fuente de verdad única: contiene bases completas junto con copias históricas
761/97 y 788/100, variantes duplicadas y resultados sin metadatos consistentes.
La recomendación es mantener intactas todas las copias y, antes de sincronizar
desde el PC, decidir una jerarquía canónica y conciliar cada dataset y corrida
con hashes, manifiestos, particiones y estados. Esta auditoría fue de solo
lectura; no se cambiaron archivos ni se lanzaron entrenamientos.

## Reconciliación PC–Drive (2026-11-24)

Se compararon las particiones del Drive personal con los manifiestos locales y
las copias aisladas de `rebaseline-800-100-v1`. Los conteos y huellas de los
nombres de archivos particionados dieron estos resultados:

| Problema | Drive contra PC | Diferencia útil | Dictamen |
|---|---|---|---|
| MVC | Mismos 800 train y 100 validation; además, los 900 SHA-256 de instancias de `source.json` en Drive coinciden con `data/manifests/MVC.json`. | Ninguna diferencia de partición ni de bytes de instancia detectada. | La fuente MVC puede conservarse como copia canónica compartida. El baseline entrenado está interrumpido; revisar los tratamientos antes de reportarlos. |
| MIS | Los nombres asignados a train/validation en Drive coinciden con los 800/100 de `data/manifests/MIS.json` y con `rebaseline-800-100-v1/MIS-Full`. | El manifiesto local amplio incluye 1500 archivos, 600 sin asignación; su auditoría anterior fallaba por las etiquetas. La copia aislada actual pasa auditoría 900/900. | Usar MIS-Full como candidata de trabajo; mantener su procedencia y resolver cuatro etiquetas pendientes de homogeneizar antes de declarar la fuente totalmente conciliada. |
| CA | La partición de Drive coincide con `rebaseline-800-100-v1/CA-Full` en 800/100. | El manifiesto local histórico asignaba 788 train. Los 12 nombres faltantes están en el PC con instancia y backbone; CA-Full los incorpora. | CA-Full pasa auditoría 900/900 y es la copia local completa que corresponde a Drive. La carpeta CA antigua sigue siendo histórica/incompleta como partición. |

Los 12 train de CA no asignados por el manifiesto local histórico son:
`train_instance_300_600_{43,67,126,146,251,331,363,460,516,540,561,641}.lp`.

### Estado actualizado de MIS-Full

La auditoría actual de `MIS-Full` terminó válida: 900 instancias, 800 train,
100 validation, 0 test, 900 backbones completos, 0 errores y 0 advertencias. Se
encontraron 38 logs de extracción, todos con retorno 0, mensaje de extracción
completa y runtime registrado. Las 38 etiquetas que estaban incompletas en
Drive ya tienen extracción local completa.

Quedan cuatro archivos de backbone cuya etiqueta actual es byte por byte igual
a la referencia de Drive, pero distinta de la etiqueta del MIS local previo y
sin log propio de reextracción con GuroBack:

- `train_easy_instance_1015.lp`
- `train_easy_instance_102.lp`
- `train_easy_instance_1021.lp`
- `train_easy_instance_1103.lp`

Por el criterio acordado de homogeneizar la procedencia, esas cuatro etiquetas
siguen pendientes de reextracción y validación con GuroBack. Esto no invalida
la auditoría estructural actual; distingue integridad/formato de procedencia.

La auditoría actual de `CA-Full` también pasó: 900 instancias, 800/100/0,
900 backbones completos, 0 errores y 0 advertencias. Los manifiestos estables
quedaron en `results/audits/MIS-Full-reconciliation-2026-11-24.json` y
`results/audits/CA-Full-reconciliation-2026-11-24.json`, con reportes `.txt`
homónimos.

### Jerarquía reconciliada propuesta

1. Mantener en Drive las fuentes MVC, MIS y CA y todos los derivados históricos;
   no sobrescribir ni borrar versiones.
2. Para MVC, conservar la fuente de 800/100, cuyas instancias coinciden con el
   manifiesto local por SHA-256.
3. Para MIS, conservar la fuente Drive y completar la homologación de las cuatro
   etiquetas indicadas en la copia aislada local. Luego sincronizar la copia
   auditada y sus logs/manifiesto como una nueva versión identificada.
4. Para CA, usar CA-Full como snapshot 800/100; archivar la partición histórica
   788/100 sin presentarla como completa.
5. Mantener polarity/permutation/constraint-perturbation bajo sus nombres y
   estados actuales. No incluir como aumentación efectiva los casos con cero
   ejemplos generados ni como corrida final las permutaciones con metadatos
   preliminares o incompletos.

### Cambios aplicados en Drive (2026-11-24)

Se creó `data/backpas/reconciled-800-100-v1-2026-11-24` en Drive. Se subieron
los ZIP completos de MIS-Full y CA-Full, auditorías TXT/JSON, checksums,
`CONTENIDO_Y_ESTADO.txt` y la bitácora PDF/LaTeX. Los ZIP se validaron localmente
antes de subir; cada uno contiene los 900 archivos LP y el contenido de su
carpeta de dataset, incluidos backbones, particiones, fuente y logs disponibles.

MIS se subió como snapshot estructuralmente válido, con cuatro etiquetas
marcadas explícitamente como pendientes de reextraer para homogeneizar su
procedencia. No debe tratarse como procedencia GuroBack completamente cerrada
hasta resolver esos cuatro casos. CA-Full representa la partición 800/100 e
incorpora los 12 train presentes en Drive que faltaban al manifiesto local
histórico.

El directorio `MIS-constraint-perturbation-v1.invalid-pilot100-attempts5` se
movió desde la raíz activa de `data/backpas` a `data/backpas/nonreportable-archive`.
No se eliminó definitivamente. Las fuentes originales, aumentaciones,
entrenamientos y auditorías históricas restantes se conservaron sin cambios.
