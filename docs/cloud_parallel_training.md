# Entrenamiento paralelo en nube gratuita

La matriz histórica está en [`configs/cloud_jobs.json`](../configs/cloud_jobs.json).
Para el trabajo nuevo, usar `configs/pending_augmentation_jobs.json`, generada
por `python scripts/materialize_pending_augmentations.py`.
Cada fila es independiente y usa el directorio:

```text
results/<problema>/<técnica>/seed-<semilla>/
```

## Colab + Drive

1. Abrir los cuatro notebooks maestros, uno por técnica, cada uno en su propia
   pestaña y runtime de Colab: selective-polarity, constraint-perturbation,
   objective-perturbation y synthetic-conditioned.
2. En cada runtime, ejecutar la celda de configuración y elegir **un solo
   problema** (MVC, MIS o CA). Así quedan cuatro trabajos distintos ejecutándose
   en paralelo, cada uno en su propia sesión; no se combinan técnicas dentro de
   un notebook ni de un proceso.
3. Ejecutar la celda de entrenamiento correspondiente. No asignar a dos
   runtimes el mismo problema/técnica/semilla.
4. Cada runtime solicita GPU T4 y usa CPU si Colab no asigna GPU. Colab puede
   limitar las sesiones concurrentes de la cuenta. Si no hay cuatro runtimes
   disponibles, iniciar tantos trabajos distintos como runtimes conceda y
   arrancar los restantes en cuanto uno quede libre. No detener un entrenamiento
   activo para abrir otra ventana; sus checkpoints quedan en Drive.

Hay cuatro cuadernos maestros, uno por técnica, para las doce combinaciones
pendientes (cuatro técnicas por tres benchmarks). Cada notebook deja elegir MVC,
MIS o CA, y lanza un solo trabajo. Para el objetivo de cuatro entrenamientos
concurrentes, seleccionar un problema distinto en cada notebook y conectar cada
runtime por separado. Los doce trabajos son combinaciones pendientes disponibles
para distribuir en tandas; no se pretende ejecutar los doce simultáneamente. El paquete autocontenido
`artifacts/colab-techniques/pbo-colab-techniques.tar.gz` contiene los cuatro
notebooks. Cada sesión combina el código ya publicado en
`PBO-Tesis/runs/pbo-tesis-source.tar.gz` con ese notebook. La matriz correspondiente
se guarda en `PBO-Tesis/runs/pending_augmentation_jobs.json`. Los generadores de
perturbaciones escriben una carpeta parcial reanudable y publican el dataset
cuando terminan; los checkpoints permanecen en Drive.

El runner escribe checkpoints, logs, métricas y `status.json` directamente en
Drive y detecta checkpoints para reanudar una sesión interrumpida. No se deben ejecutar dos
sesiones sobre la misma combinación problema/técnica/semilla.

## Orden experimental

Baseline/full, polarity y permutation ya tienen entrenamiento local para MVC,
MIS y CA. No se relanzan. La matriz pendiente compara cuatro técnicas nuevas:
polaridad selectiva, perturbación de restricciones, perturbación del objetivo y
generación condicionada. La validación y test permanecen intactos.

## Comandos locales/Colab

```bash
python scripts/materialize_pending_augmentations.py
python scripts/cloud_run.py --matrix configs/pending_augmentation_jobs.json --job MVC/selective-polarity/seed-0 --drive-root /content/drive/MyDrive/PBO-Tesis
```

## Lanzamiento simultáneo

Las técnicas materializadas en la matriz pueden ejecutarse en paralelo con
`scripts/run_parallel_jobs.py`. Cada trabajo conserva su propio directorio de
checkpoints; una reanudación explícita usa `--resume`:

```bash
python scripts/run_parallel_jobs.py \
  MVC/selective-polarity/seed-0 MVC/constraint-perturbation/seed-0 \
  MIS/objective-perturbation/seed-0 CA/synthetic-conditioned/seed-0 \
  --matrix configs/pending_augmentation_jobs.json \
  --drive-root /content/drive/MyDrive/PBO-Tesis
```

En Colab, la configuración objetivo es una sesión por técnica (cuatro sesiones
simultáneas), cada una con un trabajo problema/técnica/semilla. Si la cuenta
rechaza una conexión por el límite de sesiones, mantener los trabajos activos y
conectar la técnica pendiente cuando Colab libere capacidad; el estado y los
checkpoints en Drive permiten continuar después. Los `baseline` no forman parte
de este lanzamiento porque ya están entrenados.
