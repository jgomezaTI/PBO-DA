# Entrenamiento paralelo en nube gratuita

La matriz de trabajos está en [`configs/cloud_jobs.json`](../configs/cloud_jobs.json).
Cada fila es independiente y usa el directorio:

```text
results/<problema>/<técnica>/seed-<semilla>/
```

## Colab + Drive

1. Abrir `notebooks/colab_parallel_training.ipynb` en Colab.
2. Montar el mismo Drive en cada sesión.
3. Seleccionar una fila distinta de la matriz en cada sesión.
4. Ejecutar el notebook. Si la sesión se corta, repetir con `--resume`.

El runner copia el estado previo desde Drive, ejecuta un solo trabajo y vuelve a
sincronizar checkpoints, logs, métricas y `status.json`. No se deben ejecutar dos
sesiones sobre la misma combinación problema/técnica/semilla.

## Orden experimental

Primero completar `baseline` y `polarity` en MVC. Después repetir el mismo par
para MIS y CA cuando sus datasets hayan sido auditados. La validación y cualquier
test permanecen sin aumentación; la matriz no autoriza a cambiar arquitectura,
particiones, semilla o criterio de checkpoint entre tratamientos.

## Comandos locales/Colab

```bash
python scripts/cloud_run.py --job MVC/baseline/seed-0 --drive-root /content/drive/MyDrive/PBO-Tesis
python scripts/cloud_run.py --job MVC/baseline/seed-0 --drive-root /content/drive/MyDrive/PBO-Tesis --resume
python scripts/cloud_run.py --job MVC/polarity/seed-0 --drive-root /content/drive/MyDrive/PBO-Tesis --dry-run
```

## Lanzamiento simultáneo

Las técnicas materializadas en la matriz pueden ejecutarse en paralelo con
`scripts/run_parallel_jobs.py`. Cada trabajo conserva su propio directorio de
checkpoints y `--resume` permanece activo por defecto:

```bash
python scripts/run_parallel_jobs.py \
  MVC/polarity/seed-0 MVC/permutation/seed-0 \
  MIS/polarity/seed-0 MIS/permutation/seed-0 \
  CA/polarity/seed-0 CA/permutation/seed-0 \
  --drive-root /content/drive/MyDrive/PBO-Tesis
```

En Colab se recomienda abrir una sesión por trabajo para evitar contención de
GPU. Los `baseline` no forman parte de este lanzamiento porque ya están
entrenados.
