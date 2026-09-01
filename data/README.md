# Datos

Los datasets no se versionan directamente en Git. Antes de ejecutar experimentos se
debe registrar, como mínimo:

- fuente y licencia;
- versión o fecha de obtención;
- familias y número de instancias;
- checksums de los archivos originales;
- formato de los backbones;
- particiones train/validation/test y su seed;
- procedimiento exacto de generación o transformación.

La aumentación se aplica **después** de crear las particiones y solamente a train para
evitar leakage entre una instancia y su versión invertida.

## Estado del repositorio BackPaS

El repositorio público de BackPaS contiene el código de procesamiento y un archivo
`dataset/dataset.txt`, pero no contiene las instancias ni los archivos `.backbone` de
los experimentos. Su README indica que las instancias deben ubicarse manualmente en
`dataset/DATASET_NAME/instance/`; los backbones pueden colocarse en
`dataset/DATASET_NAME/backbone/` o generarse con GuroBack. Luego se ejecutan
`1_create_ml_dataset.py` y `2_create_partitions.py`.

El formato de backbone observado en `1_create_ml_dataset.py` es:

```text
b -x1   # B0
b x2    # B1
b 0     # extracción completa; debe ser la última línea
```

La extracción requiere GuroBack, Gurobi y una licencia válida. Por ahora el repositorio
incluye una fixture pequeña para probar el parser de este formato, pero no inventa ni
redistribuye los datasets experimentales.
