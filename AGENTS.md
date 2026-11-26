# Instrucciones permanentes de la tesis

## Regla prioritaria

Antes de realizar cualquier acción en este repositorio —inspección, edición,
generación de datos, uso de Colab/Drive, entrenamiento o reporte— se debe leer
este archivo y aplicar sus reglas. Estas instrucciones reflejan el criterio
metodológico acordado con el autor de la tesis.

## Flujo operativo obligatorio PC → GitHub → Colab → Drive

- Los cambios se desarrollan y validan primero en el PC.
- Antes de usar Colab, el agente debe revisar el diff, registrar los cambios
  pertinentes, crear un commit y hacer `push` al repositorio de GitHub que
  consumirá Colab.
- Colab debe abrirse desde ese repositorio y consumir explícitamente el commit
  publicado; no debe ejecutarse una versión local no publicada cuando el
  objetivo sea reproducibilidad.
- Antes de ejecutar, confirmar la cuenta, repositorio, rama y commit que usará
  la sesión de Colab.
- Drive es la fuente de verdad de datos, estados, logs, checkpoints y
  resultados. El agente debe auditar allí la corrida y no tratar la copia local
  como evidencia suficiente.
- Si el código local tiene cambios sin publicar o el commit consumido por
  Colab no está identificado, detener el lanzamiento experimental y explicarlo.

## Integridad experimental

- No optimizar el proceso reduciendo el dataset, usando subconjuntos, creando
  versiones pequeñas, acortando el entrenamiento o eliminando validaciones para
  ahorrar tiempo o GPU.
- Si se trabaja con un dataset definido, entrenar con el dataset completo y
  conservar sus particiones oficiales. La T4 puede justificar ajustar
  `batch_size`, acumulación o `threads`, pero nunca cambiar el conjunto
  experimental.
- No crear pilotos o smoke tests como resultado de tesis. Solo pueden usarse si
  el usuario los solicita explícitamente y deben quedar marcados como pruebas
  de infraestructura no reportables.
- La aumentación se aplica únicamente a la partición autorizada. En las
  aumentaciones actuales se transforma `train`; `validation` y `test` se copian
  sin modificar y no se usan como fuente de aumentación.
- Cada entrenamiento de tesis debe registrar validación cuando exista una
  partición `validation`, guardar el mejor checkpoint según validación, el
  último checkpoint, logs y el estado de la corrida.
- No considerar terminado ni reportable un trabajo sin filas de validación,
  conteos de particiones verificables, logs y checkpoints.
- No reutilizar checkpoints de una corrida incompatible: piloto, dataset o
  particiones distintas, configuración distinta o etiquetas no comparables.
- La validación matemática/estructural de la aumentación es distinta de la
  validación del entrenamiento; ambas deben documentarse.

## Síntesis del trabajo inspeccionado

- El repositorio separa datasets originales, aumentaciones, entrenamiento y
  resultados; preserva la procedencia y exige reproducibilidad.
- MIS constraint-perturbation modifica exactamente una restricción de arista por
  instancia de `train`, conserva la función objetivo y valida las etiquetas.
- El objetivo de los tres benchmarks actuales es `900` instancias: `800` de
  `train` y `100` de `validation`. Drive es la fuente de datos para entrenar;
  cualquier copia local con menos archivos debe tratarse como incompleta y no
  como una nueva partición experimental. La ausencia de `test` debe declararse,
  no ocultarse ni compensarse con un subconjunto artificial.
- Los trabajos finales deben registrar rutas estables en Drive, manifiesto,
  parámetros de transformación, semilla, configuración, hardware, logs,
  checkpoints y auditoría posterior.
- Auditoría de particiones de aumentaciones revisada el 18-11-2026: MVC
  polarity conserva las 800 instancias de entrenamiento fuente y las 100 de
  validación; CA constraint-perturbation también conserva 800/100. Son las dos
  ejecuciones aumentadas actualmente reportables. MIS polarity conserva solo
  761/800 instancias fuente y tiene 97 de validación; CA polarity conserva
  788/800 instancias fuente, aunque mantiene 100 de validación. Ambas corridas
  completaron entrenamiento, pero quedan excluidas hasta regenerar sus datos y
  entrenarlas con las particiones íntegras. No volver a presentarlas como
  resultados reportables.

## Preflight obligatorio

Antes de entrenar o modificar artefactos:

1. Leer este `AGENTS.md`.
2. Inspeccionar el dataset y contar `train`, `validation` y `test`.
3. Confirmar que no se usa un marcador de piloto ni una partición vacía creada
   artificialmente.
4. Verificar que la aumentación conserva intactas las particiones no transformadas.
5. Confirmar que el runner lee `validation` y guarda checkpoints.
6. Revisar manifiesto, logs y estado al terminar.

Si alguna comprobación falla, detener la corrida experimental, explicar la causa
y corregir el protocolo antes de entrenar. Nunca ocultar el problema reduciendo
el alcance del dataset.

## Alcance y seguridad

Preservar los archivos del usuario y evitar operaciones destructivas. El usuario
puede eliminar artefactos inválidos; no recrearlos sin solicitud explícita.

## Criterio editorial de la tesis

- En el manuscrito informar solo las características de cómputo pertinentes para
  interpretar o reproducir cada modelo: CPU/GPU, acelerador específico cuando
  esté verificado y configuración relevante. No mencionar el proveedor o servicio
  de ejecución (por ejemplo, Colab) salvo que sea metodológicamente necesario.
- No incluir dónde se guardaron los archivos, si se almacenaron en Drive ni
  dependencias de un equipo personal. La persistencia y ubicación de artefactos
  pertenecen a la gestión del proyecto, no al argumento académico.
