# PRD: Desarrollo experimental y documental de la tesis PBO

**Estado**: Aprobado para planificación  
**Responsable**: Javier Gómez  
**Profesor guía**: Roberto Asín  
**Fecha**: 2026-10-20  
**Especificación activa local**: `specs/001-thesis-workplan/spec.md` (excluida de Git)

## 1. Resumen ejecutivo

El repositorio estudia técnicas de aumentación de datos para mitigar el desbalance de
las etiquetas de backbone B0, B1 y NB en modelos GNN aplicados a Optimización
Pseudo-Booleana (PBO). El primer caso de estudio utiliza inversión de polaridad y el
pipeline de BackPaS sobre Maximum Vertex Cover (MVC).

La siguiente etapa debe transformar el trabajo ya realizado en una investigación
trazable y reproducible. Para ello se completará el estado del arte, se formalizará la
metodología y la procedencia de los datos, se ejecutará la comparación controlada entre
baseline y polaridad, se automatizará el reporte de resultados y se mantendrá la tesis
sincronizada con la evidencia disponible.

## 2. Problema

El repositorio ya contiene una transformación de polaridad, pruebas matemáticas, un
dataset MVC auditado y un baseline en ejecución. Sin embargo, todavía existen brechas:

- falta un estado del arte que compare sistemáticamente trabajos relacionados;
- la metodología experimental está distribuida entre la tesis, documentación y scripts;
- la procedencia y licencia de cada dataset deben quedar trazadas;
- aún no existe un experimento completo y comparable con polaridad;
- las métricas se trasladan manualmente desde los logs a la tesis;
- las conclusiones preliminares no reflejan todo el avance actual; y
- MVC no contiene ejemplos NB ni una partición independiente de test.

## 3. Objetivo

Construir un proceso de investigación reproducible que permita responder si la
inversión de polaridad modifica favorablemente el aprendizaje del backbone, manteniendo
control experimental, trazabilidad de datos y correspondencia verificable entre la
evidencia del repositorio y lo escrito en la tesis.

## 4. Usuarios y partes interesadas

- **Tesista**: planifica, ejecuta y documenta los experimentos.
- **Profesor guía**: revisa el fundamento, el diseño experimental y las conclusiones.
- **Evaluadores**: deben poder seguir la evidencia desde la tesis hasta los artefactos.
- **Investigadores futuros**: deben poder reproducir o extender el trabajo a MIS y CA.

## 5. Alcance

### 5.1 Incluido

1. Estado del arte sobre aprendizaje para optimización, GNN, backbone,
   Predict-and-Search, desbalance y aumentación de grafos o instancias matemáticas.
2. Comparación explícita con BackPaS y otros antecedentes relevantes.
3. Metodología experimental completa desde el origen del dato hasta el reporte.
4. Registro de fuente, licencia, partición, transformación y versión de los datasets.
5. Generación y auditoría del entrenamiento MVC aumentado por polaridad.
6. Entrenamiento comparable baseline versus polaridad.
7. Extracción reproducible de métricas y generación de tablas.
8. Actualización de resultados, limitaciones y conclusiones de la tesis.
9. Extensión posterior del protocolo a MIS y CA cuando sus datos estén disponibles.

### 5.2 Fuera de alcance inmediato

- afirmar mejoras del solucionador sin integrar y medir la etapa de búsqueda;
- mezclar polaridad con cambios de arquitectura, focal loss o ponderación de clases;
- evaluar NB usando MVC, porque la clase no aparece en el dataset reconstruido;
- tratar validación como test independiente;
- inventar particiones, métricas, procedencia o licencias no verificadas; y
- incorporar problemas no binarios sin redefinir formalmente el alcance PBO.

## 6. Estado actual verificable

- Dataset MVC: 800 instancias de entrenamiento y 100 de validación.
- Test independiente: no disponible.
- Etiquetas: 418.707 B0, 661.293 B1 y 0 NB en el total reconstruido.
- Baseline: BackPaS con grafo de literales, GTR, 8 capas, semilla 0 y 200 épocas.
- Mejor resultado preliminar observado al redactar: época 117, pérdida de validación
  0,1251, accuracy micro 94,44 % y F1 macro 94,11 %.
- Técnica implementada: inversión de polaridad con correspondencia B0 ↔ B1 y NB → NB.

Los valores de entrenamiento son preliminares hasta finalizar y validar la ejecución.

## 7. Entregables

### E1. Estado del arte

- Capítulo separado del marco conceptual.
- Matriz comparativa de trabajos relacionados.
- Bibliografía ampliada y revisada, priorizando fuentes primarias.
- Identificación explícita de la brecha que aborda esta tesis.

### E2. Metodología y procedencia

- Flujo experimental completo y versionado.
- Ficha de procedencia por dataset: autor, fuente, licencia, familia, partición y hash o
  versión cuando corresponda.
- Reglas contra leakage y criterios de selección del checkpoint.
- Tratamiento explícito de datasets sin NB o sin test.

### E3. Experimento de polaridad

- Aumentación aplicada exclusivamente al conjunto de entrenamiento.
- Validación matemática y estructural de las instancias transformadas.
- Manifiesto del dataset aumentado.
- Entrenamiento con la misma arquitectura, semilla y presupuesto del baseline.

### E4. Reporte reproducible

- Resumen del mejor checkpoint por ejecución.
- Precision, recall y F1 por clase; métricas macro y micro; matriz de confusión.
- Tabla comparable baseline versus polaridad.
- Registro de configuración, fecha, versión de datos y estado de la ejecución.
- Media y desviación estándar si se completan múltiples semillas.

### E5. Tesis sincronizada

- Resultados marcados como preliminares o definitivos según la evidencia.
- Conclusiones actualizadas después de cada hito experimental.
- Limitaciones sobre NB, test, semillas y utilidad para búsqueda.
- Referencias cruzadas hacia los artefactos reproducibles del repositorio.

## 8. Requisitos

- **REQ-01**: cada afirmación experimental debe vincularse a una ejecución y
  configuración identificables.
- **REQ-02**: la partición debe realizarse antes de cualquier aumentación.
- **REQ-03**: validación y test, si existe, deben permanecer sin aumentación.
- **REQ-04**: baseline y polaridad deben diferir únicamente en la aumentación durante
  la comparación principal.
- **REQ-05**: la selección del checkpoint debe usar el mismo criterio en ambos casos.
- **REQ-06**: las métricas de una clase ausente deben informarse como no aplicables.
- **REQ-07**: los resultados de validación no deben presentarse como generalización a
  test.
- **REQ-08**: toda fuente externa debe registrar procedencia y condiciones de uso.
- **REQ-09**: tablas y cifras deben poder regenerarse desde artefactos de ejecución.
- **REQ-10**: cambios de alcance o metodología deben quedar reflejados en Spec Kit.

## 9. Indicadores de éxito

- Existe una matriz con al menos 15 fuentes primarias pertinentes y la brecha de
  investigación queda sustentada explícitamente.
- El 100 % de los datasets usados tiene fuente y particiones documentadas.
- El 100 % de las instancias aumentadas supera las validaciones estructurales definidas.
- Baseline y polaridad tienen configuraciones comparables y artefactos completos.
- Todas las cifras incluidas en resultados pueden regenerarse desde los logs.
- No quedan citas ni referencias indefinidas al compilar la tesis.
- Un tercero puede seguir el procedimiento documentado sin decisiones implícitas.

## 10. Fases y prioridades

| Fase | Prioridad | Resultado |
|---|---:|---|
| Estado del arte | P1 | Brecha científica y matriz comparativa sustentadas |
| Metodología y procedencia | P1 | Protocolo auditable y datasets trazados |
| Dataset de polaridad | P1 | Datos aumentados validados y manifestados |
| Entrenamiento comparativo | P1 | Baseline y polaridad comparables |
| Reporte automatizado | P2 | Tablas y resúmenes regenerables |
| Extensión a MIS y CA | P3 | Evidencia en otras familias |
| Integración con búsqueda | P3 | Medición del impacto en el solucionador |

## 11. Riesgos y mitigaciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| No existe test independiente en MVC | Sobreestimación del desempeño | Declarar la limitación y obtener una partición externa antes de afirmar generalización |
| MVC no contiene NB | Evaluación incompleta de tres clases | Restringir conclusiones a B0/B1 y buscar evidencia en MIS o CA |
| Entrenamientos costosos | Pocas semillas | Priorizar una comparación controlada y ampliar semillas según recursos |
| Datos externos incompletos | Baja reproducibilidad | Mantener manifiestos, auditorías y procedencia por artefacto |
| Cambios simultáneos | No se puede atribuir el efecto | Mantener arquitectura e hiperparámetros constantes |
| Sesgo de selección bibliográfica | Estado del arte débil | Definir criterios de búsqueda, inclusión y exclusión |

## 12. Definición de terminado

La iniciativa se considera terminada cuando los entregables E1–E5 están completos, la
comparación baseline/polaridad puede reproducirse, las limitaciones están declaradas y
la tesis compila con resultados consistentes con los artefactos. MIS, CA y la integración
con búsqueda son extensiones posteriores y no bloquean el cierre del caso MVC, salvo que
sean necesarios para evaluar NB o generalización.

## 13. Trazabilidad Spec Kit

El PRD define el propósito y alcance global. La carpeta local
`specs/001-thesis-workplan/` contiene la especificación verificable de esta iniciativa,
pero no se incorpora a Git por decisión del usuario. Las siguientes etapas se ejecutan
mediante `$speckit-plan`, `$speckit-tasks` y `$speckit-implement`, manteniendo los cambios
de alcance dentro de la especificación antes de implementarlos.

### Historias verificables

| Historia | Entregable principal | Gates constitucionales |
|---|---|---|
| US1 — Fundamentar la contribución | Protocolo, matriz y estado del arte | Alcance, evidencia, tesis |
| US2 — Trazar datos y metodología | Procedencia, particiones y auditoría MVC | Datos, trazabilidad |
| US3 — Comparar baseline y polaridad | Tratamiento auditado y ejecución controlada | Corrección, smoke, comparación |
| US4 — Regenerar resultados | Resúmenes y tabla generados desde artefactos | Reporte, evidencia |
| US5 — Mantener la tesis coherente | Resultados, limitaciones y conclusiones sincronizadas | Tesis, alcance |

La relación ejecutable entre estas historias, FR-001--FR-014 y T001--T060 se mantiene
en los artefactos locales de `specs/001-thesis-workplan/`. Ningún resultado puede omitir
los gates de datos, corrección, smoke, comparación, reporte y compilación que le sean
aplicables.
