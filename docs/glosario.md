# Glosario de la tesis

Definiciones de los términos técnicos usados en el proyecto. Cada entrada
describe el significado, la aplicación dentro del proyecto y el fundamento que
la respalda.

## Problema y modelo matemático

### Optimización Pseudo-Booleana (PBO)

**Significado** = Problema de optimización que utiliza variables binarias,
una función objetivo lineal y restricciones pseudo-booleanas lineales.

**Aplicación** = Es el tipo de problema que se representa, transforma y
resuelve en este proyecto.

**Fundamento** = Las variables solo pueden tomar los valores `0` o `1`, y las
restricciones se expresan como sumas lineales de esas variables.

### Variable binaria

**Significado** = Variable cuyo dominio está restringido a `{0, 1}`.

**Aplicación** = Las variables `x1`, `x2`, ..., `xN` representan decisiones del
problema PBO y reciben una etiqueta de backbone.

**Fundamento** = La representación binaria permite interpretar la instancia
como un problema combinatorio de decisiones sí/no.

### Instancia PBO

**Significado** = Una configuración concreta de variables, función objetivo,
restricciones y, cuando corresponde, etiquetas del backbone.

**Aplicación** = El código la representa mediante `PBOInstance` y la almacena
en el formato OPB restringido.

**Fundamento** = Una instancia define el problema matemático específico sobre
el cual se evalúan factibilidad, optimalidad y predicciones.

### Función objetivo

**Significado** = Expresión que asigna un valor a cada asignación de variables y
que el solucionador intenta minimizar.

**Aplicación** = En el código se representa mediante `LinearExpression` y se
utiliza para comparar soluciones óptimas.

**Fundamento** = Una solución es mejor que otra cuando obtiene un valor menor
de la función objetivo, respetando las restricciones.

### Restricción pseudo-booleana

**Significado** = Relación lineal que una asignación debe satisfacer, por
ejemplo, `a1 x1 + a2 x2 >= b` o `a1 x1 + a2 x2 = b`.

**Aplicación** = Define las condiciones de factibilidad de cada instancia PBO.

**Fundamento** = Las restricciones limitan las combinaciones de valores
binarios permitidas.

### Solución factible

**Significado** = Asignación binaria que satisface todas las restricciones de
una instancia.

**Aplicación** = Se utiliza para comprobar la correspondencia entre una
instancia original y su versión transformada.

**Fundamento** = La factibilidad depende únicamente del cumplimiento de todas
las restricciones.

### Solución óptima

**Significado** = Solución factible que obtiene el menor valor posible de la
función objetivo.

**Aplicación** = Las soluciones óptimas se utilizan para determinar si una
variable pertenece al backbone.

**Fundamento** = El backbone se define observando el valor de cada variable en
el conjunto completo de soluciones óptimas.

## Backbone y etiquetas

### Backbone

**Significado** = Conjunto de variables que mantienen el mismo valor en todas
las soluciones óptimas de una instancia.

**Aplicación** = Es la estructura que la GNN intenta predecir para orientar un
proceso de búsqueda.

**Fundamento** = Una variable pertenece al backbone si existe un valor fijo
`v ∈ {0,1}` tal que `x_i = v` en todas las soluciones óptimas.

### B0

**Significado** = Etiqueta del backbone para una variable que vale `0` en todas
las soluciones óptimas.

**Aplicación** = Indica una variable que puede clasificarse como fija en cero.

**Fundamento** = Se asigna cuando la variable puede tomar `0` en una solución
óptima, pero no puede tomar `1` en ninguna solución óptima.

### B1

**Significado** = Etiqueta del backbone para una variable que vale `1` en todas
las soluciones óptimas.

**Aplicación** = Indica una variable que puede clasificarse como fija en uno.

**Fundamento** = Se asigna cuando la variable puede tomar `1` en una solución
óptima, pero no puede tomar `0` en ninguna solución óptima.

### NB (No-Backbone)

**Significado** = Etiqueta para una variable que no pertenece al backbone.

**Aplicación** = Indica que la variable cambia de valor entre distintas
soluciones óptimas.

**Fundamento** = Una variable es `NB` cuando existen soluciones óptimas en las
que toma `0` y otras en las que toma `1`.

### Etiqueta de clase

**Significado** = Valor que identifica la categoría asignada a una variable.

**Aplicación** = La implementación utiliza `B0 = 0`, `B1 = 1` y `NB = 2` como
clases de clasificación.

**Fundamento** = La GNN produce una predicción multiclase para cada nodo de
variable.

### Desbalance de clases

**Significado** = Situación en la que algunas clases tienen muchas más
observaciones que otras.

**Aplicación** = El proyecto estudia el desbalance entre `B0`, `B1` y `NB` en
los datos usados para entrenar la GNN.

**Fundamento** = Una clase mayoritaria puede dominar la función de pérdida y
ocultar un desempeño deficiente en las clases minoritarias.

## Aumentación y transformación

### Aumentación de datos

**Significado** = Generación de ejemplos adicionales a partir de datos
existentes, manteniendo la información relevante para la tarea.

**Aplicación** = Se utiliza para ampliar el conjunto de entrenamiento de la
GNN y estudiar el efecto sobre el desbalance del backbone.

**Fundamento** = Un ejemplo aumentado debe conservar una correspondencia
conocida entre sus datos y sus etiquetas.

### Inversión de polaridad

**Significado** = Transformación que sustituye una variable seleccionada por
`x_i = 1 - y_i`.

**Aplicación** = Es la primera técnica de aumentación implementada en
`backbone_pbo.augmentation.polarity`.

**Fundamento** = La sustitución es biyectiva y permite transformar restricciones,
objetivo y etiquetas de forma coherente.

### Inversión global

**Significado** = Inversión de polaridad aplicada a todas las variables de una
instancia.

**Aplicación** = Corresponde al escenario de aumentación usado en la primera
comparación experimental.

**Fundamento** = Todas las variables cambian de representación y sus etiquetas
`B0` y `B1` se intercambian, mientras `NB` permanece `NB`.

### Inversión selectiva

**Significado** = Inversión de polaridad aplicada solo a un subconjunto de
variables.

**Aplicación** = La función `invert_polarity` acepta una colección opcional de
variables seleccionadas.

**Fundamento** = Las variables seleccionadas cambian de polaridad y las demás
conservan sus coeficientes y etiquetas.

### Bijección

**Significado** = Correspondencia uno a uno entre los elementos de dos
conjuntos.

**Aplicación** = Relaciona cada asignación de la instancia original con una
asignación de la instancia transformada.

**Fundamento** = La transformación `x_i = 1 - y_i` puede invertirse de forma
única.

### Involución

**Significado** = Transformación que, al aplicarse dos veces, recupera el objeto
original.

**Aplicación** = Se comprueba aplicando dos veces la misma inversión de
polaridad.

**Fundamento** = `1 - (1 - x_i) = x_i` para cada variable invertida.

### Offset del objetivo

**Significado** = Constante que aparece al sustituir variables en la función
objetivo.

**Aplicación** = El escritor OPB la registra en el comentario
`backbone-pbo objective-offset`.

**Fundamento** = La constante no cambia las asignaciones minimizadoras, pero sí
debe considerarse al comparar valores objetivos exactos.

## Representación como grafo y aprendizaje

### Grafo bipartito

**Significado** = Grafo cuyos nodos se dividen en dos grupos y cuyas aristas
conectan nodos de grupos distintos.

**Aplicación** = Una instancia PBO se representa con nodos de variables y nodos
de restricciones.

**Fundamento** = Una arista conecta una variable con una restricción cuando esa
variable aparece en la restricción.

### Nodo de variable

**Significado** = Nodo del grafo que representa una variable binaria de la
instancia.

**Aplicación** = Es el nodo sobre el cual la GNN produce las etiquetas `B0`,
`B1` o `NB`.

**Fundamento** = Sus características incluyen información del objetivo y de su
participación en las restricciones.

### Nodo de restricción

**Significado** = Nodo del grafo que representa una restricción PBO.

**Aplicación** = Aporta al modelo información sobre el lado derecho, operador y
coeficientes de la restricción.

**Fundamento** = Las restricciones determinan las relaciones entre variables.

### Arista

**Significado** = Conexión entre un nodo de variable y un nodo de restricción.

**Aplicación** = Su atributo principal es el coeficiente con el que la variable
aparece en la restricción.

**Fundamento** = La estructura de aristas conserva la incidencia de variables y
restricciones de la instancia PBO.

### GNN (Graph Neural Network)

**Significado** = Red neuronal que procesa datos estructurados como grafos y
combina información de nodos vecinos.

**Aplicación** = Se utiliza para clasificar los nodos de variables según su
etiqueta de backbone.

**Fundamento** = La información de una variable se actualiza usando sus
características y la información propagada desde las restricciones conectadas.

### GraphConv

**Significado** = Operación de convolución sobre grafos que agrega información
de nodos conectados.

**Aplicación** = La arquitectura `PBOBackboneGNN` utiliza varias capas
`GraphConv`.

**Fundamento** = La operación permite incorporar la estructura de vecindad del
grafo en la representación aprendida.

### Predict-and-Search

**Significado** = Esquema que combina predicciones de aprendizaje automático
con una búsqueda o resolución exacta posterior.

**Aplicación** = La GNN predice variables del backbone que pueden informar el
proceso de búsqueda.

**Fundamento** = La predicción estructural se usa como información auxiliar, no
como sustituto del solucionador exacto.

## Dataset y evaluación

### Dataset

**Significado** = Colección organizada de instancias y etiquetas utilizada para
análisis o aprendizaje.

**Aplicación** = El cargador lee archivos `.opb` y sus archivos
`.opb.backbone` correspondientes.

**Fundamento** = Cada instancia debe asociarse con sus variables y etiquetas
para construir el grafo y evaluar las predicciones.

### Baseline

**Significado** = Configuración de referencia contra la cual se compara una
técnica nueva.

**Aplicación** = El baseline entrena con las instancias originales, sin añadir
las instancias invertidas.

**Fundamento** = Permite atribuir las diferencias observadas al uso de la
aumentación bajo condiciones equivalentes.

### Escenario aumentado de polaridad

**Significado** = Configuración que combina las instancias originales y sus
versiones invertidas durante el entrenamiento.

**Aplicación** = La validación y el test permanecen con instancias originales.

**Fundamento** = La aumentación se aplica solo después de crear las particiones
para evitar contaminación entre entrenamiento y evaluación.

### Train, validation y test

**Significado** = Tres particiones de datos destinadas, respectivamente, al
entrenamiento, selección o seguimiento del modelo y evaluación final.

**Aplicación** = El pipeline utiliza proporciones de 70%, 15% y 15%.

**Fundamento** = Separar las particiones permite evaluar generalización sobre
instancias no usadas para ajustar los parámetros del modelo.

### Data leakage

**Significado** = Incorporación indebida de información de validación o test en
el entrenamiento.

**Aplicación** = El pipeline evita añadir instancias invertidas a validation o
test.

**Fundamento** = La versión aumentada de una instancia no debe permitir que el
modelo vea indirectamente la instancia original durante la evaluación.

### Accuracy

**Significado** = Proporción de predicciones correctas sobre el total de
predicciones.

**Aplicación** = Se reporta como métrica global de clasificación.

**Fundamento** = `accuracy = predicciones correctas / predicciones totales`.

### Precision

**Significado** = Proporción de predicciones positivas correctas entre todas las
predicciones positivas de una clase.

**Aplicación** = Permite medir cuántas variables predichas como una clase
pertenecen realmente a esa clase.

**Fundamento** = `precision = verdaderos positivos / predicciones positivas`.

### Recall

**Significado** = Proporción de elementos de una clase que el modelo identifica
correctamente.

**Aplicación** = Permite medir cuántas variables reales de una clase fueron
recuperadas.

**Fundamento** = `recall = verdaderos positivos / elementos reales de la clase`.

### F1-score

**Significado** = Media armónica entre precision y recall.

**Aplicación** = Se utiliza por clase y de forma macro para considerar el
desempeño en clases desbalanceadas.

**Fundamento** = `F1 = 2 · precision · recall / (precision + recall)`.

### Macro-F1

**Significado** = Promedio del F1-score calculado por separado para cada clase,
con el mismo peso para todas.

**Aplicación** = Resume el desempeño conjunto de `B0`, `B1` y `NB` sin que la
clase mayoritaria domine directamente el promedio.

**Fundamento** = Se calcula como la media aritmética de los F1 de las clases.

### Balanced accuracy

**Significado** = Promedio del recall de cada clase.

**Aplicación** = Complementa la accuracy cuando las clases tienen tamaños
distintos.

**Fundamento** = Da el mismo peso al desempeño de cada clase.

### Solver exacto

**Significado** = Método que busca una solución óptima respetando formalmente
las restricciones del problema.

**Aplicación** = El proyecto utiliza `scipy.optimize.milp` para resolver
instancias lineales y extraer etiquetas de backbone.

**Fundamento** = La solución y las etiquetas sirven como referencia para
comprobar la transformación y entrenar el predictor.

### MILP

**Significado** = Optimización lineal entera mixta; modelo que combina función
objetivo lineal, restricciones lineales y variables con restricciones de
integralidad.

**Aplicación** = Se utiliza como mecanismo de resolución exacta para las
variables binarias de las instancias PBO.

**Fundamento** = Las variables binarias se modelan como variables enteras con
cotas entre `0` y `1`.

### OPB restringido

**Significado** = Subconjunto del formato OPB utilizado por este proyecto para
expresar objetivos lineales, restricciones `>=`, `<=` o `=`, y variables
`x1`...`xN`.

**Aplicación** = Es el formato que leen y escriben `read_opb` y `dumps_opb`.

**Fundamento** = El parser valida expresiones lineales y la secuencia contigua
de variables.

### BackPaS

**Significado** = Proyecto de referencia que utiliza información del backbone
en un flujo de predicción y búsqueda para PBO.

**Aplicación** = Su formato de archivos `.backbone` sirve como referencia para
el lector de etiquetas de este repositorio.

**Fundamento** = El formato usa `b -xN` para `B0`, `b xN` para `B1` y `b 0` como
marca de extracción completa.

### GuroBack

**Significado** = Herramienta utilizada para extraer información del backbone
con un solucionador y una formulación PBO compatibles.

**Aplicación** = Se toma como referencia externa para el formato de etiquetas;
el núcleo actual también contiene un extractor basado en MILP.

**Fundamento** = La extracción del backbone requiere verificar qué valores puede
tomar cada variable manteniendo el valor óptimo.

### MIS (Maximum Independent Set)

**Significado** = Problema de conjunto independiente máximo: dado un grafo, busca
el conjunto de vértices de mayor cardinalidad en el que ningún par esté conectado
por una arista.

**Aplicación** = Es uno de los tres benchmarks PBO utilizados por BackPaS.

**Fundamento** = Se representa con variables binarias de selección y restricciones
que impiden seleccionar simultáneamente los dos extremos de cada arista.

### MVC (Minimum Vertex Cover)

**Significado** = Problema de cobertura mínima de vértices: dado un grafo, busca el
conjunto de vértices de menor costo que cubra todas sus aristas.

**Aplicación** = Es uno de los tres benchmarks PBO utilizados por BackPaS.

**Fundamento** = Se representa con variables binarias de selección y una restricción
por arista que exige seleccionar al menos uno de sus extremos.

### CA (Combinatorial Auctions)

**Significado** = Problema de subastas combinatorias en el que cada oferta asigna un
valor a un conjunto de artículos y se seleccionan ofertas compatibles.

**Aplicación** = Es uno de los tres benchmarks PBO utilizados por BackPaS.

**Fundamento** = La formulación binaria selecciona ofertas para maximizar el ingreso
sin asignar un mismo artículo a más de una oferta aceptada.

### IP (Item Placement)

**Significado** = Benchmark de asignación o ubicación de elementos utilizado por
ConPaS con variables que no son exclusivamente binarias.

**Aplicación** = Sirve para explicar la diferencia entre los cuatro dominios de
ConPaS y los tres dominios PBO evaluados por BackPaS; no pertenece al alcance actual
del pipeline binario.

**Fundamento** = BackPaS lo excluye explícitamente porque contiene variables no
binarias, incompatibles con la formulación PBO binaria usada en este proyecto.

### Manifiesto de ejecución

**Significado** = Registro estructurado que identifica todos los elementos necesarios
para interpretar y repetir una ejecución experimental.

**Aplicación** = Cada entrenamiento debe registrar benchmark, particiones, técnica de
aumentación, parámetros, semilla, commits, ambiente y archivos de salida.

**Fundamento** = Dos resultados solo pueden compararse de manera controlada cuando
sus diferencias de configuración están declaradas explícitamente.

### Reproducción del baseline

**Significado** = Ejecución independiente de la configuración de referencia sin
incorporar la técnica nueva que se desea evaluar.

**Aplicación** = Se realiza para MVC, MIS y CA antes de añadir cualquier instancia
aumentada al entrenamiento.

**Fundamento** = Permite comprobar que las diferencias posteriores provienen del
tratamiento experimental y no de una implementación o configuración distinta.

### Prueba de humo de entrenamiento

**Significado** = Entrenamiento corto cuyo objetivo es comprobar el funcionamiento
completo del pipeline, no medir desempeño científico.

**Aplicación** = Verifica carga, batching, propagación hacia adelante y atrás,
checkpoints y generación de métricas antes de una corrida completa.

**Fundamento** = Detecta errores operacionales con un costo reducido y sus métricas no
se presentan como resultados finales.

### Auditoría de dataset

**Significado** = Validación automática del contrato estructural y semántico que debe
cumplir un conjunto de instancias antes de utilizarlo experimentalmente.

**Aplicación** = Comprueba formatos, particiones, variables binarias, correspondencia
con backbones, etiquetas, duplicados, checksums y distribución B0/B1/NB.

**Fundamento** = Impide que archivos omitidos o particiones incorrectas alteren un
entrenamiento sin quedar registrados.

### D-MIPLIB

**Significado** = Distributional MIPLIB, colección pública de distribuciones de
problemas MILP con particiones predefinidas de entrenamiento, validación y prueba.

**Aplicación** = La configuración `MVC-easy` proporciona las 800 instancias de
entrenamiento y 100 de validación utilizadas para reconstruir el baseline MVC.

**Fundamento** = Sus instancias MVC tienen 1200 variables binarias y 5975
restricciones, coincidiendo con la distribución descrita para el entrenamiento de
BackPaS.

### WSL 2

**Significado** = Windows Subsystem for Linux, entorno que permite ejecutar una
distribución Linux dentro de Windows.

**Aplicación** = Ejecuta GuroBack y el ambiente histórico de BackPaS sin modificar sus
dependencias para adaptarlas a Python o Windows actuales.

**Fundamento** = Los proyectos originales y sus bibliotecas fueron preparados para
Linux; conservar ese ambiente reduce diferencias de reproducción.

### Licencia académica de Gurobi

**Significado** = Autorización gratuita para uso académico del solucionador Gurobi,
asociada a una cuenta y a un equipo autorizado.

**Aplicación** = GuroBack necesita una licencia activa dentro de WSL para resolver las
instancias óptimamente y determinar sus variables backbone.

**Fundamento** = El software y las bibliotecas pueden instalarse sin la licencia, pero
Gurobi rechaza cualquier optimización hasta encontrar una autorización válida.
