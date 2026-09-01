# backbonePBO

Proyecto de TT1 sobre técnicas de aumentación de datos para mitigar el desbalance
del backbone en GNN aplicadas a Optimización Pseudo-Booleana (PBO).

El repositorio estudia las aumentaciones como aporte principal. BackPaS, GuroBack y
otros trabajos previos pueden usarse como infraestructura o baseline, pero no son el
objetivo del proyecto.

## Estado actual

La primera técnica implementada es la **inversión de polaridad** en instancias OPB
lineales. Para un conjunto de variables se aplica la sustitución

```text
x_i = 1 - y_i
```

La transformación induce una biyección entre asignaciones originales y aumentadas,
preserva factibilidad y optimalidad, y transforma las etiquetas del backbone así:

```text
B0 <-> B1
NB  -> NB
```

El soporte actual se limita deliberadamente al formato OPB lineal restringido
(`min:`, restricciones `>=` o `=`, variables `x1` a `xN`). Los productos no lineales,
WBO y la integración con GuroBack todavía no forman parte del núcleo.

## Instalación para desarrollo

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
```

En Linux/macOS, el ejecutable del entorno es `.venv/bin/python`.

## Uso

Invertir todas las variables:

```bash
backbone-pbo invert entrada.opb salida.opb
```

Invertir solo algunas variables:

```bash
backbone-pbo invert entrada.opb salida.opb --variables x1 x3 x8
```

Inspeccionar una instancia:

```bash
backbone-pbo inspect entrada.opb
```

Ejecutar las verificaciones locales:

```bash
ruff check .
ruff format --check .
pytest
```

Las fixtures que necesitan escribir archivos usan `.tmp/pytest-local/` dentro del
repositorio. Esa carpeta está ignorada por Git; así las pruebas no dependen del
directorio temporal global de Windows.

## Diseño experimental inicial

1. Obtener las instancias, backbones y particiones originales.
2. Caracterizar B0, B1 y NB por instancia, familia y partición.
3. Mantener validation y test sin aumentación.
4. Comparar el mismo entrenamiento con `train original` frente a
   `train original + train invertido`.
5. Usar las mismas semillas e hiperparámetros y reportar métricas macro y por clase.

Los detalles matemáticos y las propiedades verificadas están en
[`docs/polarity_inversion.md`](docs/polarity_inversion.md).

Para conectar una plantilla existente de Overleaf de forma segura, consulta
[`docs/overleaf.md`](docs/overleaf.md).

## Estructura

```text
src/backbone_pbo/       modelo, parser OPB, aumentación y CLI
tests/unit/             pruebas deterministas
tests/property/         propiedades matemáticas con Hypothesis
tests/integration/      futuras pruebas que requieren GuroBack/Gurobi
configs/                configuraciones experimentales iniciales
data/README.md          contrato de datos y trazabilidad
docs/                   decisiones matemáticas y metodológicas
```
