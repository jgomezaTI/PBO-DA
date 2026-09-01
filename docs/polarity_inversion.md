# Inversión de polaridad

## Definición

Sea una instancia PBO lineal con variables binarias. Para un subconjunto `F` de
variables se define una nueva asignación mediante:

```text
y_i = 1 - x_i   si i pertenece a F
y_i = x_i       en otro caso
```

Como la transformación es su propia inversa, establece una biyección entre las
asignaciones de ambas instancias.

## Restricciones

Para una restricción

```text
sum(a_i x_i) >= b
```

la sustitución de las variables de `F` produce

```text
sum(i no en F, a_i y_i) + sum(i en F, a_i (1 - y_i)) >= b
```

y, al mover la constante al lado derecho:

```text
sum(i no en F, a_i y_i) - sum(i en F, a_i y_i)
    >= b - sum(i en F, a_i)
```

La misma derivación vale para igualdades.

## Objetivo

El objetivo `sum(c_i x_i)` se transforma en:

```text
sum(i no en F, c_i y_i) - sum(i en F, c_i y_i)
    + sum(i en F, c_i)
```

El formato OPB restringido no escribe una constante en `min:`. El escritor guarda
ese desplazamiento en un comentario `backbone-pbo objective-offset`. Los solvers OPB
pueden ignorarlo porque no altera `argmin`; debe sumarse al valor informado por el
solver cuando se comparen valores objetivos entre instancias.

## Backbone

Para una variable invertida:

- si estaba fija en 0 en todas las soluciones óptimas, queda fija en 1;
- si estaba fija en 1, queda fija en 0;
- si variaba entre soluciones óptimas, continúa variando.

Las etiquetas de variables no invertidas no cambian.

## Propiedades verificadas automáticamente

- correspondencia de factibilidad para toda asignación binaria;
- correspondencia exacta del valor objetivo, incluyendo el offset;
- involución: aplicar dos veces la misma inversión recupera el modelo;
- preservación del número de variables y restricciones;
- transformación B0/B1/NB esperada.

Estas pruebas validan la implementación. La utilidad para aprendizaje automático se
debe evaluar por separado mediante experimentos controlados.

