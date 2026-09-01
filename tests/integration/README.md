# Integración con GuroBack

Aquí se incorporarán pruebas marcadas con `@pytest.mark.gurobi` cuando estén
disponibles GuroBack, una licencia válida de Gurobi y el formato exacto de sus archivos
de backbone.

La comprobación prevista es:

```text
transform(backbone(instancia)) == backbone(invert_polarity(instancia))
```

