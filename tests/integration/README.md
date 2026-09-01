# GuroBack integration

Tests marked with `@pytest.mark.gurobi` will be added when GuroBack, a valid Gurobi
license, and the exact backbone file format are available.

The planned check is:

```text
transform(backbone(instance)) == backbone(invert_polarity(instance))
```
