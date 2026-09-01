# Polarity inversion

## Definition

Let a linear PBO instance have binary variables. For a subset `F` of variables,
define a new assignment as follows:

```text
y_i = 1 - x_i   if i belongs to F
y_i = x_i       otherwise
```

Because the transformation is its own inverse, it establishes a bijection between
assignments of the two instances.

## Constraints

For a constraint

```text
sum(a_i x_i) >= b
```

substituting variables in `F` gives

```text
sum(i not in F, a_i y_i) + sum(i in F, a_i (1 - y_i)) >= b
```

Moving the constant to the right-hand side yields

```text
sum(i not in F, a_i y_i) - sum(i in F, a_i y_i)
    >= b - sum(i in F, a_i)
```

The same derivation applies to equalities.

## Objective

The objective `sum(c_i x_i)` becomes:

```text
sum(i not in F, c_i y_i) - sum(i in F, c_i y_i)
    + sum(i in F, c_i)
```

The restricted OPB format does not write a constant in `min:`. The writer stores
this offset in a `backbone-pbo objective-offset` comment. OPB solvers may ignore it
because it does not change `argmin`; add it to the solver-reported value when
comparing objective values across instances.

## Backbone

For an inverted variable:

- a variable fixed to 0 in every optimal solution becomes fixed to 1;
- a variable fixed to 1 becomes fixed to 0;
- a variable that varies across optimal solutions continues to vary.

Labels for non-inverted variables do not change.

## Automatically verified properties

- feasibility correspondence for every binary assignment;
- exact objective-value correspondence, including the offset;
- involution: applying the same inversion twice recovers the model;
- preservation of the number of variables and constraints;
- the expected B0/B1/NB transformation.

These tests validate the implementation. The usefulness for machine learning must
be evaluated separately through controlled experiments.
