from itertools import product

import pytest

from backbone_pbo.augmentation.polarity import invert_assignment, invert_polarity
from backbone_pbo.io.opb import dumps_opb, loads_opb

SAMPLE = """\
* #variable= 3 #constraint= 2
min: +2 x1 -3 x2 +1 x3 ;
+1 x1 +2 x2 >= +1 ;
-1 x1 +1 x3 = 0 ;
"""


def test_global_inversion_has_expected_coefficients_and_rhs():
    transformed = invert_polarity(loads_opb(SAMPLE))

    assert transformed.objective.coefficients == {"x1": -2, "x2": 3, "x3": -1}
    assert transformed.objective.constant == 0
    assert transformed.constraints[0].expression.coefficients == {"x1": -1, "x2": -2}
    assert transformed.constraints[0].rhs == -2
    assert transformed.constraints[1].expression.coefficients == {"x1": 1, "x3": -1}
    assert transformed.constraints[1].rhs == 0


def test_selective_inversion_preserves_feasibility_and_objective_value():
    original = loads_opb(SAMPLE)
    transformed = invert_polarity(original, {"x1", "x3"})

    for values in product((0, 1), repeat=3):
        assignment = dict(zip(("x1", "x2", "x3"), values, strict=True))
        mapped = invert_assignment(assignment, {"x1", "x3"})
        assert original.is_feasible(assignment) == transformed.is_feasible(mapped)
        assert original.objective_value(assignment) == transformed.objective_value(mapped)


def test_double_inversion_restores_mathematical_instance():
    original = loads_opb(SAMPLE)

    restored = invert_polarity(invert_polarity(original, {"x1", "x2"}), {"x1", "x2"})

    assert restored.objective == original.objective
    assert restored.constraints == original.constraints


def test_offset_survives_serialization():
    original = loads_opb(SAMPLE)
    transformed = invert_polarity(original, {"x1"})

    restored = loads_opb(dumps_opb(transformed))

    assert restored.objective.constant == 2
    assert restored.objective == transformed.objective


def test_rejects_unknown_variable():
    with pytest.raises(ValueError, match="x99"):
        invert_polarity(loads_opb(SAMPLE), {"x99"})
