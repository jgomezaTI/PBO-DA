import pytest

from backbone_pbo.io.opb import OPBFormatError, dumps_opb, loads_opb

SAMPLE = """\
* #variable= 3 #constraint= 2 #equal= 1 intsize= 4
* example
min: +2 x1 -3 x2 +1 x3 ;
+1 x1 +2 x2 >= +1 ;
-1 x1 +1 x3 = 0 ;
"""


def test_parse_linear_opb():
    instance = loads_opb(SAMPLE)

    assert instance.variables == {"x1", "x2", "x3"}
    assert instance.objective.coefficients == {"x1": 2, "x2": -3, "x3": 1}
    assert instance.constraints[0].rhs == 1
    assert instance.constraints[1].operator == "="


def test_round_trip_preserves_semantics_and_offset():
    instance = loads_opb(SAMPLE)
    transformed_text = dumps_opb(instance)
    restored = loads_opb(transformed_text)

    assert restored.objective == instance.objective
    assert restored.constraints == instance.constraints


def test_writer_normalizes_less_equal():
    instance = loads_opb("* #variable= 1 #constraint= 1\n+1 x1 <= 0;\n")

    output = dumps_opb(instance)

    assert "-1 x1 >= +0 ;" in output


def test_rejects_nonlinear_terms():
    with pytest.raises(OPBFormatError, match="Unsupported fragment"):
        loads_opb("* #variable= 2 #constraint= 1\n+1 x1 x2 >= 1;\n")


def test_validates_header_counts():
    with pytest.raises(OPBFormatError, match="header declares 2 variables"):
        loads_opb("* #variable= 2 #constraint= 1\n+1 x1 >= 1;\n")
