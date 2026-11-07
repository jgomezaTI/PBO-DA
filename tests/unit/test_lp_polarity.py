from backbone_pbo.io.lp import invert_lp


def test_invert_lp_transforms_objective_constraints_and_keeps_bounds():
    source = """Minimize
 Obj: +2 v0 +1 v1
Subject To
 c1: +1 v0 +1 v1 >= +1
Bounds
 0 <= v0 <= 1
 0 <= v1 <= 1
End
"""
    transformed = invert_lp(source, {"v0"})
    assert " -2 v0" in transformed
    assert " +1 v1" in transformed
    assert "c1:  -1 v0" in transformed and ">= 0" in transformed
    assert "0 <= v0 <= 1" in transformed


def test_invert_lp_auto_detects_x_variables():
    source = """Maximize
 Obj: +2 x1 +1 x2
Subject To
 c1: +1 x1 +1 x2 >= +1
"""

    transformed = invert_lp(source)

    assert " -2 x1" in transformed
