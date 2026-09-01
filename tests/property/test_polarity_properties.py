from itertools import product

from hypothesis import given
from hypothesis import strategies as st

from backbone_pbo.augmentation.polarity import invert_assignment, invert_polarity
from backbone_pbo.model import Constraint, LinearExpression, PBOInstance


@st.composite
def linear_instances(draw):
    count = draw(st.integers(min_value=1, max_value=6))
    variables = [f"x{index}" for index in range(1, count + 1)]
    coefficient = st.integers(min_value=-8, max_value=8)

    objective_values = draw(st.lists(coefficient, min_size=count, max_size=count))
    objective = LinearExpression(dict(zip(variables, objective_values, strict=True)))

    constraint_count = draw(st.integers(min_value=1, max_value=5))
    constraints = []
    for _ in range(constraint_count):
        values = draw(st.lists(coefficient, min_size=count, max_size=count))
        if not any(values):
            values[0] = 1
        constraints.append(
            Constraint(
                LinearExpression(dict(zip(variables, values, strict=True))),
                draw(st.sampled_from([">=", "=", "<="])),
                draw(st.integers(min_value=-15, max_value=15)),
            )
        )
    return PBOInstance(tuple(constraints), objective)


@given(linear_instances(), st.data())
def test_inversion_is_a_semantic_involution(instance, data):
    selected = frozenset(
        data.draw(
            st.sets(st.sampled_from(sorted(instance.variables)), max_size=len(instance.variables))
        )
    )
    transformed = invert_polarity(instance, selected)
    restored = invert_polarity(transformed, selected)

    assert restored.objective == instance.objective
    assert restored.constraints == instance.constraints
    assert len(transformed.variables) == len(instance.variables)
    assert len(transformed.constraints) == len(instance.constraints)

    for values in product((0, 1), repeat=len(instance.variables)):
        names = sorted(instance.variables)
        assignment = dict(zip(names, values, strict=True))
        mapped = invert_assignment(assignment, selected)
        assert instance.is_feasible(assignment) == transformed.is_feasible(mapped)
        assert instance.objective_value(assignment) == transformed.objective_value(mapped)
