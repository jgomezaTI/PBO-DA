import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from materialize_variable_permutation import permute_text, variables_in


def test_permute_text_renames_variables_only():
    source = "Maximize\n obj: +2 x1 +3 x2\nSubject To\n c: x1 <= 1\n"
    assert permute_text(source, {"x1": "x2", "x2": "x1"}) == (
        "Maximize\n obj: +2 x2 +3 x1\nSubject To\n c: x2 <= 1\n"
    )


def test_variables_in_supports_x_and_v_names():
    assert variables_in("+1 x2 +1 v1 +1 x2") == ["v1", "x2"]
