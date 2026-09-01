import json

from backbone_pbo.cli import main
from backbone_pbo.io.opb import read_opb


def test_inspect_prints_machine_readable_summary(capsys):
    exit_code = main(["inspect", "tests/fixtures/opb/tiny.opb"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["problem_type"] == "PBO"
    assert payload["variables"] == 3
    assert payload["constraints"] == 2


def test_invert_writes_parseable_augmented_instance(repo_tmp_path):
    output = repo_tmp_path / "tiny-inverted.opb"

    exit_code = main(
        [
            "invert",
            "tests/fixtures/opb/tiny.opb",
            str(output),
            "--variables",
            "x1",
            "x3",
        ]
    )

    transformed = read_opb(output)
    assert exit_code == 0
    assert transformed.objective.coefficients == {"x1": -2, "x2": -3, "x3": -1}
    assert transformed.objective.constant == 3
    assert any("variables=x1,x3" in comment for comment in transformed.comments)
