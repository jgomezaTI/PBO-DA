from types import SimpleNamespace

import pytest

from backbone_pbo.backbone import BackboneFormatError
from backbone_pbo.dataset import backpas
from backbone_pbo.dataset.backpas import normalize_guroback_output


def test_normalize_guroback_output_restores_x_prefix():
    result = normalize_guroback_output("b -1\nb 2\nb 0\n", ["x1", "x2", "x3"])
    assert result == "b -x1\nb x2\nb 0\n"


def test_normalize_guroback_output_restores_dataset_prefix():
    result = normalize_guroback_output("b -767\nb 42\nb 0\n", ["v42", "v767"])
    assert result == "b v42\nb -v767\nb 0\n"


def test_normalize_guroback_output_rejects_ambiguous_name():
    with pytest.raises(BackboneFormatError, match="2 matches"):
        normalize_guroback_output("b 1\nb 0\n", ["1", "x1"])


def test_windows_to_wsl_uses_portable_slashes(monkeypatch, repo_tmp_path):
    captured = []

    def fake_run(command, **_kwargs):
        captured.extend(command)
        return SimpleNamespace(returncode=0, stdout="/mnt/c/project\n", stderr="")

    monkeypatch.setattr(backpas.subprocess, "run", fake_run)

    assert backpas.windows_to_wsl(repo_tmp_path) == "/mnt/c/project"
    assert "\\" not in captured[-1]
