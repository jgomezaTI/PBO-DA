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


def test_native_runtime_uses_resolved_linux_path(repo_tmp_path):
    path = repo_tmp_path / "dataset"

    assert backpas._runtime_path(path, runtime="native", distribution="ignored") == (
        path.resolve().as_posix()
    )


def test_native_environment_runs_micromamba_without_wsl():
    command = backpas._environment_command(
        "python train.py --use_cuda", runtime="native", distribution="ignored"
    )

    assert command[:2] == ["bash", "-lc"]
    assert "micromamba" in command[2]
    assert "--use_cuda" in command[2]
    assert "wsl" not in command


def test_default_runtime_matches_host(monkeypatch):
    monkeypatch.setattr(backpas.platform, "system", lambda: "Linux")
    assert backpas.default_runtime() == "native"
    monkeypatch.setattr(backpas.platform, "system", lambda: "Windows")
    assert backpas.default_runtime() == "wsl"
