"""Fixtures que mantienen los artefactos efímeros dentro del repositorio."""

from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def repo_tmp_path(request: pytest.FixtureRequest) -> Path:
    """Crea un directorio efímero local en ``.tmp/pytest``.

    El sandbox de desarrollo puede bloquear el directorio temporal global de Windows.
    Estos artefactos no se versionan porque ``.tmp/`` está en ``.gitignore``.
    """

    repository_root = Path(__file__).resolve().parents[1]
    test_root = repository_root / ".tmp" / "pytest-local"
    test_root.mkdir(parents=True, exist_ok=True)
    directory = test_root / f"{request.node.name}-{uuid4().hex[:8]}"
    directory.mkdir()
    return directory
