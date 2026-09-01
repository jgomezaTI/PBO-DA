"""Fixtures that keep ephemeral artifacts inside the repository."""

from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def repo_tmp_path(request: pytest.FixtureRequest) -> Path:
    """Create a local ephemeral directory under ``.tmp/pytest``.

    The development sandbox may block Windows' global temporary directory. These
    artifacts are not versioned because ``.tmp/`` is listed in ``.gitignore``.
    """

    repository_root = Path(__file__).resolve().parents[1]
    test_root = repository_root / ".tmp" / "pytest-local"
    test_root.mkdir(parents=True, exist_ok=True)
    directory = test_root / f"{request.node.name}-{uuid4().hex[:8]}"
    directory.mkdir()
    return directory
