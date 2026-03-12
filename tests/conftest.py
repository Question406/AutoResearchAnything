from pathlib import Path

import pytest

from aura.workspace import Workspace


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Workspace:
    return Workspace.create(tmp_path / "test_run")
