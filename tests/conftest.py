import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FOVEATE_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("FOVEATE_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("FOVEATE_CACHE", str(tmp_path / "cache"))
    return tmp_path
