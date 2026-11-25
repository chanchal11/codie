"""Keep every test off the real ~/.codie directory."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_codie_home(monkeypatch, tmp_path_factory):
    home = tmp_path_factory.mktemp("codie-home")
    monkeypatch.setenv("CODIE_HOME", str(home))
    return home
