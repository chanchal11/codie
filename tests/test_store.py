"""Home directory layout. CODIE_HOME is set by conftest."""

from pathlib import Path

from codie.store import (
    append_guideline,
    codie_home,
    delete_guideline,
    guideline_lines,
    load_config,
    save_config,
    workspace_key,
)


def test_home_comes_from_the_environment(isolated_codie_home: Path):
    assert codie_home() == isolated_codie_home


def test_workspace_key_is_stable(tmp_path: Path):
    assert workspace_key(tmp_path) == workspace_key(tmp_path.resolve())
    other = tmp_path / "other"
    other.mkdir()
    assert workspace_key(tmp_path) != workspace_key(other)


def test_config_round_trip_ignores_unknown_keys():
    save_config({"search_provider": "Tavily", "last_mode": "plan", "api_key": "secret"})
    loaded = load_config()
    assert loaded["search_provider"] == "tavily"
    assert loaded["last_mode"] == "plan"
    assert "api_key" not in loaded


def test_guidelines_append_and_delete(tmp_path: Path):
    assert append_guideline(tmp_path, "  prefer\nsmall edits  ") is None
    assert append_guideline(tmp_path, "prefer small edits") is None
    assert guideline_lines(tmp_path) == ["prefer small edits"]
    assert delete_guideline(tmp_path, 0) is None
    assert guideline_lines(tmp_path) == []
