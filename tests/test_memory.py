"""Skills and project instructions. Learned guidelines are covered with the agent loop."""

from pathlib import Path

from codie.prompts import system_prompt
from codie.skills import discover_skills, find_skill
from codie.tools import ToolContext
from codie.approvals import AllowAll


def test_skill_catalog_hides_the_body_until_read(tmp_path: Path):
    skill = tmp_path / ".codie" / "skills" / "pytest" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: pytest\ndescription: Use when tests fail.\n---\n"
        "BODY-MARKER run pytest -q after the fix.\n",
        encoding="utf-8",
    )
    prompt = system_prompt(tmp_path)
    assert "Use when tests fail." in prompt
    assert "BODY-MARKER" not in prompt
    found = discover_skills(tmp_path)
    assert [item.name for item in found] == ["pytest"]
    loaded = find_skill(tmp_path, "pytest")
    assert loaded is not None and "BODY-MARKER" in loaded.body
    result = ToolContext(tmp_path, AllowAll()).read_skill("pytest")
    assert "BODY-MARKER" in result


def test_project_instructions_are_included(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("Keep the public functions stable.\n", encoding="utf-8")
    prompt = system_prompt(tmp_path)
    assert "Keep the public functions stable." in prompt
    assert "Project instructions" in prompt


def test_skills_come_only_from_this_workspace(tmp_path: Path, isolated_codie_home: Path):
    home_skill = isolated_codie_home / "skills" / "notes" / "SKILL.md"
    home_skill.parent.mkdir(parents=True)
    home_skill.write_text(
        "---\nname: notes\ndescription: personal\n---\nPERSONAL-BODY\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "ws"
    other = tmp_path / "other"
    local = workspace / ".codie" / "skills" / "local" / "SKILL.md"
    elsewhere = other / ".codie" / "skills" / "notes" / "SKILL.md"
    local.parent.mkdir(parents=True)
    elsewhere.parent.mkdir(parents=True)
    local.write_text(
        "---\nname: local\ndescription: this workspace\n---\nLOCAL-BODY\n",
        encoding="utf-8",
    )
    elsewhere.write_text(
        "---\nname: notes\ndescription: other workspace\n---\nOTHER-BODY\n",
        encoding="utf-8",
    )
    names = [skill.name for skill in discover_skills(workspace)]
    assert names == ["local"]
    assert find_skill(workspace, "notes") is None
    assert "PERSONAL-BODY" not in system_prompt(workspace)
    assert "OTHER-BODY" not in system_prompt(workspace)
    assert "this workspace" in system_prompt(workspace)
