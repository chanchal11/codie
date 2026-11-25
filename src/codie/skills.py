"""SKILL.md files. The prompt lists names; the model loads a body with read_skill."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codie.guards import cap_output, check_path, is_secret_name
from codie.store import skills_dir

CATALOG_LIMIT = 4_000


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path


def discover_skills(workspace: Path) -> list[Skill]:
    """Personal skills first, then workspace skills. The workspace copy wins on a name clash."""
    found: dict[str, Skill] = {}
    for path in _iter_skill_files(skills_dir()):
        skill = _load_skill(path, fallback_name=path.parent.name)
        if skill:
            found[skill.name] = skill
    workspace_root = Path(workspace).resolve() / ".codie" / "skills"
    for path in _iter_skill_files(workspace_root):
        relative = path.relative_to(Path(workspace).resolve())
        if check_path(Path(workspace).resolve(), relative.as_posix()):
            continue
        skill = _load_skill(path, fallback_name=path.parent.name)
        if skill:
            found[skill.name] = skill
    return sorted(found.values(), key=lambda item: item.name)


def find_skill(workspace: Path, name: str) -> Skill | None:
    wanted = name.strip()
    if not wanted:
        return None
    for skill in discover_skills(workspace):
        if skill.name == wanted:
            return skill
    return None


def catalog_text(workspace: Path) -> str:
    skills = discover_skills(workspace)
    if not skills:
        return ""
    lines = [f"- {skill.name}: {skill.description or 'No description.'}" for skill in skills]
    return cap_output("\n".join(lines), CATALOG_LIMIT)


def project_guidelines(workspace: Path) -> str:
    """Committed project notes. These are read-only context, not learned memory."""
    root = Path(workspace).resolve()
    parts: list[str] = []
    for relative in ("AGENTS.md", ".codie/GUIDELINES.md"):
        if check_path(root, relative):
            continue
        path = (root / relative).resolve()
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        cleaned = text.strip()
        if not cleaned:
            continue
        parts.append(f"## {relative}\n{cap_output(cleaned, 4_000)}")
    return "\n\n".join(parts)


def _iter_skill_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("SKILL.md") if path.is_file())


def _load_skill(path: Path, *, fallback_name: str) -> Skill | None:
    if any(is_secret_name(part) for part in path.parts):
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    name, description, body = _split_frontmatter(text)
    name = name or fallback_name
    name = name.strip()
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        return None
    return Skill(name=name, description=description.strip(), body=body.strip(), path=path)


def _split_frontmatter(text: str) -> tuple[str, str, str]:
    stripped = text.lstrip("\ufeff")
    if not stripped.startswith("---"):
        return "", "", stripped.strip()
    parts = stripped.split("---", 2)
    if len(parts) < 3:
        return "", "", stripped.strip()
    meta: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip().lower()] = value.strip().strip("\"'")
    return meta.get("name", ""), meta.get("description", ""), parts[2].strip()
