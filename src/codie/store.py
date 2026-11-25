"""Files under ~/.codie. Every other module asks here instead of building paths."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

GUIDELINE_FILE_LIMIT = 8_000
GUIDELINE_LINE_LIMIT = 500
MAX_GUIDELINES = 40

DEFAULT_CONFIG: dict = {
    "search_provider": "",
    "searxng_url": "http://127.0.0.1:8080",
    "openai_search_model": "gpt-4.1-mini",
    "last_mode": "agent",
}

DEFAULT_MCP: dict = {"mcpServers": {}}
DEFAULT_PERMISSIONS: dict = {"rules": []}


def codie_home() -> Path:
    """Home directory for config, sessions, and skills.

    ``CODIE_HOME`` overrides the path so tests never touch a real account.
    """
    override = os.environ.get("CODIE_HOME")
    if override and override.strip():
        return Path(override).expanduser().resolve()
    return (Path.home() / ".codie").resolve()


def workspace_key(workspace: Path) -> str:
    """Stable id for one resolved folder. The path itself is stored beside the data."""
    resolved = str(Path(workspace).resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp = Path(handle.name)
    try:
        handle.write(text)
        handle.close()
        os.replace(tmp, path)
    except Exception:
        handle.close()
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: object) -> None:
    atomic_write_text(path, json.dumps(data, indent=2) + "\n")


def read_json(path: Path, default: dict) -> dict:
    if not path.is_file():
        return json.loads(json.dumps(default))
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return json.loads(json.dumps(default))
    if not isinstance(loaded, dict):
        return json.loads(json.dumps(default))
    return loaded


def config_path() -> Path:
    return codie_home() / "config.json"


def load_config() -> dict:
    data = read_json(config_path(), DEFAULT_CONFIG)
    merged = dict(DEFAULT_CONFIG)
    merged.update({key: value for key, value in data.items() if key in DEFAULT_CONFIG})
    if merged.get("last_mode") not in {"agent", "plan"}:
        merged["last_mode"] = "agent"
    provider = merged.get("search_provider")
    merged["search_provider"] = provider.strip().lower() if isinstance(provider, str) else ""
    return merged


def save_config(data: dict) -> None:
    current = load_config()
    for key in DEFAULT_CONFIG:
        if key in data:
            current[key] = data[key]
    if current.get("last_mode") not in {"agent", "plan"}:
        current["last_mode"] = "agent"
    atomic_write_json(config_path(), current)


def mcp_path() -> Path:
    return codie_home() / "mcp.json"


def load_mcp() -> dict:
    data = read_json(mcp_path(), DEFAULT_MCP)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        return {"mcpServers": {}}
    clean: dict[str, dict] = {}
    for name, cfg in servers.items():
        if not isinstance(name, str) or not isinstance(cfg, dict):
            continue
        clean[name] = cfg
    return {"mcpServers": clean}


def save_mcp(data: dict) -> None:
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        servers = {}
    atomic_write_json(mcp_path(), {"mcpServers": servers})


def workspace_root(workspace: Path) -> Path:
    return codie_home() / "workspaces" / workspace_key(workspace)


def remember_workspace(workspace: Path) -> Path:
    """Create the workspace record the first time we store something for it."""
    root = workspace_root(workspace)
    root.mkdir(parents=True, exist_ok=True)
    meta = root / "meta.json"
    if not meta.is_file():
        atomic_write_json(
            meta,
            {"path": str(Path(workspace).resolve()), "key": workspace_key(workspace)},
        )
    return root


def workspace_dir(workspace: Path) -> Path:
    return remember_workspace(workspace)


def conversations_dir(workspace: Path) -> Path:
    root = codie_home() / "conversations" / workspace_key(workspace)
    root.mkdir(parents=True, exist_ok=True)
    return root


def skills_dir() -> Path:
    root = codie_home() / "skills"
    root.mkdir(parents=True, exist_ok=True)
    return root


def guidelines_path(workspace: Path) -> Path:
    return workspace_root(workspace) / "guidelines.md"


def read_guidelines(workspace: Path) -> str:
    path = guidelines_path(workspace)
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def guideline_lines(workspace: Path) -> list[str]:
    lines: list[str] = []
    for raw in read_guidelines(workspace).splitlines():
        text = raw.strip()
        if text.startswith("- "):
            text = text[2:].strip()
        if text:
            lines.append(text)
    return lines


def append_guideline(workspace: Path, text: str) -> str | None:
    """Append one preference. Returns an error string, or None on success."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return "error: empty guideline"
    if len(cleaned) > GUIDELINE_LINE_LIMIT:
        return f"error: guideline is longer than {GUIDELINE_LINE_LIMIT} characters"
    lines = guideline_lines(workspace)
    if cleaned in lines:
        return None
    if len(lines) >= MAX_GUIDELINES:
        return "error: too many guidelines for this workspace; delete one in the TUI"
    lines.append(cleaned)
    remember_workspace(workspace)
    body = "".join(f"- {line}\n" for line in lines)
    if len(body) > GUIDELINE_FILE_LIMIT:
        return "error: guidelines file is full; delete one in the TUI"
    atomic_write_text(guidelines_path(workspace), body)
    return None


def delete_guideline(workspace: Path, index: int) -> str | None:
    lines = guideline_lines(workspace)
    if index < 0 or index >= len(lines):
        return "error: no guideline at that position"
    del lines[index]
    remember_workspace(workspace)
    body = "".join(f"- {line}\n" for line in lines)
    atomic_write_text(guidelines_path(workspace), body)
    return None


def permissions_path(workspace: Path) -> Path:
    return workspace_root(workspace) / "permissions.json"


def load_permissions_data(workspace: Path) -> dict:
    data = read_json(permissions_path(workspace), DEFAULT_PERMISSIONS)
    rules = data.get("rules")
    if not isinstance(rules, list):
        return {"rules": []}
    clean = [rule for rule in rules if isinstance(rule, dict)]
    return {"rules": clean}


def save_permissions_data(workspace: Path, data: dict) -> None:
    rules = data.get("rules") if isinstance(data, dict) else None
    if not isinstance(rules, list):
        rules = []
    remember_workspace(workspace)
    atomic_write_json(permissions_path(workspace), {"rules": rules})
