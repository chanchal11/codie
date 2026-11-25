"""Saved conversations for one workspace. The TUI loads these; the one-shot CLI does not."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from codie.store import atomic_write_json, conversations_dir

TITLE_LIMIT = 48


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    title: str
    updated_at: str
    mode: str
    provider: str
    model: str
    search_provider: str


@dataclass
class SessionData:
    session_id: str
    title: str
    updated_at: str
    mode: str
    provider: str
    model: str
    search_provider: str
    messages: list


class SessionRecorder:
    """Remembers the open conversation and writes it after a successful turn."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.session_id = str(uuid.uuid4())
        self.provider = "ollama"
        self.model = ""
        self.search_provider = ""

    def rotate(self) -> str:
        self.session_id = str(uuid.uuid4())
        return self.session_id

    def save(self, messages: list, mode: str) -> None:
        if not any(isinstance(message, HumanMessage) for message in messages):
            return
        path = _session_path(self.workspace, self.session_id)
        title = _title_from(messages)
        updated = _now()
        if path.is_file():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
            if isinstance(previous, dict) and str(previous.get("title") or "").strip():
                title = str(previous["title"])
        payload = {
            "id": self.session_id,
            "title": title,
            "workspace": str(self.workspace),
            "updated_at": updated,
            "mode": mode if mode in {"agent", "plan"} else "agent",
            "provider": self.provider,
            "model": self.model,
            "search_provider": self.search_provider,
            "messages": [dump_message(message) for message in messages],
        }
        atomic_write_json(path, payload)


def list_sessions(workspace: Path) -> tuple[list[SessionInfo], list[str]]:
    root = conversations_dir(workspace)
    found: list[SessionInfo] = []
    errors: list[str] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not data.get("id"):
                raise ValueError("missing id")
            found.append(
                SessionInfo(
                    session_id=str(data["id"]),
                    title=str(data.get("title") or "conversation"),
                    updated_at=str(data.get("updated_at") or ""),
                    mode=str(data.get("mode") or "agent"),
                    provider=str(data.get("provider") or "ollama"),
                    model=str(data.get("model") or ""),
                    search_provider=str(data.get("search_provider") or ""),
                )
            )
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            errors.append(path.name)
    found.sort(key=lambda item: item.updated_at, reverse=True)
    return found, errors


def load_session(workspace: Path, session_id: str) -> SessionData | str:
    path = _session_path(workspace, session_id)
    if not path.is_file():
        return "error: conversation not found"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "error: conversation file is unreadable"
    if not isinstance(data, dict):
        return "error: conversation file is unreadable"
    messages = []
    for raw in data.get("messages") or []:
        message = load_message(raw)
        if message is not None:
            messages.append(message)
    mode = str(data.get("mode") or "agent")
    if mode not in {"agent", "plan"}:
        mode = "agent"
    return SessionData(
        session_id=str(data.get("id") or session_id),
        title=str(data.get("title") or "conversation"),
        updated_at=str(data.get("updated_at") or ""),
        mode=mode,
        provider=str(data.get("provider") or "ollama"),
        model=str(data.get("model") or ""),
        search_provider=str(data.get("search_provider") or ""),
        messages=messages,
    )


def dump_message(message) -> dict:
    if isinstance(message, HumanMessage):
        return {"type": "human", "content": _text(message)}
    if isinstance(message, ToolMessage):
        return {
            "type": "tool",
            "content": _text(message),
            "tool_call_id": str(getattr(message, "tool_call_id", "") or ""),
            "name": str(getattr(message, "name", "") or ""),
        }
    if isinstance(message, AIMessage):
        calls = []
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(call, dict):
                calls.append(
                    {
                        "name": str(call.get("name", "")),
                        "args": call.get("args") if isinstance(call.get("args"), dict) else {},
                        "id": str(call.get("id") or ""),
                        "type": "tool_call",
                    }
                )
        return {"type": "ai", "content": _text(message), "tool_calls": calls}
    return {"type": "ai", "content": _text(message), "tool_calls": []}


def load_message(raw: object):
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("type") or "")
    content = raw.get("content")
    text = content if isinstance(content, str) else ""
    if kind == "human":
        return HumanMessage(content=text)
    if kind == "tool":
        return ToolMessage(
            content=text,
            tool_call_id=str(raw.get("tool_call_id") or ""),
            name=str(raw.get("name") or ""),
        )
    if kind == "ai":
        calls = []
        for call in raw.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            calls.append(
                {
                    "name": str(call.get("name", "")),
                    "args": args,
                    "id": str(call.get("id") or ""),
                    "type": "tool_call",
                }
            )
        return AIMessage(content=text, tool_calls=calls)
    return None


def _session_path(workspace: Path, session_id: str) -> Path:
    safe = "".join(char for char in session_id if char.isalnum() or char in {"-", "_"})
    return conversations_dir(workspace) / f"{safe}.json"


def _title_from(messages: list) -> str:
    for message in messages:
        if isinstance(message, HumanMessage):
            line = _text(message).strip().splitlines()
            title = line[0].strip() if line else ""
            if title:
                return title[:TITLE_LIMIT]
    return "conversation"


def _text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
