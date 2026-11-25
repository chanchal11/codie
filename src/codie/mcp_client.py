"""MCP stdio client. The human edits mcp.json from the TUI. The agent cannot."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from dataclasses import dataclass
from typing import Protocol

from codie.guards import cap_output
from codie.store import load_mcp

_NAME = re.compile(r"[^A-Za-z0-9_]+")


@dataclass(frozen=True)
class McpToolInfo:
    exposed_name: str
    server: str
    tool: str
    description: str
    input_schema: dict


class McpClient(Protocol):
    def discover(self) -> tuple[list[McpToolInfo], list[str]]:
        """Tools the model may call, and human-readable errors for servers that failed."""

    def call(self, server: str, tool: str, arguments: dict) -> str:
        """Plain tool result, or an ``error:`` string."""


def expose_name(server: str, tool: str) -> str:
    left = _NAME.sub("_", server).strip("_") or "server"
    right = _NAME.sub("_", tool).strip("_") or "tool"
    return f"mcp_{left}_{right}"


class StdioMcpClient:
    """Talks to enabled servers in mcp.json. One failed server does not hide the others."""

    def discover(self) -> tuple[list[McpToolInfo], list[str]]:
        servers = _enabled_servers()
        if not servers:
            return [], []
        found: list[McpToolInfo] = []
        errors: list[str] = []
        used: set[str] = set()
        for name, cfg in servers.items():
            try:
                tools = _run(_list_server(name, cfg))
            except Exception as exc:
                errors.append(f"{name}: {exc.__class__.__name__}")
                continue
            for tool in tools:
                exposed = expose_name(name, tool["name"])
                suffix = 2
                while exposed in used:
                    exposed = f"{expose_name(name, tool['name'])}_{suffix}"
                    suffix += 1
                used.add(exposed)
                schema = tool.get("input_schema")
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}}
                found.append(
                    McpToolInfo(
                        exposed_name=exposed,
                        server=name,
                        tool=str(tool["name"]),
                        description=str(tool.get("description") or ""),
                        input_schema=schema,
                    )
                )
        return found, errors

    def call(self, server: str, tool: str, arguments: dict) -> str:
        cfg = _enabled_servers().get(server)
        if cfg is None:
            return f"error: mcp server '{server}' is not enabled"
        try:
            text = _run(_call_server(cfg, tool, arguments))
        except Exception as exc:
            return f"error: mcp server unavailable ({exc.__class__.__name__})"
        return cap_output(text)


class EmptyMcpClient:
    """Used when the caller wants no MCP process at all."""

    def discover(self) -> tuple[list[McpToolInfo], list[str]]:
        return [], []

    def call(self, server: str, tool: str, arguments: dict) -> str:
        return "error: mcp is not configured"


def _enabled_servers() -> dict[str, dict]:
    servers = load_mcp().get("mcpServers") or {}
    enabled: dict[str, dict] = {}
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("enabled", True) is False:
            continue
        command = str(cfg.get("command") or "").strip()
        if not command:
            continue
        enabled[name] = cfg
    return enabled


def _run(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict = {}

    def runner() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - handed back to the caller
            box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


async def _list_server(name: str, cfg: dict) -> list[dict]:
    session_factory = _open_session(cfg)

    async def action(session):
        listed = await asyncio.wait_for(session.list_tools(), 30)
        tools = []
        for tool in getattr(listed, "tools", []) or []:
            tools.append(
                {
                    "name": str(getattr(tool, "name", "")),
                    "description": str(getattr(tool, "description", "") or ""),
                    "input_schema": getattr(tool, "inputSchema", None) or {},
                }
            )
        return [tool for tool in tools if tool["name"]]

    return await session_factory(action)


async def _call_server(cfg: dict, tool: str, arguments: dict) -> str:
    async def action(session):
        result = await asyncio.wait_for(session.call_tool(tool, arguments or {}), 30)
        if getattr(result, "isError", False):
            return "error: " + _content_text(result)
        return _content_text(result)

    return await _open_session(cfg)(action)


def _open_session(cfg: dict):
    command = str(cfg.get("command") or "").strip()
    args = cfg.get("args") or []
    if not isinstance(args, list):
        args = []
    env = cfg.get("env") if isinstance(cfg.get("env"), dict) else None
    cleaned_env = None
    if env:
        cleaned_env = {str(key): str(value) for key, value in env.items()}

    async def run(action):
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError("mcp package is not installed") from exc
        params = StdioServerParameters(
            command=command,
            args=[str(item) for item in args],
            env=cleaned_env,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), 30)
                return await action(session)

    return run


def _content_text(result) -> str:
    parts: list[str] = []
    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
        else:
            try:
                parts.append(json.dumps(block, default=str))
            except TypeError:
                parts.append(str(block))
    return "\n".join(parts).strip() or "(empty)"
