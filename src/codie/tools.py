"""Workspace tools. Every path and command goes through the guards first."""

from __future__ import annotations

import json
import os
import re
import subprocess
from difflib import unified_diff
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, create_model

from codie.approvals import DENIED, ApprovalRequest, Approver, NullSink, StepSink
from codie.guards import (
    COMMAND_TIMEOUT_SECONDS,
    cap_output,
    is_secret_name,
    path_is_secret,
    plan_command,
    resolve_inside,
    scrub_grep,
    scrub_listing,
    strip_ansi,
)
from codie.mcp_client import EmptyMcpClient, McpToolInfo
from codie.permissions import PermissionSet, normalize_rel
from codie.search import run_search
from codie.skills import find_skill
from codie.store import append_guideline

_PYTEST_FAILED = re.compile(r"(\d+)\s+failed")
_PYTEST_PASSED = re.compile(r"(\d+)\s+passed")

PLAN_TOOLS = {
    "list_files",
    "read_file",
    "find_files",
    "grep",
    "web_search",
    "read_skill",
    "switch_mode",
    "save_guideline",
}

READ_LIMIT = 1_000_000
GREP_HITS = 200


class ListFilesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="Directory relative to the workspace. Use '.' for the root.")


class ReadFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="File relative to the workspace.")
    start_line: int = Field(
        default=0,
        description="First line to return, 1-based. 0 starts at the beginning.",
    )
    end_line: int = Field(
        default=0,
        description="Last line to return, inclusive. 0 reads through the end.",
    )


class FindFilesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pattern: str = Field(description="Glob relative to the workspace, such as '**/*.py'.")


class WriteFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="File relative to the workspace.")
    content: str = Field(description="Full new file contents.")


class EditFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="File relative to the workspace.")
    old: str = Field(description="Exact text to replace. It must occur once.")
    new: str = Field(description="Replacement text.")


class RunCommandArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: str = Field(
        description="One command with no shell syntax. Free commands run immediately. Other bare programs wait for the user."
    )


class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="Search query for recent public information.")


class GrepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pattern: str = Field(description="Python regular expression.")
    path: str = Field(
        default=".",
        description="File or directory relative to the workspace. '.' searches the whole folder.",
    )


class SwitchModeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str = Field(description="Either 'plan' or 'agent'.")
    reason: str = Field(default="", description="Why the mode should change.")


class SaveGuidelineArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(description="One short workspace preference the user asked you to remember.")


class ReadSkillArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="Skill name from the system prompt catalog.")


def make_diff(path: str, old: str, new: str) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    if old_lines and not old_lines[-1].endswith("\n"):
        old_lines[-1] += "\n"
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"
    text = "".join(
        unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    return text or "(no changes)\n"


def quote_arg(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def format_call(name: str, args: dict) -> str:
    if name == "run_command":
        return f"-> run_command({quote_arg(str(args.get('command', '')))})"
    if name == "web_search":
        return f"-> web_search({quote_arg(str(args.get('query', '')))})"
    if name == "find_files":
        return f"-> find_files({quote_arg(str(args.get('pattern', '')))})"
    if name == "write_file":
        return f"-> write_file({quote_arg(str(args.get('path', '')))})"
    if name == "edit_file":
        return f"-> edit_file({quote_arg(str(args.get('path', '')))})"
    if name == "grep":
        return f"-> grep({quote_arg(str(args.get('pattern', '')))})"
    if name == "switch_mode":
        return f"-> switch_mode({quote_arg(str(args.get('mode', '')))})"
    if name == "save_guideline":
        return "-> save_guideline(...)"
    if name == "read_skill":
        return f"-> read_skill({quote_arg(str(args.get('name', '')))})"
    if name in {"list_files", "read_file"}:
        return f"-> {name}({quote_arg(str(args.get('path', '')))})"
    if name.startswith("mcp_"):
        return f"-> {name}(...)"
    rendered = ", ".join(f"{key}={quote_arg(_short(val))}" for key, val in args.items())
    return f"-> {name}({rendered})"


def summarize_result(name: str, result: str) -> str:
    first = result.splitlines()[0] if result else "ok"
    if first.startswith("exit code") or first.startswith("error:"):
        return first[:200]
    if name == "list_files":
        count = len([line for line in result.splitlines() if line.strip() and line != "(empty)"])
        return f"{count} entries"
    if name == "read_file":
        return f"{len(result.splitlines())} lines"
    if name == "find_files":
        count = len([line for line in result.splitlines() if line.strip() and line != "(no matches)"])
        return f"{count} matches"
    if name in {"write_file", "edit_file"}:
        return first[:200]
    if name == "grep":
        count = len([line for line in result.splitlines() if line.strip() and line != "(no matches)"])
        return f"{count} hits"
    if name == "web_search":
        count = result.count("\n\n") + (1 if result.strip() else 0)
        return f"{count} results"
    if name == "read_skill":
        return f"{len(result.splitlines())} lines"
    return first[:200]


def action_needs_approval(workspace: Path, mode: str, name: str, args: dict, mcp_tools: dict) -> bool:
    """True when the graph should pause for a human. Plan-mode blocks are not approvals."""
    if name == "switch_mode":
        target = str(args.get("mode", "")).strip().lower()
        return mode == "plan" and target == "agent"
    if name == "save_guideline":
        return bool(str(args.get("text", "")).strip())
    if mode == "plan":
        return False
    if name in mcp_tools:
        info = mcp_tools[name]
        match = f"{info.server}/{info.tool}"
        if PermissionSet.load(workspace).allows(name, match):
            return False
        return True
    if name in {"write_file", "edit_file"}:
        match = normalize_rel(str(args.get("path", "")))
        if match and PermissionSet.load(workspace).allows(name, match):
            return False
        return True
    if name == "run_command":
        plan = plan_command(workspace, str(args.get("command", "")))
        if isinstance(plan, str) or not plan.needs_approval:
            return False
        permissions = PermissionSet.load(workspace)
        program = plan.argv[0]
        if permissions.is_blocked(program):
            return False
        if program == "pytest" and permissions.allows("run_command", "pytest"):
            return False
        return True
    return False


class ToolContext:
    def __init__(
        self,
        workspace: Path,
        approver: Approver,
        sink: StepSink | None = None,
        *,
        mode: str = "agent",
        mcp_tools: dict[str, McpToolInfo] | None = None,
        mcp_client=None,
        on_mode_change=None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.approver = approver
        self.sink = sink or NullSink()
        self.mode = mode if mode in {"agent", "plan"} else "agent"
        self.mcp_tools = dict(mcp_tools or {})
        self.mcp_client = mcp_client or EmptyMcpClient()
        self.on_mode_change = on_mode_change

    def list_files(self, path: str) -> str:
        self.sink.step(format_call("list_files", {"path": path}))
        result = self._list_files(path)
        self.sink.step(f"<- {summarize_result('list_files', result)}")
        return result

    def read_file(self, path: str, start_line: int = 0, end_line: int = 0) -> str:
        self.sink.step(format_call("read_file", {"path": path}))
        result = self._read_file(path, start_line=start_line, end_line=end_line)
        self.sink.step(f"<- {summarize_result('read_file', result)}")
        return result

    def find_files(self, pattern: str) -> str:
        self.sink.step(format_call("find_files", {"pattern": pattern}))
        result = self._find_files(pattern)
        self.sink.step(f"<- {summarize_result('find_files', result)}")
        return result

    def write_file(self, path: str, content: str, *, approved: bool | None = None) -> str:
        self.sink.step(format_call("write_file", {"path": path}))
        result = self._write_file(path, content, approved=approved)
        self.sink.step(f"<- {summarize_result('write_file', result)}")
        return result

    def edit_file(self, path: str, old: str, new: str, *, approved: bool | None = None) -> str:
        self.sink.step(format_call("edit_file", {"path": path}))
        result = self._edit_file(path, old, new, approved=approved)
        self.sink.step(f"<- {summarize_result('edit_file', result)}")
        return result

    def run_command(self, command: str, *, approved: bool | None = None) -> str:
        self.sink.step(format_call("run_command", {"command": command}))
        result = self._run_command(command, approved=approved)
        self.sink.step(f"<- {summarize_result('run_command', result)}")
        return result

    def web_search(self, query: str) -> str:
        self.sink.step(format_call("web_search", {"query": query}))
        result = run_search(query)
        self.sink.step(f"<- {summarize_result('web_search', result)}")
        return result

    def grep(self, pattern: str, path: str = ".") -> str:
        self.sink.step(format_call("grep", {"pattern": pattern}))
        result = self._grep(pattern, path)
        self.sink.step(f"<- {summarize_result('grep', result)}")
        return result

    def switch_mode(self, mode: str, reason: str = "", *, approved: bool | None = None) -> str:
        self.sink.step(format_call("switch_mode", {"mode": mode}))
        result = self._switch_mode(mode, reason, approved=approved)
        self.sink.step(f"<- {summarize_result('switch_mode', result)}")
        return result

    def save_guideline(self, text: str, *, approved: bool | None = None) -> str:
        self.sink.step(format_call("save_guideline", {}))
        result = self._save_guideline(text, approved=approved)
        self.sink.step(f"<- {summarize_result('save_guideline', result)}")
        return result

    def read_skill(self, name: str) -> str:
        self.sink.step(format_call("read_skill", {"name": name}))
        result = self._read_skill(name)
        self.sink.step(f"<- {summarize_result('read_skill', result)}")
        return result

    def preview_write(self, call_id: str, path: str, content: str) -> ApprovalRequest | str:
        resolved = resolve_inside(self.workspace, path)
        if isinstance(resolved, str):
            return resolved
        if resolved.exists() and not resolved.is_file():
            return "error: path is not a file"
        old = ""
        if resolved.is_file():
            try:
                old = resolved.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return "error: file is not utf-8 text"
        return ApprovalRequest(
            call_id=call_id,
            kind="write",
            summary=f"write_file({quote_arg(path)})",
            diff=make_diff(path, old, content),
            path=path,
        )

    def preview_edit(self, call_id: str, path: str, old: str, new: str) -> ApprovalRequest | str:
        updated = self._apply_edit(path, old, new)
        if isinstance(updated, str):
            return updated
        current, revised = updated
        return ApprovalRequest(
            call_id=call_id,
            kind="edit",
            summary=f"edit_file({quote_arg(path)})",
            diff=make_diff(path, current, revised),
            path=path,
        )

    def preview_command(self, call_id: str, command: str) -> ApprovalRequest | str:
        plan = plan_command(self.workspace, command)
        if isinstance(plan, str):
            return plan
        if not plan.needs_approval:
            return "error: command does not need approval"
        program = plan.argv[0]
        if PermissionSet.load(self.workspace).is_blocked(program):
            return f"error: '{program}' is blocked for this workspace"
        kind = "pytest" if program == "pytest" else "command"
        return ApprovalRequest(
            call_id=call_id,
            kind=kind,
            summary=f"run_command({quote_arg(command)})",
            command=command,
        )

    def preview_mcp(self, call_id: str, name: str, args: dict) -> ApprovalRequest | str:
        info = self.mcp_tools.get(name)
        if info is None:
            return f"error: unknown tool '{name}'"
        shown = cap_output(json.dumps(args, default=str), 1500)
        return ApprovalRequest(
            call_id=call_id,
            kind="mcp",
            summary=name,
            command=f"{info.server}/{info.tool}\n{shown}",
            path=f"{info.server}/{info.tool}",
        )

    def dispatch(self, name: str, args: dict, *, approved: bool | None = None) -> str:
        if self.mode == "plan" and name not in PLAN_TOOLS:
            self.sink.step(format_call(name, args))
            result = "error: plan mode cannot change the workspace"
            self.sink.step(f"<- {summarize_result(name, result)}")
            return result
        try:
            if name in self.mcp_tools:
                return self._call_mcp(name, args, approved=approved)
            if name == "list_files":
                return self.list_files(args["path"])
            if name == "read_file":
                start, end = _line_args(args)
                if isinstance(start, str):
                    return start
                return self.read_file(args["path"], start_line=start, end_line=end)
            if name == "find_files":
                return self.find_files(args["pattern"])
            if name == "write_file":
                return self.write_file(args["path"], args["content"], approved=approved)
            if name == "edit_file":
                return self.edit_file(args["path"], args["old"], args["new"], approved=approved)
            if name == "run_command":
                return self.run_command(args["command"], approved=approved)
            if name == "web_search":
                return self.web_search(args["query"])
            if name == "grep":
                return self.grep(args["pattern"], str(args.get("path") or "."))
            if name == "switch_mode":
                return self.switch_mode(
                    args["mode"],
                    str(args.get("reason") or ""),
                    approved=approved,
                )
            if name == "save_guideline":
                return self.save_guideline(args["text"], approved=approved)
            if name == "read_skill":
                return self.read_skill(args["name"])
        except KeyError as exc:
            missing = str(exc).strip("'")
            return f"error: missing field {missing}"
        self.sink.step(format_call(name, args))
        result = f"error: unknown tool '{name}'"
        self.sink.step(f"<- {summarize_result(name, result)}")
        return result

    def langchain_tools(self):
        """Schemas for the model. Execution goes through ``dispatch``."""
        from langchain_core.tools import StructuredTool

        specs = [
            ("list_files", "List one directory inside the workspace. Secret names are omitted.", ListFilesArgs, self.list_files),
            ("read_file", "Read a UTF-8 text file inside the workspace. Optional start_line and end_line select a range.", ReadFileArgs, self.read_file),
            ("find_files", "Glob for paths inside the workspace. Example: **/*.py.", FindFilesArgs, self.find_files),
            ("grep", "Search file contents with a Python regular expression. Secret files are skipped.", GrepArgs, self.grep),
            ("write_file", "Write the full new contents of a file. A human approves the diff first.", WriteFileArgs, self.write_file),
            ("edit_file", "Replace one exact snippet in a file. A human approves the diff first.", EditFileArgs, self.edit_file),
            (
                "run_command",
                "Run one command with no shell. Free: ls cat head tail wc grep find pwd. "
                "pytest and any other bare program wait for the user. A blocked program must not be retried.",
                RunCommandArgs,
                self.run_command,
            ),
            ("web_search", "Search the public web. Use this for versions and events after your training cutoff.", WebSearchArgs, self.web_search),
            ("read_skill", "Load the body of a skill listed in the system prompt.", ReadSkillArgs, self.read_skill),
            (
                "switch_mode",
                "Switch between plan and agent mode. Plan to agent waits for the user. Agent to plan applies immediately.",
                SwitchModeArgs,
                self.switch_mode,
            ),
            (
                "save_guideline",
                "Remember one workspace preference the user stated. A human approves it first.",
                SaveGuidelineArgs,
                self.save_guideline,
            ),
        ]
        tools = []
        for name, description, schema, func in specs:
            if self.mode == "plan" and name not in PLAN_TOOLS:
                continue
            tools.append(
                StructuredTool.from_function(
                    func=func,
                    name=name,
                    description=description,
                    args_schema=schema,
                )
            )
        if self.mode == "agent":
            tools.extend(self._mcp_langchain_tools())
        return tools

    def _list_files(self, path: str) -> str:
        resolved = resolve_inside(self.workspace, path)
        if isinstance(resolved, str):
            return resolved
        if not resolved.is_dir():
            return "error: path is not a directory"
        names: list[str] = []
        try:
            entries = sorted(resolved.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            return f"error: cannot list directory ({exc})"
        for entry in entries:
            if self._hidden_entry(entry):
                continue
            suffix = "/" if entry.is_dir() and not entry.is_symlink() else ""
            names.append(entry.name + suffix)
        if not names:
            return "(empty)"
        return cap_output("\n".join(names))

    def _read_file(self, path: str, start_line: int = 0, end_line: int = 0) -> str:
        resolved = resolve_inside(self.workspace, path)
        if isinstance(resolved, str):
            return resolved
        if not resolved.is_file():
            return "error: path is not a file"
        try:
            with resolved.open("r", encoding="utf-8") as handle:
                data = handle.read(READ_LIMIT)
        except UnicodeDecodeError:
            return "error: file is not utf-8 text"
        except OSError as exc:
            return f"error: cannot read file ({exc})"
        if start_line or end_line:
            return _slice_lines(data, start_line, end_line)
        return cap_output(data)

    def _find_files(self, pattern: str) -> str:
        if pattern is None or str(pattern).strip() == "":
            return "error: empty pattern"
        raw = Path(str(pattern))
        if raw.is_absolute():
            return "error: absolute paths are not allowed"
        if ".." in raw.parts:
            return "error: path escapes the workspace"
        if path_is_secret(raw):
            return f"error: '{_secret_in(raw)}' looks like a secret"
        try:
            matches = list(self.workspace.glob(str(pattern)))
        except (OSError, ValueError) as exc:
            return f"error: bad pattern ({exc})"
        found: list[str] = []
        root = self.workspace
        for match in matches:
            try:
                resolved = match.resolve()
                relative = resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            if path_is_secret(relative) or is_secret_name(match.name):
                continue
            found.append(relative.as_posix())
        if not found:
            return "(no matches)"
        return cap_output("\n".join(sorted(found)))

    def _write_file(self, path: str, content: str, *, approved: bool | None) -> str:
        if content.strip() in {"", "."}:
            return "error: write_file needs the full file contents, not a placeholder"
        resolved = resolve_inside(self.workspace, path)
        if isinstance(resolved, str):
            return resolved
        if resolved.suffix == ".py":
            try:
                compile(content, path, "exec")
            except SyntaxError as exc:
                line = f" on line {exc.lineno}" if exc.lineno else ""
                return f"error: file is not valid Python ({exc.msg}{line})"
        preview = self.preview_write("", path, content)
        if isinstance(preview, str):
            return preview
        if preview.diff.strip() == "(no changes)":
            return "error: content matches the current file"
        if not self._permitted(preview, approved):
            return DENIED
        resolved = resolve_inside(self.workspace, path)
        if isinstance(resolved, str):
            return resolved
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"error: cannot write file ({exc})"
        return f"wrote {path}"

    def _edit_file(self, path: str, old: str, new: str, *, approved: bool | None) -> str:
        preview = self.preview_edit("", path, old, new)
        if isinstance(preview, str):
            return preview
        if preview.diff.strip() == "(no changes)":
            return "error: content matches the current file"
        revised = self._apply_edit(path, old, new)
        if isinstance(revised, str):
            return revised
        _current, updated = revised
        if not self._permitted(preview, approved):
            return DENIED
        resolved = resolve_inside(self.workspace, path)
        if isinstance(resolved, str):
            return resolved
        try:
            resolved.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return f"error: cannot write file ({exc})"
        return f"edited {path}"

    def _apply_edit(self, path: str, old: str, new: str) -> tuple[str, str] | str:
        if old is None or str(old) == "":
            return "error: old text is empty"
        resolved = resolve_inside(self.workspace, path)
        if isinstance(resolved, str):
            return resolved
        if not resolved.is_file():
            return "error: path is not a file"
        try:
            current = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return "error: file is not utf-8 text"
        except OSError as exc:
            return f"error: cannot read file ({exc})"
        count = current.count(str(old))
        if count == 0:
            return "error: old text was not found"
        if count > 1:
            return f"error: old text matched {count} times; include more context"
        updated = current.replace(str(old), str(new), 1)
        if resolved.suffix == ".py":
            try:
                compile(updated, path, "exec")
            except SyntaxError as exc:
                line = f" on line {exc.lineno}" if exc.lineno else ""
                return f"error: file is not valid Python ({exc.msg}{line})"
        return current, updated

    def _run_command(self, command: str, *, approved: bool | None) -> str:
        plan = plan_command(self.workspace, command)
        if isinstance(plan, str):
            return plan
        program = plan.argv[0]
        if PermissionSet.load(self.workspace).is_blocked(program):
            return f"error: '{program}' is blocked for this workspace"
        if plan.needs_approval:
            kind = "pytest" if program == "pytest" else "command"
            request = ApprovalRequest(
                call_id="",
                kind=kind,
                summary=f"run_command({quote_arg(command)})",
                command=command,
            )
            if not self._permitted(request, approved):
                return DENIED
        return self._spawn(plan)

    def _spawn(self, plan) -> str:
        try:
            completed = subprocess.run(
                plan.argv,
                cwd=self.workspace,
                shell=False,
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "error: command timed out after 30s"
        except OSError as exc:
            return f"error: cannot run command ({exc})"
        output = strip_ansi(completed.stdout or "")
        if completed.stderr:
            error_text = strip_ansi(completed.stderr)
            output = f"{output}\n{error_text}" if output else error_text
        if plan.scrub == "listing":
            output = scrub_listing(output)
        elif plan.scrub == "grep":
            output = scrub_grep(output)
        output = output.strip("\n")
        head = _command_head(plan.argv, completed.returncode, output)
        body = cap_output(output)
        if body:
            return f"{head}\n{body}"
        return head

    def _grep(self, pattern: str, path: str) -> str:
        if pattern is None or str(pattern).strip() == "":
            return "error: empty pattern"
        try:
            regex = re.compile(str(pattern))
        except re.error as exc:
            return f"error: bad pattern ({exc})"
        resolved = resolve_inside(self.workspace, path or ".")
        if isinstance(resolved, str):
            return resolved
        if not resolved.exists():
            return "error: path does not exist"
        hits: list[str] = []
        for file in self._walk_files(resolved):
            try:
                if file.stat().st_size > READ_LIMIT:
                    continue
                text = file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            try:
                relative = file.resolve().relative_to(self.workspace).as_posix()
            except (OSError, ValueError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if not regex.search(line):
                    continue
                hits.append(f"{relative}:{number}:{line}")
                if len(hits) >= GREP_HITS:
                    hits.append("...[truncated]")
                    return cap_output("\n".join(hits))
        if not hits:
            return "(no matches)"
        return cap_output("\n".join(hits))

    def _switch_mode(self, mode: str, reason: str, *, approved: bool | None) -> str:
        target = str(mode or "").strip().lower()
        if target not in {"agent", "plan"}:
            return "error: mode must be agent or plan"
        if target == self.mode:
            return f"already in {target} mode"
        if target == "agent":
            request = ApprovalRequest(
                call_id="",
                kind="mode",
                summary="Switch to agent mode?",
                command=str(reason or "").strip(),
            )
            if not self._permitted(request, approved):
                return DENIED
        self.mode = target
        if self.on_mode_change is not None:
            self.on_mode_change(target)
        return f"switched to {target} mode"

    def _save_guideline(self, text: str, *, approved: bool | None) -> str:
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            return "error: empty guideline"
        request = ApprovalRequest(
            call_id="",
            kind="guideline",
            summary="Save this workspace guideline?",
            command=cleaned,
        )
        if not self._permitted(request, approved):
            return DENIED
        problem = append_guideline(self.workspace, cleaned)
        if problem:
            return problem
        return "saved guideline"

    def _read_skill(self, name: str) -> str:
        skill = find_skill(self.workspace, str(name or ""))
        if skill is None:
            return f"error: unknown skill '{name}'"
        body = skill.body.strip() or "(empty skill)"
        return cap_output(body)

    def _call_mcp(self, name: str, args: dict, *, approved: bool | None) -> str:
        self.sink.step(format_call(name, args))
        preview = self.preview_mcp("", name, args)
        if isinstance(preview, str):
            result = preview
        elif not self._permitted(preview, approved):
            result = DENIED
        else:
            info = self.mcp_tools[name]
            result = self.mcp_client.call(info.server, info.tool, args)
        self.sink.step(f"<- {summarize_result(name, result)}")
        return result

    def _permitted(self, request: ApprovalRequest, approved: bool | None) -> bool:
        if approved is False:
            return False
        if approved is True:
            return True
        from codie.permissions import rule_for_request

        spec = rule_for_request(request)
        if spec is not None:
            tool, match, _label = spec
            if PermissionSet.load(self.workspace).allows(tool, match):
                return True
        return bool(self.approver.ask(request))

    def _walk_files(self, start: Path):
        if start.is_file():
            if self._safe_file(start):
                yield start
            return
        for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
            kept: list[str] = []
            for name in dirnames:
                candidate = Path(dirpath) / name
                if is_secret_name(name) or self._escapes(candidate):
                    continue
                kept.append(name)
            dirnames[:] = kept
            for name in filenames:
                candidate = Path(dirpath) / name
                if self._safe_file(candidate):
                    yield candidate

    def _safe_file(self, path: Path) -> bool:
        if is_secret_name(path.name):
            return False
        try:
            relative = path.resolve().relative_to(self.workspace)
        except (OSError, ValueError):
            return False
        return not path_is_secret(relative)

    def _escapes(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.workspace)
        except (OSError, ValueError):
            return True
        return False

    def _hidden_entry(self, entry: Path) -> bool:
        if is_secret_name(entry.name):
            return True
        try:
            resolved = entry.resolve()
            relative = resolved.relative_to(self.workspace)
        except (OSError, ValueError):
            return True
        return path_is_secret(relative)

    def _mcp_langchain_tools(self):
        from langchain_core.tools import StructuredTool

        tools = []
        for info in self.mcp_tools.values():
            description = f"MCP tool from {info.server}. {info.description}".strip()
            tools.append(
                StructuredTool.from_function(
                    func=_mcp_stub,
                    name=info.exposed_name,
                    description=description,
                    args_schema=_mcp_model(info),
                )
            )
        return tools


def _mcp_stub(**_kwargs) -> str:
    return "error: tool execution goes through dispatch"


def _mcp_model(info: McpToolInfo):
    properties = info.input_schema.get("properties") if isinstance(info.input_schema, dict) else None
    required = set()
    if isinstance(info.input_schema, dict):
        required = set(info.input_schema.get("required") or [])
    fields = {}
    if isinstance(properties, dict):
        for key, spec in properties.items():
            if not isinstance(spec, dict):
                spec = {}
            typ = _json_type(spec.get("type"))
            description = str(spec.get("description") or "")
            if key in required:
                fields[key] = (typ, Field(description=description))
            else:
                fields[key] = (typ | None, Field(default=None, description=description))
    return create_model(
        f"Mcp_{info.exposed_name}",
        __config__=ConfigDict(extra="allow"),
        **fields,
    )


def _json_type(name: object):
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(str(name), str)


def _line_args(args: dict) -> tuple[int, int] | tuple[str, int]:
    try:
        start = int(args.get("start_line") or 0)
        end = int(args.get("end_line") or 0)
    except (TypeError, ValueError):
        return "error: line numbers must be integers", 0
    return start, end


def _slice_lines(text: str, start_line: int, end_line: int) -> str:
    try:
        start = int(start_line)
        end = int(end_line)
    except (TypeError, ValueError):
        return "error: line numbers must be integers"
    if start < 0 or end < 0:
        return "error: line numbers must be >= 0"
    lines = text.splitlines()
    if start == 0:
        start = 1
    if end == 0 or end > len(lines):
        end = len(lines)
    if not lines or start > len(lines) or start > end:
        return "error: line range is outside the file"
    chosen = [f"{number}|{lines[number - 1]}" for number in range(start, end + 1)]
    return cap_output("\n".join(chosen))


def _short(value: object) -> str:
    text = str(value)
    if len(text) > 80:
        return text[:77] + "..."
    return text


def _secret_in(path: Path) -> str:
    for part in path.parts:
        if is_secret_name(part):
            return part
    return path.name


def _command_head(argv: list[str], code: int, output: str) -> str:
    program = Path(argv[0]).name
    if program == "pytest":
        failed = _PYTEST_FAILED.search(output)
        passed = _PYTEST_PASSED.search(output)
        parts: list[str] = []
        if failed:
            parts.append(f"{failed.group(1)} failed")
        if passed:
            parts.append(f"{passed.group(1)} passed")
        detail = ", ".join(parts) if parts else "finished"
        return f"exit code {code} · {detail}"
    first = next((line.strip() for line in output.splitlines() if line.strip()), "no output")
    return f"exit code {code} · {first[:120]}"
