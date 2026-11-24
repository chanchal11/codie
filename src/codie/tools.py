"""Workspace tools. Every path and command goes through the guards first."""

from __future__ import annotations

import re
import subprocess
from difflib import unified_diff
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

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

_PYTEST_FAILED = re.compile(r"(\d+)\s+failed")
_PYTEST_PASSED = re.compile(r"(\d+)\s+passed")


class ListFilesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="Directory relative to the workspace. Use '.' for the root.")


class ReadFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="File relative to the workspace.")


class FindFilesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pattern: str = Field(description="Glob relative to the workspace, such as '**/*.py'.")


class WriteFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="File relative to the workspace.")
    content: str = Field(description="Full new file contents.")


class RunCommandArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: str = Field(description="One allow-listed command, or pytest. No shell syntax.")


class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="Search query for recent public information.")


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
    if name in {"list_files", "read_file"}:
        return f"-> {name}({quote_arg(str(args.get('path', '')))})"
    rendered = ", ".join(f"{key}={quote_arg(str(val))}" for key, val in args.items())
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
    if name == "write_file":
        return first[:200]
    if name == "web_search":
        count = result.count("\n\n") + (1 if result.strip() else 0)
        return f"{count} results"
    return first[:200]


class ToolContext:
    def __init__(self, workspace: Path, approver: Approver, sink: StepSink | None = None) -> None:
        self.workspace = Path(workspace).resolve()
        self.approver = approver
        self.sink = sink or NullSink()

    def list_files(self, path: str) -> str:
        self.sink.step(format_call("list_files", {"path": path}))
        result = self._list_files(path)
        self.sink.step(f"<- {summarize_result('list_files', result)}")
        return result

    def read_file(self, path: str) -> str:
        self.sink.step(format_call("read_file", {"path": path}))
        result = self._read_file(path)
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

    def run_command(self, command: str, *, approved: bool | None = None) -> str:
        self.sink.step(format_call("run_command", {"command": command}))
        result = self._run_command(command, approved=approved)
        self.sink.step(f"<- {summarize_result('run_command', result)}")
        return result

    def web_search(self, query: str) -> str:
        self.sink.step(format_call("web_search", {"query": query}))
        result = self._web_search(query)
        self.sink.step(f"<- {summarize_result('web_search', result)}")
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

    def preview_pytest(self, call_id: str, command: str) -> ApprovalRequest | str:
        plan = plan_command(self.workspace, command)
        if isinstance(plan, str):
            return plan
        if not plan.needs_approval:
            return "error: command does not need approval"
        return ApprovalRequest(
            call_id=call_id,
            kind="pytest",
            summary=f"run_command({quote_arg(command)})",
            command=command,
        )

    def dispatch(self, name: str, args: dict, *, approved: bool | None = None) -> str:
        try:
            if name == "list_files":
                return self.list_files(args["path"])
            if name == "read_file":
                return self.read_file(args["path"])
            if name == "find_files":
                return self.find_files(args["pattern"])
            if name == "write_file":
                return self.write_file(args["path"], args["content"], approved=approved)
            if name == "run_command":
                return self.run_command(args["command"], approved=approved)
            if name == "web_search":
                return self.web_search(args["query"])
        except KeyError as exc:
            missing = str(exc).strip("'")
            return f"error: missing field {missing}"
        return f"error: unknown tool '{name}'"

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

    def _read_file(self, path: str) -> str:
        resolved = resolve_inside(self.workspace, path)
        if isinstance(resolved, str):
            return resolved
        if not resolved.is_file():
            return "error: path is not a file"
        try:
            with resolved.open("r", encoding="utf-8") as handle:
                data = handle.read(8_001)
        except UnicodeDecodeError:
            return "error: file is not utf-8 text"
        except OSError as exc:
            return f"error: cannot read file ({exc})"
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
        if approved is False:
            return DENIED
        if approved is None and not self.approver.ask(
            ApprovalRequest(
                call_id="",
                kind=preview.kind,
                summary=preview.summary,
                diff=preview.diff,
                path=preview.path,
            )
        ):
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

    def _run_command(self, command: str, *, approved: bool | None) -> str:
        plan = plan_command(self.workspace, command)
        if isinstance(plan, str):
            return plan
        if plan.needs_approval:
            if approved is False:
                return DENIED
            if approved is None:
                request = ApprovalRequest(
                    call_id="",
                    kind="pytest",
                    summary=f"run_command({quote_arg(command)})",
                    command=command,
                )
                if not self.approver.ask(request):
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

    def _web_search(self, query: str) -> str:
        if query is None or str(query).strip() == "":
            return "error: empty query"
        rows = None
        errors: list[str] = []
        for backend in ("auto", "brave", "bing", "mojeek"):
            try:
                from ddgs import DDGS

                rows = DDGS().text(str(query), max_results=5, backend=backend)
            except Exception as exc:
                errors.append(f"{backend}: {exc}")
                rows = None
                continue
            if rows:
                break
        if not rows:
            detail = "; ".join(errors) or "no results"
            return f"error: web search failed ({detail})"
        blocks: list[str] = []
        for row in rows:
            title = _row_field(row, "title")
            href = _row_field(row, "href")
            body = _row_field(row, "body")
            blocks.append(f"{title}\n{href}\n{body}".strip())
        return cap_output("\n\n".join(blocks))

    def _hidden_entry(self, entry: Path) -> bool:
        if is_secret_name(entry.name):
            return True
        try:
            resolved = entry.resolve()
            relative = resolved.relative_to(self.workspace)
        except (OSError, ValueError):
            return True
        return path_is_secret(relative)

    def langchain_tools(self):
        """Schemas for the model. Execution goes through ``dispatch``."""
        from langchain_core.tools import StructuredTool

        return [
            StructuredTool.from_function(
                func=self.list_files,
                name="list_files",
                description="List one directory inside the workspace. Secret names are omitted.",
                args_schema=ListFilesArgs,
            ),
            StructuredTool.from_function(
                func=self.read_file,
                name="read_file",
                description="Read a UTF-8 text file inside the workspace.",
                args_schema=ReadFileArgs,
            ),
            StructuredTool.from_function(
                func=self.find_files,
                name="find_files",
                description="Glob for paths inside the workspace. Example: **/*.py.",
                args_schema=FindFilesArgs,
            ),
            StructuredTool.from_function(
                func=self.write_file,
                name="write_file",
                description="Write the full new contents of a file. A human approves the diff first.",
                args_schema=WriteFileArgs,
            ),
            StructuredTool.from_function(
                func=self.run_command,
                name="run_command",
                description=(
                    "Run one command with no shell. Free: ls cat head tail wc grep find pwd. "
                    "pytest needs human approval. Everything else is refused."
                ),
                args_schema=RunCommandArgs,
            ),
            StructuredTool.from_function(
                func=self.web_search,
                name="web_search",
                description="Search the public web. Use this for versions and events after your training cutoff.",
                args_schema=WebSearchArgs,
            ),
        ]


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


def _row_field(row: object, key: str) -> str:
    if isinstance(row, dict):
        return strip_ansi(str(row.get(key, "")))
    return strip_ansi(str(getattr(row, key, "")))
