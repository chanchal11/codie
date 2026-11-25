"""Human approval and live step callbacks.

Approvals are decided by the caller (the CLI or the TUI), never by the model.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

DENIED = "error: user denied this action"


@dataclass(frozen=True)
class ApprovalRequest:
    call_id: str
    kind: str
    summary: str
    diff: str = ""
    command: str = ""
    path: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> ApprovalRequest:
        return cls(
            call_id=str(data.get("call_id", "")),
            kind=str(data.get("kind", "")),
            summary=str(data.get("summary", "")),
            diff=str(data.get("diff", "")),
            command=str(data.get("command", "")),
            path=str(data.get("path", "")),
        )


class Approver(Protocol):
    def ask(self, request: ApprovalRequest) -> bool:
        """Return True when the human allows the action."""


class StepSink(Protocol):
    def step(self, line: str) -> None:
        """One live step line, already formatted for a human."""

    def token(self, text: str) -> None:
        """A piece of the model's streamed answer."""

    def drop_tokens(self) -> None:
        """Forget streamed tokens when the model asked for tools instead of answering."""


class AllowAll:
    def ask(self, request: ApprovalRequest) -> bool:
        return True


class DenyAll:
    def ask(self, request: ApprovalRequest) -> bool:
        return False


class NullSink:
    def step(self, line: str) -> None:
        return None

    def token(self, text: str) -> None:
        return None

    def drop_tokens(self) -> None:
        return None


def colorize_diff(diff: str) -> str:
    """ANSI colours for a human terminal. Never send this string to the model."""
    painted: list[str] = []
    for line in diff.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body) :]
        if body.startswith("+") and not body.startswith("+++"):
            painted.append(f"\033[32m{body}\033[0m{ending}")
        elif body.startswith("-") and not body.startswith("---"):
            painted.append(f"\033[31m{body}\033[0m{ending}")
        elif body.startswith("@@"):
            painted.append(f"\033[36m{body}\033[0m{ending}")
        else:
            painted.append(line)
    return "".join(painted)


class CliApprover:
    """Ask on stdin. Colours are printed for the human only."""

    def __init__(self, input_fn=input, output_fn=print, workspace: Path | None = None) -> None:
        self._input = input_fn
        self._output = output_fn
        self.workspace = workspace

    def ask(self, request: ApprovalRequest) -> bool:
        if request.kind in {"write", "edit"} and request.diff:
            shown = colorize_diff(request.diff) if sys.stdout.isatty() else request.diff
            self._output(shown, end="" if shown.endswith("\n") else "\n")
        elif request.command:
            self._output(request.command)
        if request.kind == "command":
            self._output("b blocks this program for this workspace.")
            self._output(f"{request.summary}\nAllow? [y/N/b] ", end="")
        else:
            self._output("a always allows this action for this workspace.")
            self._output(f"{request.summary}\nAllow? [y/N] ", end="")
        try:
            answer = self._input()
        except EOFError:
            self._output()
            return False
        normalized = answer.strip().lower()
        if normalized in {"b", "block"} and request.kind == "command":
            from codie.guards import command_program

            program = command_program(request.command)
            if program and self.workspace is not None:
                from codie.permissions import PermissionSet

                PermissionSet.load(self.workspace).block(self.workspace, program)
            return False
        if normalized in {"a", "always"}:
            if self.workspace is not None:
                from codie.permissions import remember_request

                remember_request(self.workspace, request)
            return True
        return normalized in {"y", "yes"}


class CliSink:
    def __init__(self) -> None:
        self._mid_line = False
        self._color = sys.stdout.isatty()
        self.streamed = ""

    def token(self, text: str) -> None:
        self.streamed += text
        print(text, end="", flush=True)
        self._mid_line = not text.endswith("\n")

    def step(self, line: str) -> None:
        if self._mid_line:
            print()
            self._mid_line = False
        print(self._paint(line), flush=True)

    def drop_tokens(self) -> None:
        self.streamed = ""
        if self._mid_line:
            print()
            self._mid_line = False

    def finish(self) -> None:
        if self._mid_line:
            print()
            self._mid_line = False

    def _paint(self, line: str) -> str:
        if not self._color:
            return line
        if line.startswith("->"):
            return f"\033[36m{line}\033[0m"
        if "error:" in line or "denied" in line or "step limit" in line:
            return f"\033[31m{line}\033[0m"
        if line.startswith("<-"):
            return f"\033[32m{line}\033[0m"
        return line
