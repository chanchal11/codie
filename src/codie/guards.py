"""Path, secret, and command guards.

Failures are plain strings that start with ``error:``. They are returned to the
model instead of being raised, and they never contain ANSI colour codes.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

OUTPUT_LIMIT = 8_000
COMMAND_TIMEOUT_SECONDS = 30
MAX_ROUNDS_DEFAULT = 12

_SECRET_EXACT = {".ssh", ".aws", ".git", "id_rsa"}
_SHELL_META = set("|;&><`$")
_ALLOWED = {"ls", "cat", "head", "tail", "wc", "grep", "find", "pwd"}
_FIND_BLOCKED = {"-exec", "-execdir", "-ok", "-okdir", "-delete", "--delete"}
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

GREP_EXCLUDES = [
    "--exclude=.env",
    "--exclude=.env.*",
    "--exclude=id_rsa",
    "--exclude=id_rsa.*",
    "--exclude=*.pem",
    "--exclude=*.PEM",
    "--exclude=*.key",
    "--exclude=*.KEY",
    "--exclude-dir=.ssh",
    "--exclude-dir=.aws",
    "--exclude-dir=.git",
]


def is_secret_name(name: str) -> bool:
    """True when a single path component should never be read or listed."""
    if name in {".", "..", ""}:
        return False
    if name in _SECRET_EXACT or name.startswith("id_rsa"):
        return True
    if name == ".env" or name.startswith(".env."):
        return True
    lowered = name.lower()
    return lowered.endswith(".pem") or lowered.endswith(".key")


def path_is_secret(path: Path) -> bool:
    return any(is_secret_name(part) for part in path.parts)


def secret_label(path: Path) -> str:
    for part in path.parts:
        if is_secret_name(part):
            return part
    return path.name


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def cap_output(text: str, limit: int = OUTPUT_LIMIT) -> str:
    """Plain text, truncated, with no colour codes."""
    cleaned = strip_ansi(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "\n...[truncated]"


def check_path(workspace: Path, user_path: str) -> str | None:
    """Return an error string when ``user_path`` must not be touched."""
    if user_path is None:
        return "error: empty path"
    raw_text = str(user_path).strip()
    if raw_text == "":
        return "error: empty path"
    raw = Path(raw_text)
    if raw.is_absolute():
        return "error: absolute paths are not allowed"
    if path_is_secret(raw):
        return f"error: '{secret_label(raw)}' looks like a secret"
    root = workspace.resolve()
    try:
        candidate = (root / raw).resolve()
    except OSError as exc:
        return f"error: cannot resolve path ({exc})"
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return "error: path escapes the workspace"
    if path_is_secret(relative):
        return f"error: '{secret_label(relative)}' looks like a secret"
    return None


def resolve_inside(workspace: Path, user_path: str) -> Path | str:
    """Resolved path inside the workspace, or an error string."""
    error = check_path(workspace, user_path)
    if error:
        return error
    return (workspace.resolve() / user_path).resolve()


@dataclass(frozen=True)
class CommandPlan:
    argv: list[str]
    needs_approval: bool
    scrub: str
    """How to scrub stdout: ``none``, ``listing``, or ``grep``."""


def plan_command(workspace: Path, command: str) -> CommandPlan | str:
    """Validate a command string. Returns a plan, or an error string.

    The command is never passed to a shell. Metacharacters are rejected before
    ``shlex`` splits the string into an argv list.
    """
    if command is None or str(command).strip() == "":
        return "error: empty command"
    text = str(command)
    if any(char in text for char in _SHELL_META) or "\n" in text or "\r" in text or "\x00" in text:
        return "error: shell metacharacters are not allowed"
    try:
        argv = shlex.split(text, posix=True)
    except ValueError as exc:
        return f"error: could not parse command ({exc})"
    if not argv:
        return "error: empty command"
    program = argv[0]
    if "/" in program or "\\" in program or program.startswith("."):
        return "error: command must be a bare name on the allow-list"
    if program == "find" and any(arg in _FIND_BLOCKED for arg in argv[1:]):
        return "error: find -exec, -ok, and -delete are not allowed"
    for arg in argv[1:]:
        if arg.startswith("-") and arg != "-":
            continue
        problem = _check_command_arg(workspace, arg)
        if problem:
            return problem
    if program == "pytest":
        return CommandPlan(argv=argv, needs_approval=True, scrub="none")
    if program not in _ALLOWED:
        return f"error: '{program}' is not on the allow-list"
    scrub = "none"
    if program == "grep":
        argv = [argv[0], *GREP_EXCLUDES, *argv[1:]]
        scrub = "grep"
    elif program in {"ls", "find"}:
        scrub = "listing"
    return CommandPlan(argv=argv, needs_approval=False, scrub=scrub)


def _check_command_arg(workspace: Path, arg: str) -> str | None:
    path = Path(arg)
    if path.is_absolute():
        return "error: absolute paths are not allowed"
    if path_is_secret(path):
        return f"error: '{secret_label(path)}' looks like a secret"
    looks_like_path = ".." in path.parts or "/" in arg or "\\" in arg
    if looks_like_path or (workspace / arg).exists() or (workspace / arg).is_symlink():
        return check_path(workspace, arg)
    return None


def scrub_listing(text: str) -> str:
    """Drop ls/find lines that name a secret."""
    kept: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        if parts and _token_is_secret(parts[-1]):
            continue
        if _token_is_secret(line.strip()):
            continue
        kept.append(line)
    return "\n".join(kept)


def scrub_grep(text: str) -> str:
    """Drop grep hits whose file path is a secret. Leave ordinary lines alone."""
    kept: list[str] = []
    for line in text.splitlines():
        filename = line.split(":", 1)[0].strip()
        if _token_is_secret(filename):
            continue
        kept.append(line)
    return "\n".join(kept)


def _token_is_secret(token: str) -> bool:
    if not token:
        return False
    return path_is_secret(Path(token))
