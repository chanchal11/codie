"""Workspace always-allow rules. Guards still run; a rule only skips the prompt."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from codie.approvals import ApprovalRequest
from codie.guards import command_program, is_secret_name, path_is_secret
from codie.store import load_permissions_data, save_permissions_data


@dataclass(frozen=True)
class Rule:
    id: str
    tool: str
    match: str
    label: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "tool": self.tool, "match": self.match, "label": self.label}


class PermissionSet:
    def __init__(self, rules: list[Rule], blocked: list[str] | None = None) -> None:
        self.rules = list(rules)
        self.blocked = list(blocked or [])

    @classmethod
    def load(cls, workspace: Path) -> PermissionSet:
        data = load_permissions_data(workspace)
        rules: list[Rule] = []
        for raw in data["rules"]:
            tool = str(raw.get("tool", "")).strip()
            match = str(raw.get("match", "")).strip()
            if not tool or not match:
                continue
            rules.append(
                Rule(
                    id=str(raw.get("id") or uuid.uuid4().hex),
                    tool=tool,
                    match=match,
                    label=str(raw.get("label") or f"{tool} {match}"),
                )
            )
        blocked: list[str] = []
        for name in data["blocked"]:
            if command_program(name) == name and name not in blocked:
                blocked.append(name)
        return cls(rules, blocked)

    def allows(self, tool: str, match: str) -> bool:
        if _match_is_blocked(tool, match):
            return False
        return any(rule.tool == tool and rule.match == match for rule in self.rules)

    def add(self, workspace: Path, tool: str, match: str, label: str) -> Rule | str:
        problem = _validate(tool, match)
        if problem:
            return problem
        if self.allows(tool, match):
            existing = next(rule for rule in self.rules if rule.tool == tool and rule.match == match)
            return existing
        rule = Rule(id=uuid.uuid4().hex, tool=tool, match=match, label=label)
        self.rules.append(rule)
        self._save(workspace)
        return rule

    def revoke(self, workspace: Path, rule_id: str) -> bool:
        kept = [rule for rule in self.rules if rule.id != rule_id]
        if len(kept) == len(self.rules):
            return False
        self.rules = kept
        self._save(workspace)
        return True

    def is_blocked(self, program: str) -> bool:
        return program in self.blocked

    def block(self, workspace: Path, program: str) -> str:
        """Remember a bare program name. Returns the name, or an error string."""
        if command_program(program) != program:
            return "error: only a bare program name can be blocked"
        if program not in self.blocked:
            self.blocked.append(program)
            self._save(workspace)
        return program

    def unblock(self, workspace: Path, program: str) -> bool:
        if program not in self.blocked:
            return False
        self.blocked = [name for name in self.blocked if name != program]
        self._save(workspace)
        return True

    def _save(self, workspace: Path) -> None:
        save_permissions_data(
            workspace,
            {
                "rules": [item.to_dict() for item in self.rules],
                "blocked": list(self.blocked),
            },
        )


def normalize_rel(path: str) -> str | None:
    """Workspace-relative posix path, or None when it must not be remembered."""
    if path is None:
        return None
    text = str(path).strip()
    if text == "":
        return None
    raw = Path(text)
    if raw.is_absolute() or ".." in raw.parts:
        return None
    if path_is_secret(raw) or is_secret_name(raw.name):
        return None
    return raw.as_posix()


def rule_for_request(request: ApprovalRequest) -> tuple[str, str, str] | None:
    """Narrow always-allow rule for a prompt, or None when the action cannot be remembered."""
    if request.kind == "write" and request.path:
        match = normalize_rel(request.path)
        if not match:
            return None
        return ("write_file", match, f"write {match}")
    if request.kind == "edit" and request.path:
        match = normalize_rel(request.path)
        if not match:
            return None
        return ("edit_file", match, f"edit {match}")
    if request.kind == "pytest":
        return ("run_command", "pytest", "pytest")
    if request.kind == "mcp" and request.path and request.summary.startswith("mcp_"):
        return (request.summary, request.path, request.summary)
    return None


def remember_request(workspace: Path, request: ApprovalRequest) -> str | None:
    """Store the narrow rule for this prompt. Returns the label, or None if it cannot be stored."""
    spec = rule_for_request(request)
    if spec is None:
        return None
    tool, match, label = spec
    saved = PermissionSet.load(workspace).add(workspace, tool, match, label)
    if isinstance(saved, str):
        return None
    return saved.label


def _match_is_blocked(tool: str, match: str) -> bool:
    if tool in {"write_file", "edit_file"}:
        return normalize_rel(match) != match
    if tool == "run_command":
        return match != "pytest"
    if tool.startswith("mcp_"):
        return "/" not in match or match.startswith("/") or ".." in match
    return True


def _validate(tool: str, match: str) -> str | None:
    if _match_is_blocked(tool, match):
        return "error: that action cannot be always-allowed"
    return None
