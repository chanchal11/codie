"""Workspace always-allow rules. Guards still run; a rule only skips the prompt."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from codie.approvals import ApprovalRequest
from codie.guards import is_secret_name, path_is_secret
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
    def __init__(self, rules: list[Rule]) -> None:
        self.rules = list(rules)

    @classmethod
    def load(cls, workspace: Path) -> PermissionSet:
        rules: list[Rule] = []
        for raw in load_permissions_data(workspace)["rules"]:
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
        return cls(rules)

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
        save_permissions_data(workspace, {"rules": [item.to_dict() for item in self.rules]})
        return rule

    def revoke(self, workspace: Path, rule_id: str) -> bool:
        kept = [rule for rule in self.rules if rule.id != rule_id]
        if len(kept) == len(self.rules):
            return False
        self.rules = kept
        save_permissions_data(workspace, {"rules": [item.to_dict() for item in self.rules]})
        return True


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
