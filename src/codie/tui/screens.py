"""Modals for approvals, mode, search, sessions, skills, rules, and MCP."""

from __future__ import annotations

import shlex

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, OptionList, RichLog, Static

from codie.approvals import ApprovalRequest
from codie.permissions import rule_for_request
from codie.search import AVAILABLE_PROVIDERS, LATER_PROVIDERS

_DIALOG = """
ApprovalModal, PickerModal, SearchModal, McpModal, McpFormModal, SkillsModal {
    align: center middle;
}
#dialog {
    width: 88%;
    max-width: 100;
    height: auto;
    max-height: 86%;
    background: $surface;
    border: solid $accent;
    padding: 1 2;
}
#dialog-body {
    height: auto;
    max-height: 24;
    margin: 1 0;
}
#buttons {
    layout: horizontal;
    height: auto;
    align: right middle;
    margin-top: 1;
}
#buttons Button {
    margin-left: 1;
}
#form-error {
    color: $error;
    height: auto;
}
"""


def render_request(request: ApprovalRequest) -> Text:
    text = Text()
    if request.kind in {"write", "edit"} and request.diff:
        for line in request.diff.splitlines():
            style = ""
            if line.startswith("+") and not line.startswith("+++"):
                style = "green"
            elif line.startswith("-") and not line.startswith("---"):
                style = "red"
            elif line.startswith("@@"):
                style = "cyan"
            text.append(line + "\n", style=style)
        return text
    if request.command:
        text.append(request.command + "\n", style="yellow")
    return text


class ApprovalModal(ModalScreen[str]):
    BINDINGS = [
        Binding("y", "allow", "Allow"),
        Binding("a", "always", "Always"),
        Binding("n", "deny", "Deny"),
        Binding("escape", "deny", "Deny"),
    ]
    DEFAULT_CSS = _DIALOG + """
    ApprovalModal > #dialog {
        height: 80%;
        border: solid $warning;
    }
    ApprovalModal #diff {
        height: 1fr;
        margin: 1 0;
    }
    """

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        titles = {
            "write": "Allow this write?",
            "edit": "Allow this edit?",
            "pytest": "Allow pytest?",
            "mode": "Switch to agent mode?",
            "guideline": "Remember this guideline?",
            "mcp": "Allow this MCP tool?",
        }
        spec = rule_for_request(self.request)
        with Vertical(id="dialog"):
            yield Static(titles.get(self.request.kind, "Allow this action?"), id="ask")
            if spec is not None:
                yield Static(f"Always allow will remember: {spec[2]}")
            yield Static("y allow once    a always    n deny")
            yield RichLog(id="diff", wrap=True, markup=False, min_width=20)
            with Horizontal(id="buttons"):
                yield Button("Deny", id="deny", variant="error")
                yield Button("Allow once", id="allow", variant="success")
                if spec is not None:
                    yield Button("Always allow", id="always", variant="warning")

    def on_mount(self) -> None:
        self.query_one("#diff", RichLog).write(render_request(self.request))
        self.query_one("#deny", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id
        if button_id == "always":
            self.action_always()
            return
        if button_id == "allow":
            self.action_allow()
            return
        self.action_deny()

    def action_allow(self) -> None:
        self.dismiss("allow")

    def action_always(self) -> None:
        if rule_for_request(self.request) is None:
            self.dismiss("allow")
            return
        self.dismiss("always")

    def action_deny(self) -> None:
        self.dismiss("deny")


class PickerModal(ModalScreen[str | None]):
    """Choose one id. Enter confirms. Close dismisses nothing."""

    BINDINGS = [Binding("escape", "cancel", "Close")]
    DEFAULT_CSS = _DIALOG

    def __init__(self, title: str, choices: list[tuple[str, str]], confirm: str | None = None) -> None:
        super().__init__()
        self.title_text = title
        self.choices = choices
        self.confirm = confirm

    def compose(self) -> ComposeResult:
        labels = [label for _choice_id, label in self.choices]
        with Vertical(id="dialog"):
            yield Static(self.title_text)
            yield OptionList(*labels, id="choices")
            with Horizontal(id="buttons"):
                yield Button("Close", id="cancel")
                if self.confirm:
                    yield Button(self.confirm, id="confirm", variant="warning")

    def on_mount(self) -> None:
        if self.choices:
            self.query_one("#choices", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if self.confirm:
            return
        self.dismiss(self.choices[event.option_index][0])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "confirm":
            picked = self._highlighted()
            self.dismiss(picked)
            return
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _highlighted(self) -> str | None:
        if not self.choices:
            return None
        widget = self.query_one("#choices", OptionList)
        index = widget.highlighted
        if index is None or index < 0 or index >= len(self.choices):
            return self.choices[0][0]
        return self.choices[index][0]


class SearchModal(ModalScreen[tuple[str, str] | None]):
    BINDINGS = [Binding("escape", "cancel", "Close")]
    DEFAULT_CSS = _DIALOG + """
    SearchModal Input {
        margin-top: 1;
    }
    """

    def __init__(self, current: str, searxng_url: str) -> None:
        super().__init__()
        self.current = current
        self.searxng_url = searxng_url

    def compose(self) -> ComposeResult:
        labels = []
        for name in AVAILABLE_PROVIDERS:
            mark = "  current" if name == self.current else ""
            labels.append(f"{name}{mark}")
        later = ", ".join(LATER_PROVIDERS)
        with Vertical(id="dialog"):
            yield Static("Web search")
            yield Static(f"{later} can be added later.")
            yield OptionList(*labels, id="choices")
            yield Static("SearXNG URL")
            yield Input(value=self.searxng_url, placeholder="http://127.0.0.1:8080", id="searxng")
            with Horizontal(id="buttons"):
                yield Button("Close", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#choices", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        provider = AVAILABLE_PROVIDERS[event.option_index]
        url = self.query_one("#searxng", Input).value.strip()
        self.dismiss((provider, url))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SkillsModal(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "Close")]
    DEFAULT_CSS = _DIALOG + """
    SkillsModal #dialog {
        height: 80%;
    }
    SkillsModal #skill-body {
        height: 1fr;
        margin-top: 1;
    }
    """

    def __init__(self, rows: list[tuple[str, str, str]]) -> None:
        super().__init__()
        self.rows = rows

    def compose(self) -> ComposeResult:
        labels = [f"{name}  {description}" for name, description, _body in self.rows] or ["(no skills)"]
        with Vertical(id="dialog"):
            yield Static("Skills")
            yield Static("Personal skills live in ~/.codie/skills/<name>/SKILL.md")
            yield OptionList(*labels, id="choices")
            yield RichLog(id="skill-body", wrap=True, markup=False, min_width=20)
            with Horizontal(id="buttons"):
                yield Button("Close", id="cancel")

    def on_mount(self) -> None:
        if self.rows:
            self.query_one("#choices", OptionList).focus()
            self._show(0)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self._show(event.option_index)

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._show(event.option_index)

    def _show(self, index: int) -> None:
        if not self.rows or index < 0 or index >= len(self.rows):
            return
        _name, description, body = self.rows[index]
        log = self.query_one("#skill-body", RichLog)
        log.clear()
        log.write(description or "No description.")
        log.write("")
        log.write(body or "(empty skill)")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


class McpFormModal(ModalScreen[dict | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    DEFAULT_CSS = _DIALOG + """
    McpFormModal Input {
        margin-bottom: 1;
    }
    """

    def __init__(self, initial: dict | None = None, locked_name: str = "") -> None:
        super().__init__()
        self.initial = initial or {}
        self.locked_name = locked_name

    def compose(self) -> ComposeResult:
        env = self.initial.get("env") if isinstance(self.initial.get("env"), dict) else {}
        env_text = "\n".join(f"{key}={value}" for key, value in env.items())
        args = self.initial.get("args") if isinstance(self.initial.get("args"), list) else []
        with Vertical(id="dialog"):
            yield Static("Edit MCP server" if self.locked_name else "Add MCP server")
            yield Static("Name")
            yield Input(value=self.locked_name or str(self.initial.get("name") or ""), id="name")
            yield Static("Command")
            yield Input(value=str(self.initial.get("command") or ""), id="command")
            yield Static("Arguments")
            yield Input(value=" ".join(str(item) for item in args), id="args")
            yield Static("Environment (KEY=VALUE, one per line)")
            yield Input(value=env_text.replace("\n", ", "), id="env")
            yield Checkbox("Enabled", value=self.initial.get("enabled", True) is not False, id="enabled")
            yield Static("", id="form-error")
            with Horizontal(id="buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Save", id="save", variant="success")

    def on_mount(self) -> None:
        name = self.query_one("#name", Input)
        if self.locked_name:
            name.disabled = True
            self.query_one("#command", Input).focus()
        else:
            name.focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "save":
            self._save()
            return
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _save(self) -> None:
        name = self.query_one("#name", Input).value.strip()
        command = self.query_one("#command", Input).value.strip()
        raw_args = self.query_one("#args", Input).value.strip()
        raw_env = self.query_one("#env", Input).value.strip()
        error = self.query_one("#form-error", Static)
        if not name or not all(char.isalnum() or char in {"-", "_"} for char in name):
            error.update("Name must be letters, numbers, dash, or underscore.")
            return
        if not command or any(char in command for char in "\n\r"):
            error.update("Command is required.")
            return
        try:
            args = shlex.split(raw_args, posix=True) if raw_args else []
        except ValueError as exc:
            error.update(f"Could not parse arguments ({exc}).")
            return
        env: dict[str, str] = {}
        if raw_env:
            pieces = [piece.strip() for piece in raw_env.replace("\n", ",").split(",")]
            for piece in pieces:
                if not piece:
                    continue
                if "=" not in piece:
                    error.update("Environment entries must look like KEY=VALUE.")
                    return
                key, value = piece.split("=", 1)
                key = key.strip()
                if not key:
                    error.update("Environment entries must look like KEY=VALUE.")
                    return
                env[key] = value.strip()
        self.dismiss(
            {
                "name": name,
                "command": command,
                "args": args,
                "env": env,
                "enabled": self.query_one("#enabled", Checkbox).value,
            }
        )


class McpModal(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "Close")]
    DEFAULT_CSS = _DIALOG + """
    McpModal #dialog {
        height: 80%;
    }
    """

    def __init__(self, servers: dict, errors: list[str], on_save) -> None:
        super().__init__()
        self.servers = dict(servers)
        self.errors = list(errors)
        self.on_save = on_save

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("MCP servers")
            yield Static("Only you can change this list. The agent cannot edit it.", id="mcp-note")
            yield Static("", id="mcp-errors")
            yield OptionList(id="servers")
            with Horizontal(id="buttons"):
                yield Button("Close", id="close")
                yield Button("Remove", id="remove", variant="error")
                yield Button("Enable", id="toggle")
                yield Button("Edit", id="edit")
                yield Button("Add", id="add", variant="success")

    def on_mount(self) -> None:
        self._fill()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id
        if button_id == "add":
            self.app.push_screen(McpFormModal(), self._on_form)
            return
        if button_id == "edit":
            current = self._current()
            if current is None:
                return
            self.app.push_screen(
                McpFormModal(initial=self.servers[current], locked_name=current),
                self._on_form,
            )
            return
        if button_id == "remove":
            current = self._current()
            if current is not None:
                self.servers.pop(current, None)
                self._commit()
            return
        if button_id == "toggle":
            current = self._current()
            if current is not None:
                cfg = dict(self.servers[current])
                cfg["enabled"] = cfg.get("enabled", True) is False
                self.servers[current] = cfg
                self._commit()
            return
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)

    def _on_form(self, result: dict | None) -> None:
        if not result:
            return
        name = result.pop("name")
        self.servers[name] = result
        self._commit()

    def _commit(self) -> None:
        errors = self.on_save(self.servers)
        self.errors = list(errors or [])
        self._fill()

    def _fill(self) -> None:
        widget = self.query_one("#servers", OptionList)
        widget.clear_options()
        for name in sorted(self.servers):
            cfg = self.servers[name]
            state = "on" if cfg.get("enabled", True) is not False else "off"
            command = str(cfg.get("command") or "")
            widget.add_option(f"{name}  {state}  {command}")
        note = "; ".join(self.errors)
        self.query_one("#mcp-errors", Static).update(note)

    def _current(self) -> str | None:
        names = sorted(self.servers)
        if not names:
            return None
        widget = self.query_one("#servers", OptionList)
        index = widget.highlighted
        if index is None or index < 0 or index >= len(names):
            return names[0]
        return names[index]
