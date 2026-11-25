"""IDE-style terminal UI: chat, live steps, sessions, and approvals."""

from __future__ import annotations

import queue
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, OptionList, RichLog, Static

from codie.agent import CodingAgent, message_text, visible_text
from codie.approvals import ApprovalRequest
from codie.guards import MAX_ROUNDS_DEFAULT, command_program
from codie.permissions import PermissionSet, remember_request
from codie.providers import DEFAULT_OLLAMA_MODEL, bedrock_model_id, build_chat_model, list_ollama_models
from codie.search import active_provider, provider_label, searxng_base_url
from codie.sessions import SessionRecorder, list_sessions, load_session as read_session
from codie.skills import discover_skills, workspace_skills_dir
from codie.store import (
    codie_home,
    delete_guideline,
    guideline_lines,
    load_config,
    load_mcp,
    save_config,
    save_mcp,
)
from codie.tui.screens import (
    ApprovalModal,
    McpModal,
    PickerModal,
    SearchModal,
    SkillsModal,
)

HELP = (
    "Commands: /mode [plan|agent], /model, /provider ollama|bedrock, /theme [name], "
    "/search [ddgs|searxng|tavily|openai], "
    "/sessions, /mcp, /skills, /allow, /guidelines, /new, /exit"
)


class TuiApprover:
    """Block the agent thread until the modal returns Allow, Always, or Deny."""

    def __init__(self, app: CodieApp) -> None:
        self.app = app
        self.results: queue.Queue[bool] = queue.Queue()

    def ask(self, request: ApprovalRequest) -> bool:
        self.app.call_from_thread(self.app.request_approval, request)
        return self.results.get()


class TuiSink:
    def __init__(self, app: CodieApp) -> None:
        self.app = app

    def step(self, line: str) -> None:
        self.app.call_from_thread(self.app.add_step, line)

    def token(self, text: str) -> None:
        self.app.call_from_thread(self.app.add_token, text)

    def drop_tokens(self) -> None:
        self.app.call_from_thread(self.app.clear_live)


class CodieApp(App):
    """Chat in the centre, sessions on the left, tool steps on the right."""

    TITLE = "codie"
    CSS = """
    Screen {
        layout: vertical;
    }
    #status {
        height: 1;
        padding: 0 1;
        background: $boost;
        color: $text;
    }
    #chips {
        height: 3;
        padding: 0 1;
        background: $surface;
    }
    #chips Button {
        height: 3;
        min-width: 8;
        margin: 0 1 0 0;
        border: none;
        padding: 0 1;
    }
    #body {
        layout: horizontal;
        height: 1fr;
    }
    #sessions-panel {
        width: 28;
        height: 1fr;
        border: solid $primary;
    }
    #session-note {
        height: auto;
        max-height: 3;
        color: $text-muted;
        padding: 0 1;
    }
    #sessions {
        height: 1fr;
    }
    #chat-panel {
        width: 2fr;
        height: 1fr;
        border: solid $accent;
    }
    #steps-panel {
        width: 1fr;
        height: 1fr;
        border: solid $warning;
    }
    #chat, #steps {
        height: 1fr;
    }
    #live {
        height: auto;
        max-height: 8;
        padding: 0 1;
    }
    #prompt {
        dock: bottom;
        margin: 0 1 1 1;
    }
    """
    BINDINGS = [
        Binding("ctrl+b", "toggle_sessions", "Sessions"),
        Binding("ctrl+m", "models", "Model"),
        Binding("ctrl+p", "mode", "Mode"),
        Binding("ctrl+t", "theme", "Theme"),
        Binding("ctrl+s", "search", "Search"),
        Binding("ctrl+e", "mcp", "MCP"),
        Binding("ctrl+n", "new_chat", "New"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        workspace: Path,
        provider: str = "ollama",
        model_name: str | None = None,
        max_rounds: int = MAX_ROUNDS_DEFAULT,
        model=None,
        agent: CodingAgent | None = None,
    ) -> None:
        super().__init__()
        saved_theme = load_config().get("theme")
        if isinstance(saved_theme, str) and saved_theme in self.available_themes:
            self.theme = saved_theme
        self.workspace = Path(workspace).resolve()
        self.provider = provider
        self.model_name = model_name or _model_label(provider, model)
        self.max_rounds = max_rounds
        self._live = ""
        self._busy = False
        self._pending_request: ApprovalRequest | None = None
        self._session_rows = []
        self.transcript: list[str] = []
        self.step_lines: list[str] = []
        self.status_line = ""
        self.approver = TuiApprover(self)
        self.sink = TuiSink(self)
        self.recorder = SessionRecorder(self.workspace)
        self.recorder.provider = self.provider
        self.recorder.model = self.model_name
        self.recorder.search_provider = active_provider()
        if agent is None:
            if model is None:
                model = build_chat_model(provider, self.model_name)
            self.agent = CodingAgent(model, self.workspace, self.approver, self.sink, max_rounds)
            last_mode = load_config().get("last_mode")
            if last_mode in {"agent", "plan"}:
                self.agent.set_mode(last_mode)
        else:
            agent.approver = self.approver
            agent.sink = self.sink
            self.agent = agent
        self.agent.recorder = self.recorder

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="status")
        with Horizontal(id="chips"):
            yield Button("agent", id="chip-mode")
            yield Button("model", id="chip-model")
            yield Button("theme", id="chip-theme")
            yield Button("search", id="chip-search")
            yield Button("MCP", id="chip-mcp")
            yield Button("Skills", id="chip-skills")
            yield Button("Rules", id="chip-rules")
            yield Button("Notes", id="chip-notes")
        with Horizontal(id="body"):
            with Vertical(id="sessions-panel"):
                yield Static("", id="session-note")
                yield OptionList(id="sessions")
            with Vertical(id="chat-panel"):
                yield RichLog(id="chat", wrap=True, markup=False, min_width=20)
                yield Static("", id="live")
            with Vertical(id="steps-panel"):
                yield RichLog(id="steps", wrap=True, markup=False, min_width=20)
        yield Input(placeholder="Ask codie, or /help", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.theme_changed_signal.subscribe(self, self._remember_theme)
        self.query_one("#sessions-panel").border_title = "Sessions"
        self.query_one("#chat-panel").border_title = "Chat"
        self.query_one("#steps-panel").border_title = "Steps"
        self.refresh_sessions()
        self.refresh_status()
        self.write_chat("codie is ready. Ask a question, or use /help.")
        if self.agent.mcp_errors:
            self.write_chat("MCP: " + "; ".join(self.agent.mcp_errors), role="warn")
        self.query_one("#prompt", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if not button_id.startswith("chip-"):
            return
        event.stop()
        if self._busy:
            return
        actions = {
            "chip-mode": self.action_mode,
            "chip-model": self.action_models,
            "chip-theme": self.action_theme,
            "chip-search": self.action_search,
            "chip-mcp": self.action_mcp,
            "chip-skills": self.action_skills,
            "chip-rules": self.action_allow,
            "chip-notes": self.action_guidelines,
        }
        action = actions.get(button_id)
        if action is not None:
            action()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.control.id != "sessions":
            return
        if self._busy:
            return
        index = event.option_index
        if index < 0 or index >= len(self._session_rows):
            return
        session_id = self._session_rows[index].session_id
        if session_id == self.recorder.session_id and self.agent.messages:
            return
        self.call_later(self.load_session, session_id)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if not text or self._busy:
            return
        self.handle_line(text)

    def handle_line(self, text: str) -> None:
        if text in {"/exit", "exit"}:
            self.exit()
            return
        if text in {"/new", "new"}:
            self.action_new_chat()
            return
        if text == "/help":
            self.write_chat(HELP)
            return
        if text == "/mode":
            self.action_mode()
            return
        if text.startswith("/mode "):
            self.apply_mode_name(text.split(maxsplit=1)[1])
            return
        if text == "/theme":
            self.action_theme()
            return
        if text.startswith("/theme "):
            self.apply_theme_name(text.split(maxsplit=1)[1])
            return
        if text == "/model":
            self.action_models()
            return
        if text.startswith("/model "):
            self.switch_model_text(text[len("/model ") :].strip())
            return
        if text == "/provider" or text.startswith("/provider "):
            name = text[len("/provider") :].strip()
            if not name:
                self.write_chat("Usage: /provider ollama   or   /provider bedrock")
                return
            self.switch_provider(name)
            return
        if text == "/search":
            self.action_search()
            return
        if text.startswith("/search "):
            self.apply_search(text.split(maxsplit=1)[1])
            return
        if text == "/sessions":
            self.action_sessions()
            return
        if text == "/mcp":
            self.action_mcp()
            return
        if text == "/skills":
            self.action_skills()
            return
        if text == "/allow":
            self.action_allow()
            return
        if text == "/guidelines":
            self.action_guidelines()
            return
        self.start_turn(text)

    def action_toggle_sessions(self) -> None:
        panel = self.query_one("#sessions-panel")
        panel.display = not panel.display

    def action_new_chat(self) -> None:
        if self._busy:
            return
        self.agent.reset()
        self.recorder.rotate()
        self.query_one("#chat", RichLog).clear()
        self.query_one("#steps", RichLog).clear()
        self.transcript.clear()
        self.step_lines.clear()
        self.clear_live()
        self.refresh_sessions()
        self.refresh_status()
        self.write_chat("Started a new conversation.")

    def action_models(self) -> None:
        if self._busy:
            return
        choices = [(f"{provider}\t{model}", f"{provider}   {model}") for provider, model in self.model_choices()]
        self.push_screen(PickerModal("Switch model", choices), self._on_model_label)

    def action_theme(self) -> None:
        if self._busy:
            return
        current = self.theme
        choices = [
            (name, name + ("  current" if name == current else ""))
            for name in sorted(self.available_themes)
        ]
        self.push_screen(PickerModal("Theme", choices), self._on_theme)

    def action_mode(self) -> None:
        if self._busy:
            return
        current = self.agent.mode
        choices = [
            ("plan", "plan" + ("  current" if current == "plan" else "")),
            ("agent", "agent" + ("  current" if current == "agent" else "")),
        ]
        self.push_screen(PickerModal("Mode", choices), self._on_mode)

    def action_search(self) -> None:
        if self._busy:
            return
        self.push_screen(SearchModal(active_provider(), searxng_base_url()), self._on_search)

    def action_sessions(self) -> None:
        if self._busy:
            return
        rows, _errors = list_sessions(self.workspace)
        if not rows:
            self.write_chat("No saved conversations for this workspace.")
            return
        choices = [(row.session_id, row.title) for row in rows]
        self.push_screen(PickerModal("Conversations", choices), self._on_session)

    def action_mcp(self) -> None:
        if self._busy:
            return
        servers = load_mcp().get("mcpServers") or {}
        self.push_screen(McpModal(servers, self.agent.mcp_errors, self.save_mcp_servers), self._refocus)

    def action_skills(self) -> None:
        if self._busy:
            return
        skills = discover_skills(self.workspace)
        folder = workspace_skills_dir(self.workspace).relative_to(self.workspace).as_posix()
        if not skills:
            self.write_chat(f"No skills yet. Add SKILL.md under {folder}/<name>/ in this workspace.")
            return
        rows = [(skill.name, skill.description, skill.body) for skill in skills]
        self.push_screen(SkillsModal(rows, folder), self._refocus)

    def action_allow(self) -> None:
        if self._busy:
            return
        permissions = PermissionSet.load(self.workspace)
        if not permissions.rules and not permissions.blocked:
            self.write_chat("No always-allow rules or blocked programs for this workspace.")
            return
        choices = [(f"rule:{rule.id}", f"allow  {rule.label}") for rule in permissions.rules]
        choices.extend((f"block:{name}", f"block  {name}") for name in permissions.blocked)
        self.push_screen(PickerModal("Rules", choices, confirm="Remove"), self._on_revoke)

    def action_guidelines(self) -> None:
        if self._busy:
            return
        lines = guideline_lines(self.workspace)
        if not lines:
            self.write_chat("No workspace guidelines yet.")
            return
        choices = [(str(index), line) for index, line in enumerate(lines)]
        self.push_screen(PickerModal("Workspace guidelines", choices, confirm="Delete"), self._on_delete_guideline)

    def model_choices(self) -> list[tuple[str, str]]:
        choices = [("ollama", name) for name in list_ollama_models()]
        bedrock = bedrock_model_id()
        if bedrock:
            choices.append(("bedrock", bedrock))
        current = (self.provider, self.model_name)
        if current not in choices:
            choices.insert(0, current)
        return choices

    def _on_model_label(self, choice: str | None) -> None:
        if choice and "\t" in choice:
            provider, model_name = choice.split("\t", 1)
            self.apply_model(provider, model_name)
        self._focus_prompt()

    def _on_theme(self, choice: str | None) -> None:
        if choice:
            self.apply_theme_name(choice)
        self._focus_prompt()

    def _on_mode(self, choice: str | None) -> None:
        if choice:
            self.apply_mode_name(choice)
        self._focus_prompt()

    def _on_search(self, choice: tuple[str, str] | None) -> None:
        if choice:
            provider, url = choice
            self.apply_search(provider, url=url)
        self._focus_prompt()

    def _on_session(self, choice: str | None) -> None:
        if choice:
            self.load_session(choice)
        self._focus_prompt()

    def _on_revoke(self, choice: str | None) -> None:
        if choice:
            permissions = PermissionSet.load(self.workspace)
            if choice.startswith("block:"):
                removed = permissions.unblock(self.workspace, choice[len("block:") :])
                self.write_chat("Unblocked a program." if removed else "error: program was already unblocked")
            else:
                rule_id = choice[len("rule:") :] if choice.startswith("rule:") else choice
                removed = permissions.revoke(self.workspace, rule_id)
                self.write_chat("Revoked an always-allow rule." if removed else "error: rule was already gone")
        self._focus_prompt()

    def _on_delete_guideline(self, choice: str | None) -> None:
        if choice is not None and choice.isdigit():
            problem = delete_guideline(self.workspace, int(choice))
            self.write_chat(problem or "Removed a workspace guideline.")
        self._focus_prompt()

    def _refocus(self, _result=None) -> None:
        self._focus_prompt()

    def switch_model_text(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) == 2 and parts[0] in {"ollama", "bedrock"}:
            self.apply_model(parts[0], parts[1])
            return
        self.apply_model(self.provider, text)

    def switch_provider(self, provider: str) -> None:
        provider = provider.strip().lower()
        if provider not in {"ollama", "bedrock"}:
            self.write_chat("error: provider must be ollama or bedrock")
            return
        if provider == "bedrock":
            model_name = bedrock_model_id()
            if not model_name:
                self.write_chat("error: set BEDROCK_MODEL_ID or /model bedrock <model-id>")
                return
        elif self.provider == "ollama":
            model_name = self.model_name
        else:
            model_name = DEFAULT_OLLAMA_MODEL
        self.apply_model(provider, model_name)

    def apply_model(self, provider: str, model_name: str, *, quiet: bool = False) -> None:
        try:
            model = build_chat_model(provider, model_name)
        except Exception as exc:
            self.write_chat(f"error: {exc}")
            return
        self.agent.set_model(model)
        self.provider = provider
        self.model_name = model_name
        self.recorder.provider = provider
        self.recorder.model = model_name
        self.refresh_status()
        if not quiet:
            self.write_chat(f"Switched to {provider} · {model_name}. The conversation is unchanged.")

    def apply_theme_name(self, name: str, *, quiet: bool = False) -> None:
        name = name.strip().lower()
        if name not in self.available_themes:
            self.write_chat("error: unknown theme. Use /theme to pick one.")
            return
        self.theme = name
        save_config({"theme": name})
        self.refresh_status()
        if not quiet:
            self.write_chat(f"Theme is {name}. Saved in {codie_home()}.")

    def _remember_theme(self, theme: object) -> None:
        name = getattr(theme, "name", "")
        if isinstance(name, str) and name in self.available_themes:
            save_config({"theme": name})
            if self.is_running:
                self._chip("chip-theme", name)

    def apply_mode_name(self, name: str, *, quiet: bool = False) -> None:
        name = name.strip().lower()
        if name not in {"agent", "plan"}:
            self.write_chat("error: mode must be agent or plan")
            return
        self.agent.set_mode(name)
        save_config({"last_mode": name})
        self.refresh_status()
        if not quiet:
            self.write_chat(f"Switched to {name} mode.")

    def apply_search(self, name: str, *, url: str | None = None, quiet: bool = False) -> None:
        from codie.search import AVAILABLE_PROVIDERS, LATER_PROVIDERS

        provider = name.strip().lower()
        if provider in LATER_PROVIDERS:
            self.write_chat(f"error: {provider} is not available yet")
            return
        if provider not in AVAILABLE_PROVIDERS:
            self.write_chat("error: search provider must be ddgs, searxng, tavily, or openai")
            return
        payload: dict = {"search_provider": "" if provider == "ddgs" else provider}
        if provider == "searxng" and url is not None and url.strip():
            payload["searxng_url"] = url.strip()
        save_config(payload)
        self.recorder.search_provider = active_provider()
        self.refresh_status()
        if not quiet:
            self.write_chat(f"Web search is {provider_label()}.")

    def save_mcp_servers(self, servers: dict) -> list[str]:
        save_mcp({"mcpServers": servers})
        errors = self.agent.refresh_mcp()
        self.refresh_status()
        return errors

    def load_session(self, session_id: str) -> None:
        if self._busy:
            return
        loaded = read_session(self.workspace, session_id)
        if isinstance(loaded, str):
            self.write_chat(loaded)
            return
        self.agent.messages = list(loaded.messages)
        self.agent.last_rounds = 0
        if loaded.mode != self.agent.mode:
            self.agent.set_mode(loaded.mode)
            save_config({"last_mode": loaded.mode})
        self.recorder.session_id = loaded.session_id
        self.recorder.search_provider = loaded.search_provider or active_provider()
        if loaded.search_provider and loaded.search_provider != active_provider():
            self.apply_search(loaded.search_provider, quiet=True)
        same_model = loaded.provider == self.provider and loaded.model == self.model_name
        if loaded.provider and loaded.model and not same_model:
            self.apply_model(loaded.provider, loaded.model, quiet=True)
        else:
            self.recorder.provider = self.provider
            self.recorder.model = self.model_name
        self._replay(loaded.messages, loaded.title)
        self.refresh_status()
        self.refresh_sessions()

    def start_turn(self, text: str) -> None:
        self._busy = True
        self.query_one("#prompt", Input).disabled = True
        self.write_chat(text, role="you")
        self.clear_live()
        self.refresh_status(working=True)

        def work() -> None:
            try:
                answer = self.agent.run_turn(text)
            except Exception as exc:
                self.call_from_thread(self.finish_error, exc)
                return
            self.call_from_thread(self.finish_turn, answer)

        self.run_worker(work, name="turn", thread=True, exclusive=True, exit_on_error=False)

    def request_approval(self, request: ApprovalRequest) -> None:
        self._pending_request = request
        self.push_screen(ApprovalModal(request), self._approval_done)

    def _approval_done(self, choice: str | bool | None) -> None:
        allowed = choice in {True, "allow", "always"}
        if choice == "always" and self._pending_request is not None:
            label = remember_request(self.workspace, self._pending_request)
            if label:
                self.write_chat(f"Always allowed for this workspace: {label}")
        if choice == "block" and self._pending_request is not None:
            program = command_program(self._pending_request.command)
            if program:
                saved = PermissionSet.load(self.workspace).block(self.workspace, program)
                if saved.startswith("error:"):
                    self.write_chat(saved)
                else:
                    self.write_chat(f"Blocked {saved} for this workspace.")
        self.approver.results.put(allowed)

    def add_step(self, line: str) -> None:
        self.step_lines.append(line)
        self.query_one("#steps", RichLog).write(_paint_step(line))

    def add_token(self, text: str) -> None:
        self._live += text
        self._show_live()

    def clear_live(self) -> None:
        self._live = ""
        self._show_live()

    def _show_live(self) -> None:
        if not self.is_running:
            return
        if self._live:
            shown = _paint_chat("codie", f"codie> {self._live}")
        elif self._busy:
            shown = _paint_chat("processing", "codie> …")
        else:
            shown = ""
        self.query_one("#live", Static).update(shown)

    def finish_turn(self, answer: str) -> None:
        final = (answer or "").strip()
        if final:
            self.write_chat(final, role="codie")
        self._busy = False
        self.clear_live()
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()
        self.refresh_sessions()
        self.refresh_status()

    def finish_error(self, exc: BaseException) -> None:
        self.write_chat(f"error: {exc}")
        self._busy = False
        self.clear_live()
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()
        self.refresh_status()

    def write_chat(self, line: str, *, role: str = "info") -> None:
        if role not in {"you", "codie"} and line.startswith("error:"):
            role = "warn"
        prefix = role if role in {"you", "codie", "info", "warn"} else "info"
        plain = f"{prefix}> {line}"
        self.transcript.append(plain)
        self.query_one("#chat", RichLog).write(_paint_chat(role, plain))

    def refresh_sessions(self) -> None:
        if not self.is_running:
            return
        rows, errors = list_sessions(self.workspace)
        self._session_rows = rows
        widget = self.query_one("#sessions", OptionList)
        widget.clear_options()
        for info in rows:
            marker = ">" if info.session_id == self.recorder.session_id else " "
            widget.add_option(f"{marker} {info.title}")
        note = ("Unreadable: " + ", ".join(errors)) if errors else ""
        self.query_one("#session-note", Static).update(note)

    def refresh_status(self, working: bool = False) -> None:
        state = "working" if working or self._busy else "ready"
        folder = self.workspace.name or str(self.workspace)
        search = provider_label()
        self.status_line = (
            f"{self.agent.mode}  {self.provider}  {self.model_name}  {search}  "
            f"{folder}  round {self.agent.last_rounds}  {state}"
        )
        self.query_one("#status", Static).update(self.status_line)
        self._chip("chip-mode", self.agent.mode)
        self._chip("chip-model", f"{self.provider} {self.model_name}")
        self._chip("chip-theme", self.theme)
        self._chip("chip-search", search)

    def _replay(self, messages: list, title: str) -> None:
        self.query_one("#chat", RichLog).clear()
        self.transcript.clear()
        self.query_one("#steps", RichLog).clear()
        self.step_lines.clear()
        for message in messages:
            if isinstance(message, HumanMessage):
                self.write_chat(message_text(message), role="you")
            elif isinstance(message, AIMessage) and message.tool_calls:
                for call in message.tool_calls:
                    name = call.get("name", "tool") if isinstance(call, dict) else "tool"
                    self.add_step(f"-> {name}(...)")
            elif isinstance(message, AIMessage):
                text = visible_text(message_text(message))
                if text:
                    self.write_chat(text, role="codie")
        self.write_chat(f'Restored "{title}".')

    def _chip(self, widget_id: str, label: str) -> None:
        if not self.is_running:
            return
        text = " ".join(label.split())
        if len(text) > 22:
            text = text[:21] + "…"
        self.query_one(f"#{widget_id}", Button).label = text

    def _focus_prompt(self) -> None:
        if self.is_running and not self._busy:
            self.query_one("#prompt", Input).focus()


def _model_label(provider: str, model) -> str:
    if model is not None:
        label = getattr(model, "model", None) or getattr(model, "model_id", None)
        if label:
            return str(label)
    if provider == "bedrock":
        return bedrock_model_id() or "bedrock"
    return DEFAULT_OLLAMA_MODEL


def _paint_chat(role: str, line: str) -> Text:
    prefix_style = {
        "you": "bold cyan",
        "codie": "bold green",
        "info": "dim",
        "warn": "bold yellow",
        "processing": "dim italic",
    }.get(role, "")
    body_style = {"warn": "yellow", "info": "dim", "processing": "dim italic"}.get(role, "")
    head, sep, tail = line.partition("> ")
    if sep and head in {"you", "codie", "info", "warn"}:
        text = Text()
        text.append(head + ">", style=prefix_style)
        text.append(" " + tail, style=body_style)
        return text
    return Text(line, style=prefix_style or body_style)


def _paint_step(line: str) -> Text:
    text = Text(line)
    if line.startswith("->"):
        text.stylize("cyan")
    elif "error:" in line or "denied" in line or "step limit" in line:
        text.stylize("red")
    elif line.startswith("<-"):
        text.stylize("green")
    return text
