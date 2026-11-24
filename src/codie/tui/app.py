"""IDE-style terminal UI: chat, live steps, model switching, and approvals."""

from __future__ import annotations

import queue
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, OptionList, RichLog, Static

from codie.agent import CodingAgent
from codie.approvals import ApprovalRequest
from codie.guards import MAX_ROUNDS_DEFAULT
from codie.providers import DEFAULT_OLLAMA_MODEL, bedrock_model_id, build_chat_model, list_ollama_models


class TuiApprover:
    """Block the agent thread until the modal returns Allow or Deny."""

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


class ApprovalModal(ModalScreen[bool]):
    BINDINGS = [
        Binding("y", "allow", "Allow"),
        Binding("n", "deny", "Deny"),
        Binding("escape", "deny", "Deny"),
    ]

    DEFAULT_CSS = """
    ApprovalModal {
        align: center middle;
    }
    ApprovalModal > #dialog {
        width: 88%;
        height: 80%;
        border: solid $warning;
        background: $surface;
        padding: 1 2;
    }
    ApprovalModal #diff {
        height: 1fr;
        margin: 1 0;
    }
    ApprovalModal #buttons {
        layout: horizontal;
        height: auto;
        align: right middle;
    }
    ApprovalModal Button {
        margin-left: 1;
    }
    """

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        title = "Allow this write?" if self.request.kind == "write" else "Allow pytest?"
        with Vertical(id="dialog"):
            yield Static(title, id="ask")
            yield Static("Allow? [y/N]")
            yield RichLog(id="diff", wrap=True, markup=False, min_width=20)
            with Horizontal(id="buttons"):
                yield Button("Deny", id="deny", variant="error")
                yield Button("Allow", id="allow", variant="success")

    def on_mount(self) -> None:
        self.query_one("#diff", RichLog).write(_render_request(self.request))
        self.query_one("#deny", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(event.button.id == "allow")

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


class ModelModal(ModalScreen[tuple[str, str] | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    ModelModal {
        align: center middle;
    }
    ModelModal > #dialog {
        width: 72;
        height: 24;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }
    ModelModal #models {
        height: 1fr;
        margin-top: 1;
    }
    """

    def __init__(self, choices: list[tuple[str, str]]) -> None:
        super().__init__()
        self.choices = choices

    def compose(self) -> ComposeResult:
        labels = [f"{provider}   {model}" for provider, model in self.choices]
        with Vertical(id="dialog"):
            yield Static("Switch model")
            yield OptionList(*labels, id="models")
            yield Button("Cancel", id="cancel")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self.choices[event.option_index])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class CodieApp(App):
    """Chat on the left, tool steps on the right, model name in the status bar."""

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
    #body {
        layout: horizontal;
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
        Binding("ctrl+n", "new_chat", "New"),
        Binding("ctrl+m", "models", "Model"),
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
        self.workspace = Path(workspace).resolve()
        self.provider = provider
        self.model_name = model_name or _model_label(provider, model)
        self.max_rounds = max_rounds
        self._live = ""
        self._busy = False
        self.transcript: list[str] = []
        self.step_lines: list[str] = []
        self.status_line = ""
        self.approver = TuiApprover(self)
        self.sink = TuiSink(self)
        if agent is None:
            if model is None:
                model = build_chat_model(provider, self.model_name)
            self.agent = CodingAgent(model, self.workspace, self.approver, self.sink, max_rounds)
        else:
            agent.approver = self.approver
            agent.sink = self.sink
            self.agent = agent

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="status")
        with Horizontal(id="body"):
            with Vertical(id="chat-panel"):
                yield RichLog(id="chat", wrap=True, markup=False, min_width=20)
                yield Static("", id="live")
            with Vertical(id="steps-panel"):
                yield RichLog(id="steps", wrap=True, markup=False, min_width=20)
        yield Input(placeholder="Ask codie, or /model  /provider  /new  /exit", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chat-panel").border_title = "Chat"
        self.query_one("#steps-panel").border_title = "Steps"
        self.query_one("#prompt", Input).focus()
        self.refresh_status()
        self.write_chat("codie is ready. Ask a question, or use /model, /new, and /exit.")

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
            self.write_chat("Commands: /model, /model <name>, /provider ollama|bedrock, /new, /exit")
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
        self.start_turn(text)

    def action_new_chat(self) -> None:
        if self._busy:
            return
        self.agent.reset()
        self.query_one("#chat", RichLog).clear()
        self.query_one("#steps", RichLog).clear()
        self.transcript.clear()
        self.step_lines.clear()
        self.clear_live()
        self.refresh_status()
        self.write_chat("Started a new conversation.")

    def action_models(self) -> None:
        if self._busy:
            return
        self.push_screen(ModelModal(self.model_choices()), self._on_model_chosen)

    def model_choices(self) -> list[tuple[str, str]]:
        choices = [("ollama", name) for name in list_ollama_models()]
        bedrock = bedrock_model_id()
        if bedrock:
            choices.append(("bedrock", bedrock))
        current = (self.provider, self.model_name)
        if current not in choices:
            choices.insert(0, current)
        return choices

    def _on_model_chosen(self, choice: tuple[str, str] | None) -> None:
        if choice:
            self.apply_model(*choice)

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

    def apply_model(self, provider: str, model_name: str) -> None:
        try:
            model = build_chat_model(provider, model_name)
        except Exception as exc:
            self.write_chat(f"error: {exc}")
            return
        self.agent.set_model(model)
        self.provider = provider
        self.model_name = model_name
        self.refresh_status()
        self.write_chat(f"Switched to {provider} · {model_name}. The conversation is unchanged.")

    def start_turn(self, text: str) -> None:
        self._busy = True
        self.query_one("#prompt", Input).disabled = True
        self.write_chat(f"you> {text}")
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
        self.push_screen(ApprovalModal(request), self._approval_done)

    def _approval_done(self, allowed: bool | None) -> None:
        self.approver.results.put(bool(allowed))

    def add_step(self, line: str) -> None:
        self.step_lines.append(line)
        self.query_one("#steps", RichLog).write(_paint_step(line))

    def add_token(self, text: str) -> None:
        self._live += text
        self.query_one("#live", Static).update(self._live)

    def clear_live(self) -> None:
        self._live = ""
        if self.is_running:
            self.query_one("#live", Static).update("")

    def finish_turn(self, answer: str) -> None:
        final = (answer or "").strip()
        if final:
            self.write_chat(final)
        self.clear_live()
        self._busy = False
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()
        self.refresh_status()

    def finish_error(self, exc: BaseException) -> None:
        self.write_chat(f"error: {exc}")
        self.clear_live()
        self._busy = False
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()
        self.refresh_status()

    def write_chat(self, line: str) -> None:
        self.transcript.append(line)
        self.query_one("#chat", RichLog).write(line)

    def refresh_status(self, working: bool = False) -> None:
        state = "working" if working or self._busy else "ready"
        folder = self.workspace.name or str(self.workspace)
        self.status_line = (
            f"{self.provider}  {self.model_name}  {folder}  round {self.agent.last_rounds}  {state}"
        )
        self.query_one("#status", Static).update(self.status_line)


def _model_label(provider: str, model) -> str:
    if model is not None:
        label = getattr(model, "model", None) or getattr(model, "model_id", None)
        if label:
            return str(label)
    if provider == "bedrock":
        return bedrock_model_id() or "bedrock"
    return DEFAULT_OLLAMA_MODEL


def _paint_step(line: str) -> Text:
    text = Text(line)
    if line.startswith("->"):
        text.stylize("cyan")
    elif "error:" in line or "denied" in line or "step limit" in line:
        text.stylize("red")
    elif line.startswith("<-"):
        text.stylize("green")
    return text


def _render_request(request: ApprovalRequest) -> Text:
    text = Text()
    if request.kind == "write" and request.diff:
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
