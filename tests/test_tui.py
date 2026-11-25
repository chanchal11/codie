"""Textual pilot tests for the IDE layout, steps, model switch, and approvals."""

import os
import threading
import time
from pathlib import Path

from langchain_core.messages import AIMessage

from codie.agent import CodingAgent
from codie.approvals import AllowAll
from codie.store import codie_home, load_config, save_config
from codie.tui.app import ApprovalModal, CodieApp
from codie.tui.screens import PickerModal
from tests.support import ScriptedModel


async def _wait(pilot, predicate, steps: int = 40) -> None:
    for _ in range(steps):
        if predicate():
            return
        await pilot.pause(0.05)
    raise AssertionError("timed out waiting for the TUI")


def _app(tmp_path: Path, model, approver=AllowAll()) -> CodieApp:
    agent = CodingAgent(model, tmp_path, approver)
    return CodieApp(
        workspace=tmp_path,
        provider="ollama",
        model_name="scripted",
        agent=agent,
    )


async def test_layout_and_exit(tmp_path: Path):
    app = _app(tmp_path, ScriptedModel([AIMessage(content="unused")]))
    async with app.run_test(size=(120, 36)) as pilot:
        assert app.query_one("#chat")
        assert app.query_one("#steps")
        assert app.query_one("#prompt")
        assert app.query_one("#stop").disabled
        assert "ollama" in app.status_line
        assert "scripted" in app.status_line
        assert "agent" in app.status_line
        assert "ddgs" in app.status_line
        assert "ready" in app.status_line
        assert app.query_one("#sessions")
        assert app.query_one("#chip-mode")
        assert app.query_one("#chip-search")
        await pilot.press("/", "e", "x", "i", "t", "enter")
        await pilot.pause()
    assert app.return_value is None or app._exit


async def test_turn_shows_steps_and_answer(tmp_path: Path):
    (tmp_path / "hello.txt").write_text("hello\n", encoding="utf-8")
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"path": "hello.txt"}, "id": "r1", "type": "tool_call"}
                ],
            ),
            AIMessage(content="The file says hello."),
        ]
    )
    app = _app(tmp_path, model)
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press(*"look", "enter")
        await _wait(pilot, lambda: any("says hello" in line for line in app.transcript))
        assert any(line.startswith('-> read_file("hello.txt")') for line in app.step_lines)
        assert any(line.startswith("<-") for line in app.step_lines)
        assert any(line.startswith("you> ") for line in app.transcript)
        assert any(line.startswith("codie> ") and "says hello" in line for line in app.transcript)
        assert any(line.startswith("info> ") for line in app.transcript)


async def test_new_clears_the_conversation(tmp_path: Path):
    model = ScriptedModel(
        [
            AIMessage(content="remember me"),
            AIMessage(content="fresh start"),
        ]
    )
    app = _app(tmp_path, model)
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press(*"first", "enter")
        await _wait(pilot, lambda: any("remember me" in line for line in app.transcript))
        await pilot.press(*"/new", "enter")
        await pilot.pause()
        assert app.transcript == ["info> Started a new conversation."]
        assert app.agent.messages == []
        await pilot.press(*"second", "enter")
        await _wait(pilot, lambda: any("fresh start" in line for line in app.transcript))
        seen = " ".join(getattr(message, "content", "") for message in model.prompts[-1])
        assert "first" not in seen
        assert "second" in seen


async def test_deny_write_from_the_diff_modal(tmp_path: Path):
    target = tmp_path / "hello.txt"
    target.write_text("old\n", encoding="utf-8")
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": "hello.txt", "content": "new\n"},
                        "id": "w1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Left the file alone."),
        ]
    )
    app = _app(tmp_path, model)
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press(*"rewrite", "enter")
        await _wait(pilot, lambda: isinstance(app.screen, ApprovalModal))
        modal = app.screen
        assert isinstance(modal, ApprovalModal)
        assert "\x1b[" not in modal.request.diff
        assert "+new" in modal.request.diff
        await pilot.press("n")
        await _wait(pilot, lambda: any("Left the file alone." in line for line in app.transcript))
    assert target.read_text(encoding="utf-8") == "old\n"
    assert any("write_file" in line for line in app.step_lines)
    assert any("denied" in line for line in app.step_lines)


async def test_allow_write_from_the_diff_modal(tmp_path: Path):
    target = tmp_path / "hello.txt"
    target.write_text("old\n", encoding="utf-8")
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": "hello.txt", "content": "new\n"},
                        "id": "w1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Updated hello.txt."),
        ]
    )
    app = _app(tmp_path, model)
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press(*"rewrite", "enter")
        await _wait(pilot, lambda: isinstance(app.screen, ApprovalModal))
        await pilot.press("y")
        await _wait(pilot, lambda: any("Updated hello.txt." in line for line in app.transcript))
    assert target.read_text(encoding="utf-8") == "new\n"


async def test_model_command_keeps_history(tmp_path: Path, monkeypatch):
    first = ScriptedModel([AIMessage(content="from the first model")])
    second = ScriptedModel([AIMessage(content="from the second model")])

    def fake_build(provider, model_name):
        assert provider == "ollama"
        assert model_name == "other-model"
        return second

    monkeypatch.setattr("codie.tui.app.build_chat_model", fake_build)
    app = _app(tmp_path, first)
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press(*"remember", "enter")
        await _wait(pilot, lambda: any("from the first model" in line for line in app.transcript))
        await pilot.press(*"/model other-model", "enter")
        await pilot.pause()
        assert app.model_name == "other-model"
        assert "other-model" in app.status_line
        assert any("Switched to ollama" in line for line in app.transcript)
        await pilot.press(*"continue", "enter")
        await _wait(pilot, lambda: any("from the second model" in line for line in app.transcript))
        seen = " ".join(str(getattr(message, "content", "")) for message in second.prompts[0])
        assert "remember" in seen
        assert "from the first model" in seen


async def test_status_switches_to_plan_mode(tmp_path: Path):
    app = _app(tmp_path, ScriptedModel([]))
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press(*"/mode plan", "enter")
        await pilot.pause()
        assert app.agent.mode == "plan"
        assert "plan" in app.status_line


async def test_session_switch_restores_the_conversation(tmp_path: Path):
    model = ScriptedModel([AIMessage(content="remember me")])
    app = _app(tmp_path, model)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press(*"first topic", "enter")
        await _wait(pilot, lambda: any("remember me" in line for line in app.transcript))
        await pilot.press(*"/new", "enter")
        await pilot.pause()
        assert app.agent.messages == []
        await pilot.press(*"/sessions", "enter")
        await _wait(pilot, lambda: isinstance(app.screen, PickerModal))
        await pilot.press("enter")
        await _wait(pilot, lambda: any("remember me" in line for line in app.transcript))
        assert any(
            getattr(message, "content", "") == "first topic" for message in app.agent.messages
        )


async def test_always_allow_from_the_modal(tmp_path: Path):
    target = tmp_path / "hello.txt"
    target.write_text("old\n", encoding="utf-8")

    def write(content: str, call_id: str) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"path": "hello.txt", "content": content},
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        )

    model = ScriptedModel(
        [
            write("new\n", "w1"),
            AIMessage(content="once"),
            write("newer\n", "w2"),
            AIMessage(content="twice"),
        ]
    )
    app = _app(tmp_path, model)
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press(*"rewrite", "enter")
        await _wait(pilot, lambda: isinstance(app.screen, ApprovalModal))
        await pilot.press("a")
        await _wait(pilot, lambda: any("once" in line for line in app.transcript))
        assert target.read_text(encoding="utf-8") == "new\n"
        await pilot.press(*"again", "enter")
        await _wait(pilot, lambda: any("twice" in line for line in app.transcript))
    assert target.read_text(encoding="utf-8") == "newer\n"
    assert any("Always allowed" in line for line in app.transcript)


async def test_provider_bedrock_without_config_is_an_error(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    app = _app(tmp_path, ScriptedModel([]))
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press(*"/provider bedrock", "enter")
        await pilot.pause()
        assert any("BEDROCK_MODEL_ID" in line for line in app.transcript)
        assert app.provider == "ollama"
    assert "BEDROCK_MODEL_ID" not in os.environ or os.environ.get("BEDROCK_MODEL_ID", "") == ""


async def test_theme_command_is_saved_under_codie_home(tmp_path: Path):
    app = _app(tmp_path, ScriptedModel([]))
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press(*"/theme nord", "enter")
        await pilot.pause()
        assert app.theme == "nord"
        assert app.query_one("#chip-theme").label == "nord"
        assert any(str(codie_home()) in line for line in app.transcript)
    assert load_config()["theme"] == "nord"
    assert (codie_home() / "config.json").is_file()


async def test_saved_theme_is_restored(tmp_path: Path):
    save_config({"theme": "gruvbox"})
    app = _app(tmp_path, ScriptedModel([]))
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        assert app.theme == "gruvbox"
        assert app.query_one("#chip-theme").label == "gruvbox"


async def test_unknown_theme_is_rejected(tmp_path: Path):
    app = _app(tmp_path, ScriptedModel([]))
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press(*"/theme nope", "enter")
        await pilot.pause()
        assert app.theme == "textual-dark"
        assert any(line.startswith("warn> ") and "unknown theme" in line for line in app.transcript)


def _command_script(commands: list[str], reply: str) -> ScriptedModel:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_command",
                    "args": {"command": command},
                    "id": f"c{index}",
                    "type": "tool_call",
                }
            ],
        )
        for index, command in enumerate(commands)
    ]
    messages.append(AIMessage(content=reply))
    return ScriptedModel(messages)


async def test_allow_runs_an_off_list_command(tmp_path: Path, monkeypatch):
    calls: list = []

    def fake_run(argv, *_args, **_kwargs):
        calls.append(list(argv))

        class Result:
            returncode = 0
            stdout = "hello\n"
            stderr = ""

        return Result()

    monkeypatch.setattr("codie.tools._launch_command", fake_run)
    app = _app(tmp_path, _command_script(["echo hello"], "Ran echo."))
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press(*"run it", "enter")
        await _wait(pilot, lambda: isinstance(app.screen, ApprovalModal))
        modal = app.screen
        assert isinstance(modal, ApprovalModal)
        assert modal.request.kind == "command"
        assert modal.query_one("#block")
        await pilot.press("y")
        await _wait(pilot, lambda: any(line.startswith("codie> ") and "Ran echo." in line for line in app.transcript))
    assert calls == [["echo", "hello"]]
    assert any("you> run it" == line for line in app.transcript)


async def test_deny_does_not_run_an_off_list_command(tmp_path: Path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("subprocess should not run")

    monkeypatch.setattr("codie.tools.subprocess.run", boom)
    app = _app(tmp_path, _command_script(["echo hello"], "Left it alone."))
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press(*"run it", "enter")
        await _wait(pilot, lambda: isinstance(app.screen, ApprovalModal))
        await pilot.press("n")
        await _wait(pilot, lambda: "Left it alone." in "".join(app.transcript))
    assert any("denied" in line for line in app.step_lines)


async def test_block_skips_the_modal_the_second_time(tmp_path: Path, monkeypatch):
    from codie.permissions import PermissionSet

    def boom(*_args, **_kwargs):
        raise AssertionError("subprocess should not run")

    monkeypatch.setattr("codie.tools.subprocess.run", boom)
    app = _app(
        tmp_path,
        _command_script(["python -c 'print(1)'", "python -c 'print(2)'"], "Stopped."),
    )
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press(*"run python", "enter")
        await _wait(pilot, lambda: isinstance(app.screen, ApprovalModal))
        await pilot.press("b")
        await _wait(pilot, lambda: "Stopped." in "".join(app.transcript))
        assert not isinstance(app.screen, ApprovalModal)
    assert PermissionSet.load(tmp_path).is_blocked("python")
    assert any("blocked" in line for line in app.step_lines)
    assert any(line.startswith("info> ") and "Blocked python" in line for line in app.transcript)


async def test_stop_button_ends_the_running_turn(tmp_path: Path):
    class Hold:
        def __init__(self) -> None:
            self.agent = None
            self.started = threading.Event()
            self.calls = 0

        def bind_tools(self, tools, **_kwargs):
            return self

        def invoke(self, _messages):
            return AIMessage(content="after stop")

        def stream(self, _messages):
            self.calls += 1
            if self.calls > 1:
                yield AIMessage(content="after stop")
                return
            self.started.set()
            while self.agent is None or not self.agent.cancelled():
                time.sleep(0.01)
            yield AIMessage(content="too late")

    model = Hold()
    app = _app(tmp_path, model)
    model.agent = app.agent
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press(*"hello", "enter")
        await _wait(pilot, lambda: model.started.is_set() and not app.query_one("#stop").disabled)
        await pilot.click("#stop")
        await _wait(pilot, lambda: any(line == "codie> Stopped." for line in app.transcript))
        assert any(line == "<- stopping" for line in app.step_lines)
        assert not any("too late" in line for line in app.transcript)
        assert app.query_one("#stop").disabled
        assert app.query_one("#prompt").disabled is False
        await pilot.press(*"next", "enter")
        await _wait(pilot, lambda: any("after stop" in line for line in app.transcript))
