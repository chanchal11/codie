"""Textual pilot tests for the IDE layout, steps, model switch, and approvals."""

import os
from pathlib import Path

from langchain_core.messages import AIMessage

from codie.agent import CodingAgent
from codie.approvals import AllowAll
from codie.tui.app import ApprovalModal, CodieApp
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
        assert "ollama" in app.status_line
        assert "scripted" in app.status_line
        assert "ready" in app.status_line
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
        assert any(line.startswith("you>") for line in app.transcript)


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
        await _wait(pilot, lambda: "remember me" in app.transcript)
        await pilot.press(*"/new", "enter")
        await pilot.pause()
        assert app.transcript == ["Started a new conversation."]
        assert app.agent.messages == []
        await pilot.press(*"second", "enter")
        await _wait(pilot, lambda: "fresh start" in app.transcript)
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
        await _wait(pilot, lambda: "Left the file alone." in app.transcript)
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
        await _wait(pilot, lambda: "Updated hello.txt." in app.transcript)
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
        await _wait(pilot, lambda: "from the first model" in app.transcript)
        await pilot.press(*"/model other-model", "enter")
        await pilot.pause()
        assert app.model_name == "other-model"
        assert "other-model" in app.status_line
        assert any("Switched to ollama" in line for line in app.transcript)
        await pilot.press(*"continue", "enter")
        await _wait(pilot, lambda: "from the second model" in app.transcript)
        seen = " ".join(str(getattr(message, "content", "")) for message in second.prompts[0])
        assert "remember" in seen
        assert "from the first model" in seen


async def test_provider_bedrock_without_config_is_an_error(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    app = _app(tmp_path, ScriptedModel([]))
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press(*"/provider bedrock", "enter")
        await pilot.pause()
        assert any("BEDROCK_MODEL_ID" in line for line in app.transcript)
        assert app.provider == "ollama"
    assert "BEDROCK_MODEL_ID" not in os.environ or os.environ.get("BEDROCK_MODEL_ID", "") == ""
