"""Agent loop tests with a scripted chat model. No Ollama or AWS required."""

from pathlib import Path

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from codie.agent import CodingAgent, visible_text
from codie.approvals import AllowAll, DenyAll
from tests.support import ScriptedModel


class AlwaysTools:
    def __init__(self) -> None:
        self.n = 0

    def bind_tools(self, tools, **_kwargs):
        return self

    def invoke(self, _messages):
        self.n += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"path": "hello.txt"},
                    "id": f"c{self.n}",
                    "type": "tool_call",
                }
            ],
        )

    def stream(self, messages):
        yield self.invoke(messages)


class RecordingApprover:
    def __init__(self, allow: bool) -> None:
        self.allow = allow
        self.requests = []

    def ask(self, request) -> bool:
        self.requests.append(request)
        return self.allow


class ListSink:
    def __init__(self) -> None:
        self.steps: list[str] = []
        self.tokens: list[str] = []

    def step(self, line: str) -> None:
        self.steps.append(line)

    def token(self, text: str) -> None:
        self.tokens.append(text)

    def drop_tokens(self) -> None:
        return None


def _text(messages) -> str:
    parts = []
    for message in messages:
        content = getattr(message, "content", "")
        parts.append(content if isinstance(content, str) else str(content))
    return "\n".join(parts)


def test_visible_text_drops_thinking_tags():
    raw = "<thinking>scratch</thinking>\n\nThe folder splits a bill."
    assert visible_text(raw) == "The folder splits a bill."


def test_several_tool_calls_in_one_round_are_all_answered(tmp_path: Path):
    (tmp_path / "hello.txt").write_text("hello\n", encoding="utf-8")
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "list_files", "args": {"path": "."}, "id": "1", "type": "tool_call"},
                    {"name": "read_file", "args": {"path": "hello.txt"}, "id": "2", "type": "tool_call"},
                ],
            ),
            AIMessage(content="It says hello."),
        ]
    )
    agent = CodingAgent(model, tmp_path, AllowAll(), ListSink())
    answer = agent.run_turn("What is in the folder?")
    assert answer == "It says hello."
    tool_messages = [message for message in model.prompts[1] if isinstance(message, ToolMessage)]
    assert {message.tool_call_id for message in tool_messages} == {"1", "2"}
    assert any("hello.txt" in (message.content or "") for message in tool_messages)
    assert isinstance(model.prompts[0][0], SystemMessage)
    assert isinstance(model.prompts[1][0], SystemMessage)
    assert any(line.startswith("-> list_files(") for line in agent.sink.steps)
    assert any(line.startswith("-> read_file(") for line in agent.sink.steps)


def test_follow_up_keeps_history_and_new_drops_it(tmp_path: Path):
    model = ScriptedModel(
        [
            AIMessage(content="first answer"),
            AIMessage(content="second answer"),
            AIMessage(content="fresh"),
        ]
    )
    agent = CodingAgent(model, tmp_path, AllowAll())
    assert agent.run_turn("what does this do?") == "first answer"
    assert agent.run_turn("now add a test") == "second answer"
    history = _text(model.prompts[1])
    assert "what does this do?" in history
    assert "first answer" in history
    assert "now add a test" in history
    agent.reset()
    assert agent.run_turn("hello again") == "fresh"
    fresh = _text(model.prompts[-1])
    assert "hello again" in fresh
    assert "what does this do?" not in fresh


def test_switching_models_keeps_the_thread(tmp_path: Path):
    first = ScriptedModel([AIMessage(content="from one")])
    second = ScriptedModel([AIMessage(content="from two")])
    agent = CodingAgent(first, tmp_path, AllowAll())
    agent.run_turn("remember this")
    agent.set_model(second)
    assert agent.run_turn("and this") == "from two"
    seen = _text(second.prompts[0])
    assert "remember this" in seen
    assert "from one" in seen
    assert "and this" in seen


def test_round_cap_closes_open_tool_calls(tmp_path: Path):
    (tmp_path / "hello.txt").write_text("hello\n", encoding="utf-8")
    sink = ListSink()
    agent = CodingAgent(AlwaysTools(), tmp_path, AllowAll(), sink, max_rounds=1)
    answer = agent.run_turn("keep going")
    assert answer == "Stopped: step limit reached for this question."
    assert not any(line.startswith("-> read_file(") for line in sink.steps)
    assert any("step limit" in line for line in sink.steps)
    tool_messages = [message for message in agent.messages if isinstance(message, ToolMessage)]
    assert tool_messages
    assert all(message.content.startswith("error: step limit") for message in tool_messages)
    assert {message.tool_call_id for message in tool_messages} == {"c1"}


def test_round_cap_runs_tools_before_the_last_round(tmp_path: Path):
    (tmp_path / "hello.txt").write_text("hello\n", encoding="utf-8")
    sink = ListSink()
    agent = CodingAgent(AlwaysTools(), tmp_path, AllowAll(), sink, max_rounds=2)
    agent.run_turn("keep going")
    reads = [line for line in sink.steps if line.startswith("-> read_file(")]
    assert len(reads) == 1
    limit_messages = [
        message
        for message in agent.messages
        if isinstance(message, ToolMessage) and "step limit" in message.content
    ]
    assert len(limit_messages) == 1
    assert limit_messages[0].tool_call_id == "c2"


def test_denied_write_does_not_change_the_file(tmp_path: Path):
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
            AIMessage(content="I left it alone."),
        ]
    )
    approver = RecordingApprover(allow=False)
    agent = CodingAgent(model, tmp_path, approver)
    assert agent.run_turn("rewrite it") == "I left it alone."
    assert target.read_text(encoding="utf-8") == "old\n"
    assert approver.requests[0].kind == "write"
    assert "\x1b[" not in approver.requests[0].diff
    assert "+new" in approver.requests[0].diff.replace(" ", "")
    denial = next(message for message in agent.messages if isinstance(message, ToolMessage))
    assert denial.content == "error: user denied this action"
    assert denial.tool_call_id == "w1"
    assert "user denied" in _text(model.prompts[1])


def test_web_search_is_a_normal_tool(tmp_path: Path, monkeypatch):
    class FakeDDGS:
        def text(self, query, max_results=5, backend="auto"):
            assert query == "latest pytest version"
            return [{"title": "pytest", "href": "https://example.com/pytest", "body": "pytest 9.2.0"}]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "latest pytest version"},
                        "id": "s1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="pytest 9.2.0 is current. Source: https://example.com/pytest"),
        ]
    )
    agent = CodingAgent(model, tmp_path, AllowAll())
    answer = agent.run_turn("What's the latest pytest?")
    assert "9.2.0" in answer
    assert "https://example.com/pytest" in answer
    tool_message = next(message for message in agent.messages if isinstance(message, ToolMessage))
    assert "https://example.com/pytest" in tool_message.content
    assert "\x1b[" not in tool_message.content


def test_plan_mode_blocks_a_write_without_asking(tmp_path: Path):
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
            AIMessage(content="I only planned it."),
        ]
    )
    approver = RecordingApprover(allow=True)
    agent = CodingAgent(model, tmp_path, approver, mode="plan")
    assert agent.run_turn("rewrite it") == "I only planned it."
    assert target.read_text(encoding="utf-8") == "old\n"
    assert approver.requests == []
    denial = next(message for message in agent.messages if isinstance(message, ToolMessage))
    assert denial.content == "error: plan mode cannot change the workspace"
    assert "write_file" not in {tool.name for tool in agent._schemas}


def test_plan_to_agent_can_be_denied(tmp_path: Path):
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "switch_mode",
                        "args": {"mode": "agent", "reason": "apply the plan"},
                        "id": "m1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Staying in plan mode."),
        ]
    )
    approver = RecordingApprover(allow=False)
    agent = CodingAgent(model, tmp_path, approver, mode="plan")
    assert agent.run_turn("go ahead") == "Staying in plan mode."
    assert agent.mode == "plan"
    assert approver.requests[0].kind == "mode"
    denial = next(message for message in agent.messages if isinstance(message, ToolMessage))
    assert denial.content == "error: user denied this action"


def test_agent_to_plan_does_not_ask(tmp_path: Path):
    target = tmp_path / "hello.txt"
    target.write_text("old\n", encoding="utf-8")
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "switch_mode",
                        "args": {"mode": "plan", "reason": "design first"},
                        "id": "m1",
                        "type": "tool_call",
                    }
                ],
            ),
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
            AIMessage(content="Left it alone."),
        ]
    )
    approver = RecordingApprover(allow=True)
    agent = CodingAgent(model, tmp_path, approver)
    assert agent.run_turn("think first") == "Left it alone."
    assert approver.requests == []
    assert agent.mode == "plan"
    assert target.read_text(encoding="utf-8") == "old\n"


def test_always_allow_skips_the_next_prompt(tmp_path: Path):
    from codie.permissions import remember_request

    target = tmp_path / "hello.txt"
    target.write_text("old\n", encoding="utf-8")

    class Remember:
        def __init__(self) -> None:
            self.asks = 0

        def ask(self, request) -> bool:
            self.asks += 1
            remember_request(tmp_path, request)
            return True

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
    approver = Remember()
    agent = CodingAgent(model, tmp_path, approver)
    assert agent.run_turn("rewrite") == "once"
    assert approver.asks == 1
    assert agent.run_turn("again") == "twice"
    assert approver.asks == 1
    assert target.read_text(encoding="utf-8") == "newer\n"


def test_guideline_is_added_to_the_next_prompt(tmp_path: Path):
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "save_guideline",
                        "args": {"text": "run tests with pytest -q"},
                        "id": "g1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Remembered."),
        ]
    )
    agent = CodingAgent(model, tmp_path, RecordingApprover(allow=True))
    assert agent.run_turn("please remember that") == "Remembered."
    assert "run tests with pytest -q" in _text(model.prompts[1])
    assert "run tests with pytest -q" not in _text([model.prompts[0][0]])


def test_tool_schemas_are_strict(tmp_path: Path):
    from codie.tools import ToolContext

    tools = ToolContext(tmp_path, DenyAll()).langchain_tools()
    by_name = {tool.name: tool for tool in tools}
    assert set(by_name) == {
        "list_files",
        "read_file",
        "find_files",
        "grep",
        "write_file",
        "edit_file",
        "run_command",
        "web_search",
        "read_skill",
        "switch_mode",
        "save_guideline",
    }
    for tool in tools:
        schema = tool.args_schema.model_json_schema()
        assert schema.get("additionalProperties") is False
        assert "approved" not in schema.get("properties", {})
        required = set(schema.get("required", []))
        properties = schema.get("properties", {})
        for field, spec in properties.items():
            if "default" not in spec:
                assert field in required
