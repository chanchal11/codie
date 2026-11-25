"""MCP tools go through the same approval path. The fake client never starts a process."""

from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from codie.agent import CodingAgent
from codie.mcp_client import EmptyMcpClient, McpToolInfo, StdioMcpClient, expose_name
from tests.support import ScriptedModel


class RecordingApprover:
    def __init__(self, allow: bool) -> None:
        self.allow = allow
        self.requests = []

    def ask(self, request) -> bool:
        self.requests.append(request)
        return self.allow


class FakeMcp:
    def __init__(self) -> None:
        self.calls = []

    def discover(self):
        info = McpToolInfo(
            exposed_name="mcp_demo_echo",
            server="demo",
            tool="echo",
            description="Echo text",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
        return [info], []

    def call(self, server: str, tool: str, arguments: dict) -> str:
        self.calls.append((server, tool, arguments))
        return f"echo {arguments.get('text', '')}"


def test_expose_name_is_a_tool_token():
    assert expose_name("demo.server", "do-thing") == "mcp_demo_server_do_thing"


def test_empty_config_discovers_nothing():
    tools, errors = StdioMcpClient().discover()
    assert tools == []
    assert errors == []
    assert EmptyMcpClient().call("demo", "echo", {}).startswith("error:")


def test_mcp_call_requires_approval(tmp_path: Path):
    fake = FakeMcp()
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "mcp_demo_echo",
                        "args": {"text": "hi"},
                        "id": "m1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The tool said echo hi."),
        ]
    )
    approver = RecordingApprover(allow=True)
    agent = CodingAgent(model, tmp_path, approver, mcp_client=fake)
    assert "mcp_demo_echo" in {tool.name for tool in agent._schemas}
    assert agent.run_turn("echo hi") == "The tool said echo hi."
    assert approver.requests[0].kind == "mcp"
    assert approver.requests[0].path == "demo/echo"
    assert fake.calls == [("demo", "echo", {"text": "hi"})]
    tool_message = next(message for message in agent.messages if isinstance(message, ToolMessage))
    assert "echo hi" in tool_message.content


def test_mcp_is_blocked_in_plan_mode(tmp_path: Path):
    fake = FakeMcp()
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "mcp_demo_echo",
                        "args": {"text": "hi"},
                        "id": "m1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="I could not call it."),
        ]
    )
    approver = RecordingApprover(allow=True)
    agent = CodingAgent(model, tmp_path, approver, mode="plan", mcp_client=fake)
    assert agent.run_turn("echo") == "I could not call it."
    assert approver.requests == []
    assert fake.calls == []
    tool_message = next(message for message in agent.messages if isinstance(message, ToolMessage))
    assert tool_message.content == "error: plan mode cannot change the workspace"
