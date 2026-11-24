"""LangGraph loop. The model asks for tools; this module decides whether they run."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from codie.approvals import ApprovalRequest, Approver, NullSink, StepSink
from codie.guards import MAX_ROUNDS_DEFAULT, plan_command
from codie.prompts import system_prompt
from codie.tools import ToolContext

_STEP_LIMIT = "error: step limit reached before this tool ran"
_THINKING = re.compile(r"<thinking>.*?</thinking>\s*", re.DOTALL | re.IGNORECASE)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    round_count: int
    approvals: dict[str, bool]


class CodingAgent:
    """One conversation. Switching models keeps the message list and starts a new graph."""

    def __init__(
        self,
        model,
        workspace: Path,
        approver: Approver,
        sink: StepSink | None = None,
        max_rounds: int = MAX_ROUNDS_DEFAULT,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.approver = approver
        self.sink = sink or NullSink()
        self.max_rounds = max_rounds
        self.messages: list = []
        self.last_rounds = 0
        self._schemas = ToolContext(self.workspace, approver, NullSink()).langchain_tools()
        self._raw_model = model
        self.model = _bind(model, self._schemas)
        self.graph = self._compile()

    def set_model(self, model) -> None:
        self._raw_model = model
        self.model = _bind(model, self._schemas)
        self.graph = self._compile()

    def reset(self) -> None:
        self.messages = []
        self.last_rounds = 0

    def run_turn(self, text: str) -> str:
        thread_id = str(uuid.uuid4())
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.max_rounds * 4 + 8,
        }
        payload: Any = {
            "messages": [*self.messages, HumanMessage(content=text)],
            "round_count": 0,
            "approvals": {},
        }
        while True:
            result = self.graph.invoke(payload, config)
            pending = _pending_interrupts(self.graph, config, result)
            if pending:
                payload = Command(resume=_decisions(pending, self.approver))
                continue
            values = _state_values(self.graph, config, result)
            self.messages = list(values.get("messages", []))
            self.last_rounds = int(values.get("round_count", 0))
            return _final_text(self.messages)

    def complete(self, messages: list) -> AIMessage:
        """Call the model, streaming tokens when the reply is a final answer."""
        stream = getattr(self.model, "stream", None)
        if stream is None:
            return _as_ai_message(self.model.invoke(messages))
        chunks: list = []
        pieces: list[str] = []
        try:
            for chunk in stream(messages):
                piece = message_text(chunk)
                if piece:
                    pieces.append(piece)
                chunks.append(chunk)
        except Exception:
            if chunks:
                raise
            return _as_ai_message(self.model.invoke(messages))
        if not chunks:
            return _as_ai_message(self.model.invoke(messages))
        if len(chunks) == 1 and type(chunks[0]) is AIMessage:
            message = chunks[0]
        else:
            merged = chunks[0]
            for chunk in chunks[1:]:
                merged = merged + chunk
            message = _as_ai_message(merged)
        if _tool_calls(message):
            self.sink.drop_tokens()
            return message
        shown = visible_text("".join(pieces) if pieces else message_text(message))
        if shown:
            self.sink.token(shown)
        if isinstance(message.content, str):
            message.content = visible_text(message.content)
        return message

    def _compile(self):
        agent = self

        def agent_node(state: AgentState) -> dict:
            prompt = [SystemMessage(content=system_prompt(agent.workspace)), *state["messages"]]
            response = agent.complete(prompt)
            return {"messages": [response], "round_count": state["round_count"] + 1}

        def approve_node(state: AgentState) -> dict:
            quiet = ToolContext(agent.workspace, agent.approver, NullSink())
            requests: list[dict[str, str]] = []
            for call in _tool_calls(state["messages"][-1]):
                if not _needs_approval(agent.workspace, call["name"], call["args"]):
                    continue
                preview = _preview(quiet, call)
                if isinstance(preview, str):
                    continue
                requests.append(preview.to_dict())
            if not requests:
                return {"approvals": {}}
            # One interrupt for the whole round so the tools node runs only once.
            decisions = interrupt({"requests": requests})
            if not isinstance(decisions, dict):
                decisions = {}
            return {"approvals": {str(key): bool(value) for key, value in decisions.items()}}

        def tools_node(state: AgentState) -> dict:
            context = ToolContext(agent.workspace, agent.approver, agent.sink)
            approvals = state.get("approvals") or {}
            outputs = []
            for call in _tool_calls(state["messages"][-1]):
                approved = approvals[call["id"]] if call["id"] in approvals else None
                result = context.dispatch(call["name"], call["args"], approved=approved)
                outputs.append(ToolMessage(content=result, tool_call_id=call["id"]))
            return {"messages": outputs}

        def close_node(state: AgentState) -> dict:
            extras = []
            for call in _tool_calls(state["messages"][-1]):
                extras.append(ToolMessage(content=_STEP_LIMIT, tool_call_id=call["id"]))
            extras.append(AIMessage(content="Stopped: step limit reached for this question."))
            agent.sink.step("<- step limit reached")
            return {"messages": extras}

        def route(state: AgentState) -> str:
            calls = _tool_calls(state["messages"][-1])
            if not calls:
                return "end"
            if state["round_count"] >= agent.max_rounds:
                return "close"
            if any(_needs_approval(agent.workspace, call["name"], call["args"]) for call in calls):
                return "approve"
            return "tools"

        builder = StateGraph(AgentState)
        builder.add_node("agent", agent_node)
        builder.add_node("approve", approve_node)
        builder.add_node("tools", tools_node)
        builder.add_node("close", close_node)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            route,
            {"end": END, "close": "close", "approve": "approve", "tools": "tools"},
        )
        builder.add_edge("approve", "tools")
        builder.add_edge("tools", "agent")
        builder.add_edge("close", END)
        return builder.compile(checkpointer=InMemorySaver())


def visible_text(text: str) -> str:
    """Drop provider thinking tags. The human sees the answer, not the scratch pad."""
    return _THINKING.sub("", text).strip()


def message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


def _bind(model, schemas):
    bind = getattr(model, "bind_tools", None)
    if bind is None:
        return model
    return bind(schemas)


def _as_ai_message(message) -> AIMessage:
    if type(message) is AIMessage:
        return message
    return AIMessage(
        content=getattr(message, "content", "") or "",
        tool_calls=list(getattr(message, "tool_calls", None) or []),
        id=getattr(message, "id", None),
    )


def _tool_calls(message) -> list[dict[str, Any]]:
    raw = getattr(message, "tool_calls", None) or []
    calls: list[dict[str, Any]] = []
    for index, call in enumerate(raw):
        if isinstance(call, dict):
            name = str(call.get("name", ""))
            args = call.get("args") or {}
            call_id = str(call.get("id") or f"call_{index}")
        else:
            name = str(getattr(call, "name", ""))
            args = getattr(call, "args", {}) or {}
            call_id = str(getattr(call, "id", None) or f"call_{index}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": name, "args": args, "id": call_id})
    return calls


def _needs_approval(workspace: Path, name: str, args: dict) -> bool:
    if name == "write_file":
        return True
    if name != "run_command":
        return False
    plan = plan_command(workspace, str(args.get("command", "")))
    return not isinstance(plan, str) and plan.needs_approval


def _preview(context: ToolContext, call: dict) -> ApprovalRequest | str:
    if call["name"] == "write_file":
        return context.preview_write(
            call["id"],
            str(call["args"].get("path", "")),
            str(call["args"].get("content", "")),
        )
    return context.preview_pytest(call["id"], str(call["args"].get("command", "")))


def _pending_interrupts(graph, config, result) -> list:
    if isinstance(result, dict) and result.get("__interrupt__"):
        return list(result["__interrupt__"])
    snapshot = graph.get_state(config)
    found = []
    for task in getattr(snapshot, "tasks", ()) or ():
        found.extend(getattr(task, "interrupts", ()) or ())
    return found


def _state_values(graph, config, result) -> dict:
    if isinstance(result, dict) and "messages" in result:
        return result
    return dict(graph.get_state(config).values)


def _decisions(pending, approver: Approver) -> dict[str, bool]:
    decisions: dict[str, bool] = {}
    for item in pending:
        value = item.value if hasattr(item, "value") else item
        requests = value.get("requests", []) if isinstance(value, dict) else []
        for raw in requests:
            request = ApprovalRequest.from_dict(raw)
            decisions[request.call_id] = bool(approver.ask(request))
    return decisions


def _final_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not _tool_calls(message):
            return visible_text(message_text(message))
    return ""
